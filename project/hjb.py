# -*- coding: utf-8 -*-
# project/hjb.py
from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import numpy as _np

from .config import SimConfig


# ---------- RNG ----------
def _make_rng(seed: Optional[int]) -> _np.random.Generator:
    """SeedSequence 기반 default_rng 생성 (seed=None이면 OS 엔트로피)."""
    if seed is None:
        return _np.random.default_rng()
    ss = _np.random.SeedSequence(int(seed))
    return _np.random.default_rng(ss)


# ---------- Small helpers ----------
def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _crra_u(c: _np.ndarray, gamma: float = 3.0) -> _np.ndarray:
    """벡터화된 CRRA 효용(γ≈1이면 로그 근사)."""
    c = _np.maximum(_np.asarray(c, dtype=float), 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return _np.log(c)
    return (c ** (1.0 - gamma) - 1.0) / (1.0 - gamma)


def _interp1d_grid(x_grid: _np.ndarray, y_on_grid: _np.ndarray, xq: _np.ndarray) -> _np.ndarray:
    """
    등간격이 아닐 수 있는 1D grid에서 선형보간(클램프 포함).
    x_grid: (N,), y_on_grid: (N,), xq: (M,)
    """
    xg = _np.asarray(x_grid, dtype=float)
    yg = _np.asarray(y_on_grid, dtype=float)
    xq = _np.asarray(xq, dtype=float)
    idx = _np.clip(_np.searchsorted(xg, xq) - 1, 0, xg.size - 2)
    xl = xg[idx]
    xr = xg[idx + 1]
    w = _np.where(xr > xl, (xq - xl) / (xr - xl), 0.0)
    return (1.0 - w) * yg[idx] + w * yg[idx + 1]


def _gauss_hermite_shocks(mu: float, sigma: float, n: int = 32) -> Tuple[_np.ndarray, _np.ndarray]:
    """
    N(mu, sigma^2)에 대한 기댓값을 몬테카를로 없이 정확히 근사하는
    Gauss-Hermite 구적 노드(shocks)와 가중치(weights)를 반환한다.

    사용법: E[f(X)] ≈ np.dot(weights, f(shocks))   (weights의 합은 1)

    노드 수 n은 20~40 정도면 부드러운 함수에 대해 몬테카를로 수천~수만 개
    표본보다 훨씬 정확하다(샘플링 노이즈 자체가 없음). n이 너무 크면(수백 이상)
    오히려 Hermite 다항식 계수의 수치적 불안정성이 생길 수 있어 상한을 둔다.
    """
    n_eff = int(max(2, min(int(n), 64)))
    z, wgt = _np.polynomial.hermite_e.hermegauss(n_eff)  # weight function e^{-z^2/2}
    shocks = mu + max(0.0, float(sigma)) * z
    weights = wgt / _np.sqrt(2.0 * _np.pi)  # 합이 1이 되도록 정규화
    return shocks, weights


# ---------- HJB ----------
class HJBSolver:
    """
    Backward DP on (t, W) with discrete controls (q, w).

    q-grid: 5 pts = [0, 0.25*q4, 0.5*q4, 0.75*q4, q4], q4 = 월환산 4%룰
    w-grid: cfg.hjb_w_grid (없으면 [0, .25, .5, .75, 1]∩[0,w_max]), w_min_dev 필터 적용

    Terminal CVaR (RU-dual):
      V_T(W) = - λ * [ η + (1/(1-α)) * max(F - W - η, 0) ]
      (λ<=0이면 η 탐색 생략; η=0 고정)

    Hedge: cfg.hedge / hedge_mode / hedge_sigma_k / hedge_cost를 간단 규칙으로 반영
    Fee: 월 수수료 φ_m = m['phi_m'] (t 시점의 W_t 기준)

    기댓값 계산(Nshock):
      cfg.hjb_expectation == "mc" 이면 기존 몬테카를로 방식(회귀테스트/비교용).
      그 외(기본값)에는 Gauss-Hermite 구적법을 사용해 샘플링 노이즈 없이
      기댓값을 계산한다. 노드 수는 hjb_Nshock 값을 그대로 재사용하되
      내부적으로 64개로 캡한다(구적법은 노드가 많다고 더 정확해지지 않음).
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.m = cfg.monthly()

        # --- grids ---
        # NOTE (2026-07): 균일 격자(np.linspace)를 그대로 쓰면 실제 인출경로가
        # 거의 항상 머무는 저자산 구간(W~0~2)의 해상도가 너무 낮아져서, HJB가
        # W_max=10 등 넓은 범위를 균일하게 나눌 때 저자산 구간에서 위험자산
        # 비중을 0으로 잘못 수렴시키는 이산화 아티팩트가 발생함을 확인했다
        # (rule 기반 고정정책 w=0.30보다 EU가 낮게 나오는 역설의 근본 원인).
        # 해결: 저자산 구간(W_focus 이하)에 격자점 대부분을 몰아주고, 그 위쪽은
        # 성기게(안전망 목적) 배치하는 2단 비균일 격자로 변경.
        W_min = float(getattr(cfg, "hjb_W_min", 0.0) or 0.0)
        W_max = float(getattr(cfg, "hjb_W_max", 2.0) or 2.0)
        W_n   = int(getattr(cfg, "hjb_W_grid", 33) or 33)
        W_focus = float(getattr(cfg, "hjb_W_focus", min(2.0, W_max)) or min(2.0, W_max))
        focus_frac = float(getattr(cfg, "hjb_W_focus_frac", 0.75) or 0.75)

        if W_focus is not None and 0.0 < W_focus < W_max and W_n >= 4:
            n_focus = max(2, int(round(max(2, W_n) * focus_frac)))
            n_tail = max(2, max(2, W_n) - n_focus + 1)  # +1: 접합점 중복 제거용
            dense = _np.linspace(W_min, W_focus, n_focus)
            sparse = _np.linspace(W_focus, W_max, n_tail)[1:]  # 접합점(W_focus) 중복 제거
            self.W_grid = _np.concatenate([dense, sparse])
        else:
            self.W_grid = _np.linspace(W_min, W_max, max(2, W_n))
        self.T = int(getattr(cfg, "horizon_years", 35)) * int(getattr(cfg, "steps_per_year", 12))

        # --- w actions (dedup + filter + clamp to w_max) ---
        w_max = float(getattr(cfg, "w_max", 1.0) or 1.0)
        base_w = getattr(cfg, "hjb_w_grid", None)
        if base_w is None:
            base_w = [0.0, 0.25, 0.5, 0.75, 1.0]
        wa = [min(max(float(w), 0.0), w_max) for w in base_w]
        w_min_dev = float(getattr(cfg, "w_min_dev", 0.0) or 0.0)
        wa = [w for w in wa if w >= w_min_dev]
        if not wa:
            wa = [0.0, min(0.5, w_max), w_max]
        self.w_actions = _np.array(sorted(set(wa)), dtype=float)

        # --- q actions (5-pt grid around 4% rule, with q_floor clamp later) ---
        # [2026-07] 기존엔 격자 상한이 q4(4%룰 월환산)로 하드코딩되어 있어서,
        # 진짜 무제약 최적 소비율이 4%보다 높은지 검증할 방법이 없었다.
        # hjb_q_max_mult(기본 1.0=기존과 동일)로 상한을 조절할 수 있게 확장.
        spm = int(getattr(cfg, "steps_per_year", 12) or 12)
        q4 = 1.0 - (1.0 - 0.04) ** (1.0 / max(1, spm))
        q_max_mult = float(getattr(cfg, "hjb_q_max_mult", 1.0) or 1.0)
        q_max = q4 * max(0.0, q_max_mult)
        self.q_actions = _np.array([0.0, 0.25 * q_max, 0.5 * q_max, 0.75 * q_max, q_max], dtype=float)

        # --- preferences (utility scale & gamma for tie-breaking consistency) ---
        self.gamma = float(getattr(cfg, "crra_gamma", 3.0) or 3.0)
        self.beta  = float(self.m.get("beta_m", 1.0) or 1.0)

        # --- risk process params (monthly) ---
        self.mu    = float(self.m.get("mu_m", 0.0) or 0.0)
        self.rf    = float(self.m.get("rf_m", 0.0) or 0.0)
        sigma_m = self.m.get("sigma_m", None)
        if sigma_m is None:
            # fallback: 연 18%를 월로, 또는 cfg.sigma_annual 사용
            sig_ann = float(getattr(cfg, "sigma_annual", 0.18) or 0.18)
            spm = float(getattr(cfg, "steps_per_year", 12) or 12)
            self.sigma = float(sig_ann) / _np.sqrt(spm)
        else:
            self.sigma = float(sigma_m)

        # --- hedge mapping (env 규칙 요약 반영) ---
        if str(getattr(cfg, "hedge", getattr(cfg, "hedge_on", "off"))).lower() == "on":
            mode = str(getattr(cfg, "hedge_mode", "sigma")).lower()
            if mode == "mu":
                # 기대수익 haircut
                self.mu = self.mu - float(getattr(cfg, "hedge_cost", 0.0) or 0.0)
            elif mode == "sigma":
                k = float(getattr(cfg, "hedge_sigma_k", 0.0) or 0.0)
                self.sigma = max(0.0, self.sigma * (1.0 - k))
            elif mode in ("downside", "down"):
                # 간단 다운사이드 완화: 평균 영향 없고 변동만 소폭 축소
                k = float(getattr(cfg, "hedge_sigma_k", 0.0) or 0.0)
                self.sigma = max(0.0, self.sigma * (1.0 - 0.5 * k))

        # --- CVaR terminal ---
        self.alpha = float(getattr(cfg, "alpha", 0.95) or 0.95)
        self.lam   = float(getattr(cfg, "lambda_term", 0.0) or 0.0)
        self.F     = float(getattr(cfg, "F_target", 1.0) or 1.0)

        # --- 유증동기(bequest) terminal utility ---
        # [2026-07 신규구현] 기존엔 project/env.py(사용 안 하는 죽은 코드)에만
        # bequest_kappa/bequest_gamma가 연결되어 있었고, HJB의 가치함수 계산에는
        # 전혀 반영되지 않았다(즉 유증동기=0으로 강제된 것과 동일). 여기서는
        # 종단시점 자산 W_T에 대해 CRRA 형태의 유증효용
        #     bequest_kappa * (W_T^(1-bequest_gamma) - 1) / (1-bequest_gamma)
        # 을 기존 CVaR 페널티에 추가로 더해, 소비 효용과 별개로 "남기는 자산" 자체에서도
        # 효용을 얻도록 한다. bequest_kappa=0(기본값)이면 완전히 기존과 동일하게 동작한다.
        self.bequest_kappa = float(getattr(cfg, "bequest_kappa", 0.0) or 0.0)
        self.bequest_gamma = float(getattr(cfg, "bequest_gamma", 1.0) or 1.0)

        # --- 국민연금(소득대체율 ρ) 외생소득 ---
        # [2026-07 신규구현, 2026-07 개정] 소득대체율 ρ는 정의상 "은퇴 전 소득 대비
        # 연금액 비율"인데, 본 모델은 자산(W)만 정규화된 상태변수로 쓰므로, ρ를
        # 반영하려면 "은퇴 전 연소득이 초기자산(W0) 대비 몇 배였는지"를 나타내는
        # 환산배율이 하나 더 필요하다.
        # [개정] 은퇴직전 소득 기준값은 가계 평균소득이 아니라 "국민연금 A값"
        # (전체 가입자 평균소득월액, 2025년 309만원/월)으로 확정 — A값은 국민연금
        # 급여 산정에 실제 쓰이는 제도상 대표소득이라 ρ 정의와 더 정합적이다.
        # X ≈ 3.69 = 2025년 가계금융복지조사 가구 평균 금융자산(1.37억, 부동산 제외)
        #            / A값 연환산(3,708만원).
        # ρ 정책실험 확정구간: {0.20, 0.30, 0.40} (실질 소득대체율 기준, 한정림·이항석
        # 2013 / 한국일보 2025.4 근거). pension_income_mult(=X)가 클수록 "은퇴 전
        # 소득 대비 자산이 넉넉했다"는 뜻이 되어 국민연금이 상대적으로 덜 중요해진다.
        # 국민연금이 상대적으로 덜 중요해진다.
        self.pension_rho = float(getattr(cfg, "pension_rho", 0.0) or 0.0)
        self.pension_income_mult = float(getattr(cfg, "pension_income_mult", 3.692) or 3.692)
        self.pension_claim_age = float(getattr(cfg, "pension_claim_age", 65.0) or 65.0)
        self.age0 = float(getattr(cfg, "age0", 55) or 55)

        # 월간 국민연금 소득(W0=1.0 정규화 단위). 수급개시 이전 구간은 0.
        Y_month = 0.0
        if self.pension_rho > 0.0 and self.pension_income_mult > 0.0:
            Y_month = self.pension_rho / (self.pension_income_mult * 12.0)
        claim_month_idx = max(0, int(round((self.pension_claim_age - self.age0) * spm)))
        self.Y_sched = _np.zeros(self.T, dtype=float)
        if claim_month_idx < self.T:
            self.Y_sched[claim_month_idx:] = Y_month

        # --- 종신연금(1회성 매입) 외생소득 ---
        # [2026-07 신규] θ(연금전환비율)는 t=0에서 1회 선택되는 결정으로 취급한다.
        # 이미 runner.annuity_wiring.setup_annuity_overlay()가 HJBSolver 생성 전에
        # 실행되어 cfg.y_ann(월 지급액, 사망률 기반 a_factor로 환산됨)과 cfg.W0(연금
        # 매입 후 잔여자산)을 계산해 두므로, 별도 메커니즘을 새로 만들지 않고 이 값을
        # 그대로 재사용한다(기존에 검증된 계산 경로를 이중화하지 않기 위함).
        # θ 자체의 최적화(여러 후보값 비교)는 --ann_alpha를 바꿔가며 CLI를 반복
        # 실행하고 EU를 비교하는 외부 그리드서치로 수행한다.
        ann_y_month = float(getattr(cfg, "y_ann", 0.0) or 0.0)
        if ann_y_month > 0.0:
            self.Y_sched = self.Y_sched + ann_y_month

        # --- fee (monthly) ---
        self.phi_m = float(self.m.get("phi_m", 0.0) or 0.0)

        # --- sampling for expectation ---
        self.Nshock = int(getattr(cfg, "hjb_Nshock", 32) or 32)

        # --- tie-break tolerance ---
        self.tie_eps = 1e-9

    # --- Terminal CVaR penalty (RU-dual) ---
    def _cvar_T(self, W: _np.ndarray, eta: float) -> _np.ndarray:
        if self.lam <= 0.0:
            return _np.zeros_like(W, dtype=float)
        inv = 1.0 / max(1e-12, (1.0 - self.alpha))
        return self.lam * (eta + inv * _np.maximum(self.F - W - float(eta), 0.0))

    # --- Terminal bequest utility (CRRA on terminal wealth) ---
    def _bequest_U(self, W: _np.ndarray) -> _np.ndarray:
        if self.bequest_kappa <= 0.0:
            return _np.zeros_like(W, dtype=float)
        return self.bequest_kappa * _crra_u(W, self.bequest_gamma)

    # --- value backup for (W, q, w) ---
    def _backup_val(self, V_next: _np.ndarray, W: float, q: float, w: float, shocks: _np.ndarray,
                     weights: Optional[_np.ndarray] = None, Y_t: float = 0.0) -> float:
        # 1) consume
        #    [2026-07] 국민연금 등 외생소득 Y_t가 있으면 "총소비 = 개인인출(q*W) + Y_t"로
        #    CRRA 효용을 계산하되, 포트폴리오에서 실제로 빠져나가는 금액은 q*W뿐이다
        #    (연금소득은 외부에서 들어와 그대로 소비되고 자산에는 반영되지 않는다고 가정).
        q_w = q * W
        c = q_w + Y_t
        W_net = max(W - q_w, 0.0)

        # 2) returns
        gross = 1.0 + (w * shocks) + ((1.0 - w) * self.rf)
        W_next = _np.clip(W_net * gross - self.phi_m * W, 0.0, float(self.W_grid[-1]))

        # 3) continuation value
        #    - weights가 있으면(Gauss-Hermite 구적) 가중평균으로 노이즈 없는 정확한 기댓값 계산
        #    - 없으면(기존 몬테카를로 방식) 단순평균으로 폴백
        Vn = _interp1d_grid(self.W_grid, V_next, W_next)
        if weights is not None:
            Vn_mean = float(_np.dot(Vn, weights))
        else:
            Vn_mean = float(Vn.mean())
        return float(_crra_u([c], self.gamma)[0] + self.beta * Vn_mean)

    def solve(self, seed: Optional[int] = None, rng: Optional[_np.random.Generator] = None) -> Dict[str, Any]:
        """
        Returns
        -------
        dict: { "Pi_w": (T×|W|), "Pi_q": (T×|W|), "eta": float, "W_grid": np.ndarray }
        """
        rng_local = rng if rng is not None else _make_rng(seed)

        # 기댓값 계산 방식 선택: 기본은 Gauss-Hermite 구적(노이즈 없음),
        # cfg.hjb_expectation == "mc" 이면 기존 몬테카를로 방식(비교/회귀테스트용).
        use_mc = str(getattr(self.cfg, "hjb_expectation", "quad")).lower() == "mc"
        if use_mc:
            shocks = rng_local.normal(loc=self.mu, scale=max(0.0, self.sigma), size=max(1, self.Nshock))
            weights = None
        else:
            shocks, weights = _gauss_hermite_shocks(self.mu, self.sigma, n=self.Nshock)

        # η 후보 (λ<=0이면 0만)
        eta_values = tuple(getattr(self.cfg, "hjb_eta_grid", (0.0,))) if (self.lam > 0.0) else (0.0,)

        best_eta = 0.0
        best_obj = -1e30
        best_PiW: _np.ndarray | None = None
        best_PiQ: _np.ndarray | None = None

        spm = int(getattr(self.cfg, "steps_per_year", 12) or 12)
        q_floor = float(getattr(self.cfg, "q_floor", 0.0) or 0.0)
        w_max   = float(getattr(self.cfg, "w_max", 1.0) or 1.0)

        for eta in eta_values:
            V   = _np.zeros((self.T + 1, self.W_grid.size), dtype=float)
            PiW = _np.zeros((self.T,     self.W_grid.size), dtype=float)
            PiQ = _np.zeros((self.T,     self.W_grid.size), dtype=float)

            # terminal
            V[self.T, :] = self._bequest_U(self.W_grid) - self._cvar_T(self.W_grid, float(eta))

            # backward
            for t in reversed(range(self.T)):
                for i, W in enumerate(self.W_grid):
                    # q grid with floor & floor_on(f_min_real)
                    q_min = q_floor
                    if bool(getattr(self.cfg, "floor_on", False)) and float(getattr(self.cfg, "f_min_real", 0.0)) > 0.0 and W > 0.0:
                        q_min = max(q_min, min(1.0, float(getattr(self.cfg, "f_min_real")) / float(W)))
                    q_grid = _np.maximum(self.q_actions, q_min)

                    best_val = -1e30
                    bw = float(self.w_actions[0]); bq = float(q_grid[0])

                    for w in self.w_actions:
                        w = min(max(float(w), 0.0), w_max)
                        for q in q_grid:
                            val = self._backup_val(V[t + 1, :], float(W), float(q), float(w), shocks, weights, Y_t=float(self.Y_sched[t]))
                            # tie-break: 값 동률이면 더 보수적 w(작은 w) 선택
                            if (val > best_val + self.tie_eps) or (abs(val - best_val) <= self.tie_eps and w < bw):
                                best_val = float(val)
                                bw, bq = float(w), float(q)

                    V[t, i]   = best_val
                    PiW[t, i] = bw
                    PiQ[t, i] = bq

            # 초기자산 W≈1.0에서의 값으로 η 선택
            j = int(_np.clip(_np.searchsorted(self.W_grid, 1.0), 0, self.W_grid.size - 1))
            obj = float(V[0, j])
            if obj > best_obj:
                best_obj = obj
                best_eta = float(eta)
                best_PiW = PiW
                best_PiQ = PiQ

        if best_PiW is None or best_PiQ is None:
            # 드문 fallback: 균등 정책
            const_w = float(min(max(self.w_actions.mean(), 0.0), w_max))
            const_q = float(self.q_actions[-1])
            best_PiW = _np.full((self.T, self.W_grid.size), const_w, dtype=float)
            best_PiQ = _np.full((self.T, self.W_grid.size), const_q, dtype=float)

        return {"Pi_w": best_PiW, "Pi_q": best_PiQ, "eta": best_eta, "W_grid": self.W_grid}

# =====================================================================
# [2026-07 참고] 종신연금 전환비율(theta, ann_alpha)의 1회성 결정 최적화 방법
# =====================================================================
# theta 자체를 HJB가 매 시점 반복 선택하는 진짜 3번째 통제변수로 만들면
# 상태공간이 (W, 누적연금소득) 2차원으로 늘어나 계산량이 크게 증가한다.
# 본 연구는 "1회성 결정" 스코프로 한정하여 (Milevsky 2007의 부분적/점진적
# 연금화 논의와도 부합), theta는 t=0에서 한 번만 선택되는 것으로 취급한다.
#
# 구현: runner.annuity_wiring.setup_annuity_overlay()가 HJBSolver 생성 전에
# cfg.y_ann(월 지급액)과 cfg.W0(연금 매입 후 잔여자산)을 계산해 두면,
# HJBSolver.__init__은 이를 그대로 읽어 Y_sched에 반영한다(위 참조).
#
# theta 자체의 최적화는 여러 --ann_alpha 후보값(예: 0, 0.1, ..., 0.6)으로
# CLI를 반복 실행하고, 각 실행의 EU(evaluate() 결과)를 비교하는 외부
# 그리드서치로 수행한다. 이는 bequest_kappa/pension_rho 민감도 분석과
# 동일한 패턴이며, 이미 검증된 evaluate() 파이프라인을 그대로 재사용한다.
