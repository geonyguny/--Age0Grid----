# -*- coding: utf-8 -*-
# project/env/retirement_env.py
from __future__ import annotations
from typing import Tuple, Any, Optional, Dict
from types import SimpleNamespace

import os
import math
import numpy as np
import pandas as pd

# ===== (옵션) 행동편향: 효용-레이어 훅 =====
try:
    from ..policy.behavioral import (  # type: ignore
        BehavioralSpec,                # type: ignore
        distort_utility, habit_utility # type: ignore
    )
except Exception:
    BehavioralSpec = None  # type: ignore

    def distort_utility(u: float, *, ref: float = 0.0, spec=None) -> float:  # type: ignore
        return float(u)

    def habit_utility(u: float, prev_u: float, *, spec=None) -> float:  # type: ignore
        return float(u)


# ---------- helpers ----------
def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _crra_u(c: float, gamma: float) -> float:
    """
    CRRA utility:
      u(c) = log c                if gamma ≈ 1
           = (c^{1-gamma}-1)/(1-g) otherwise
    c는 최소 1e-12로 바운드하여 수치 폭주 방지.
    """
    c = max(float(c), 1e-12)
    if abs(float(gamma) - 1.0) < 1e-12:
        return math.log(c)
    return (c ** (1.0 - float(gamma)) - 1.0) / (1.0 - float(gamma))


def _to_monthly_rate_like(x: np.ndarray) -> np.ndarray:
    """지수면 전월대비율로, 이미 월간률이면 그대로."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    is_index_like = (np.nanmax(x) > 5.0) or (np.nanmedian(np.abs(x)) > 0.2)
    if is_index_like and x.size >= 2:
        r = np.empty_like(x, dtype=float)
        r[1:] = x[1:] / x[:-1] - 1.0
        r[0] = r[1] if x.size > 1 and np.isfinite(x[1]) else 0.0
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        return r
    return np.nan_to_num(x.astype(float), nan=0.0, posinf=0.0, neginf=0.0)


def _nan_guard_arr(
    a: np.ndarray,
    *,
    fill: float = 0.0,
    clip: Tuple[float, float] | None = None
) -> np.ndarray:
    """배열 NaN/Inf 정화(+선택적 클리핑)."""
    arr = np.nan_to_num(np.asarray(a, dtype=float), nan=fill, posinf=fill, neginf=fill)
    if clip is not None:
        lo, hi = float(clip[0]), float(clip[1])
        arr = np.clip(arr, lo, hi)
    if not np.isfinite(arr).all():
        arr = np.zeros_like(arr, dtype=float)
    return arr


def _safe_float(x: Any, default: float = 0.0) -> float:
    """스칼라 NaN/Inf 방호."""
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return float(default)
    except Exception:
        return float(default)


def _load_market_arrays(csv_path: str, use_real_rf: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    CSV columns (required): date, risky_nom, tbill_nom, cpi
    반환: risky, safe, cpi_rate (모두 월간률). use_real_rf='on'이면 CPI로 실질화.
    """
    try:
        data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        names = {n.lower() for n in (data.dtype.names or ())}
        required = {"risky_nom", "tbill_nom", "cpi"}
        if not required.issubset(names):
            raise ValueError(f"CSV missing columns: {sorted(required - names)}")

        risky_nom = np.asarray(data["risky_nom"], dtype=float)
        tbill_nom = np.asarray(data["tbill_nom"], dtype=float)
        cpi_col   = np.asarray(data["cpi"], dtype=float)

        cpi_rate = _to_monthly_rate_like(np.nan_to_num(cpi_col, nan=0.0))

        if str(use_real_rf).lower() == "on":
            risky = (1.0 + np.nan_to_num(risky_nom, nan=0.0)) / (1.0 + cpi_rate) - 1.0
            safe  = (1.0 + np.nan_to_num(tbill_nom, nan=0.0)) / (1.0 + cpi_rate) - 1.0
        else:
            risky = np.nan_to_num(risky_nom, nan=0.0)
            safe  = np.nan_to_num(tbill_nom, nan=0.0)

        return _nan_guard_arr(risky), _nan_guard_arr(safe), _nan_guard_arr(cpi_rate)
    except Exception:
        # 안전한 최종 fallback (parametric i.i.d.) + CPI=0%
        rng = np.random.default_rng(7)
        risky = rng.normal(0.06 / 12, 0.18 / np.sqrt(12), size=6000)
        safe  = np.full(6000, 0.02 / 12)
        cpi_rate = np.zeros(6000, dtype=float)
        return risky, safe, cpi_rate


# ---------- Environment ----------
class RetirementEnv:
    """
    Retirement decumulation env (월 리밸런스)

    state(obs): ndarray([t_norm, W_t])
    action: (q, w) ∈ [0,1]^2
      - q: 소비 비율
      - w: 위험자산 비중(나머지는 안전자산)

    보상 구조(기본값 기준):
      base_reward_t = u_scale * u_CRRA(c_t; gamma) + survive_bonus
      reward_t = base_reward_t            (beta=1, 패널티 계수=0일 때)

    선택적 이론형 옵션(논문 목적함수 맞추기용):
      - beta              : env 내부 시간할인계수 (기본 1.0 → 변경 없으면 기존과 동일)
      - lambda_ruin       : 파산(wealth=0) 시 패널티 계수
      - lambda_shortfall  : 만기 단기부족 L_T = max(F_target - W_T, 0)에 대한 패널티 계수
      - ruin_penalty_once : 파산 패널티를 최초 1회만 줄지 여부("on"/"off")

    최종 보상(옵션 포함):
      reward_t = beta^t * ( base_reward_t
                            - lambda_ruin * I{ruin}
                            - lambda_shortfall * L_T )
    """

    # --- cfg/kwargs 통합 접근자 ---
    @staticmethod
    def _get(cfg: Any, kwargs: dict, name: str, default: Any) -> Any:
        # 1) kwargs 우선
        if kwargs and (name in kwargs):
            return kwargs[name]

        if cfg is not None:
            # 2) 정확히 같은 이름
            if hasattr(cfg, name):
                return getattr(cfg, name)

            # 3) CamelCase 변형 (ann_alpha -> AnnAlpha)
            if name:
                alt1 = name[0].upper() + name[1:]
                if hasattr(cfg, alt1):
                    return getattr(cfg, alt1)

            # 4) 언더스코어 제거 후 대소문자 무시 비교
            base = name.replace("_", "").lower()
            for attr in dir(cfg):
                if attr.startswith("_"):
                    continue
                a_low = attr.lower()
                if a_low == name.lower() or a_low == base:
                    return getattr(cfg, attr)

        # 5) 어느 쪽에서도 못 찾으면 기본값
        return default

    def __init__(self, cfg: Any = None, **kwargs):
        # --- time / wealth / prefs ---
        self.steps_per_year = int(max(1, self._get(cfg, kwargs, "steps_per_year", 12)))
        self.T  = int(max(1, self._get(cfg, kwargs, "horizon_years", 45))) * self.steps_per_year
        self.W0 = _safe_float(self._get(cfg, kwargs, "W0", 1.0), 1.0)
        self.w_max = _safe_float(self._get(cfg, kwargs, "w_max", 1.0), 1.0)

        # floor_on + f_min_real 지원 / q_floor None→0.0
        _qf = self._get(cfg, kwargs, "q_floor", 0.0)
        spm = int(getattr(self, "steps_per_year", 12) or 12)
        self.q_floor_base = _safe_float(0.0 if _qf is None else _qf, 0.0) / spm  # 연→월 환산 저장
        self.floor_on = str(self._get(cfg, kwargs, "floor_on", "off") or "off").lower() == "on"
        self.f_min_real = _safe_float(self._get(cfg, kwargs, "f_min_real", 0.0), 0.0)

        # ---- 수수료 체계: 연 펀드보수(지속) / annuity 프런트로딩(1회) 분리 ----
        # (A) 포트폴리오 운용보수: 월차감 기준(잔여 금융자산 대상)
        self.fee_annual  = _safe_float(self._get(cfg, kwargs, "fee_annual", 0.004), 0.004)
        self.fee_m       = self.fee_annual / self.steps_per_year

        # (B) 종신연금 프런트 로딩(가입 시 1회): annuity 오버레이에서만 사용
        self.phi_adval   = _safe_float(self._get(cfg, kwargs, "phi_adval", 0.0), 0.0)

        # “연금 전환 후 펀드보수 0” 정책 스위치(기본 on)
        self.ann_zero_fee_after_purchase = str(
            self._get(cfg, kwargs, "ann_zero_fee_after_purchase", "on") or "on"
        ).lower() == "on"

        # 효용 관련 파라미터 (CRRA)
        self.survive_bonus = _safe_float(self._get(cfg, kwargs, "survive_bonus", 0.0), 0.0)
        self.u_scale = _safe_float(self._get(cfg, kwargs, "u_scale", 0.05), 0.05)
        self.gamma   = _safe_float(self._get(cfg, kwargs, "crra_gamma", 3.0), 3.0)

        # [NEW] HJB형 목적함수 옵션: 할인계수 및 패널티 계수
        self.beta = _safe_float(self._get(cfg, kwargs, "beta", 1.0), 1.0)
        self.lambda_ruin = _safe_float(self._get(cfg, kwargs, "lambda_ruin", 0.0), 0.0)
        self.lambda_shortfall = _safe_float(self._get(cfg, kwargs, "lambda_shortfall", 0.0), 0.0)
        self.ruin_penalty_once = str(
            self._get(cfg, kwargs, "ruin_penalty_once", "on") or "on"
        ).lower() == "on"
        # 내부 할인 계수 상태(에피소드별로 reset 시 1.0에서 시작)
        self._disc_factor: float = 1.0
        # 파산 패널티를 한 번만 줄 경우, 이미 부과했는지 여부
        self._ruin_penalized: bool = False

        # --- [ANN] annuity overlay params ---
        # 기본값 'auto': ann_alpha>0 이면 자동으로 annuity init 시도, 'off'일 때만 강제 비활성화
        self.ann_on    = str(self._get(cfg, kwargs, "ann_on", "auto") or "auto").lower()
        self.ann_alpha = _safe_float(self._get(cfg, kwargs, "ann_alpha", 0.0), 0.0)
        self.ann_L     = _safe_float(self._get(cfg, kwargs, "ann_L", 0.0), 0.0)
        self.ann_d     = int(self._get(cfg, kwargs, "ann_d", 0) or 0)
        self.ann_index = str(self._get(cfg, kwargs, "ann_index", "real") or "real")
        self.y_ann = max(0.0, _safe_float(self._get(cfg, kwargs, "y_ann", 0.0), 0.0))
        self.ann_purchased = False
        self.ann_P = 0.0
        self.ann_a_factor = 0.0

        # --- meta ---
        self.age0 = int(self._get(cfg, kwargs, "age0", 55))
        self.age_years = float(self.age0)

        # --- market sources ---
        self.market_mode = str(self._get(cfg, kwargs, "market_mode", "bootstrap") or "bootstrap").lower()
        self.market_csv  = str(self._get(cfg, kwargs, "market_csv", "") or "")
        self.bootstrap_block = int(max(1, self._get(cfg, kwargs, "bootstrap_block", 24)))
        self.use_real_rf = str(self._get(cfg, kwargs, "use_real_rf", "on") or "on").lower()

        # --- IID 파라미터(테스트 스텁 호환) ---
        self.mu_risky    = _safe_float(self._get(cfg, kwargs, "mu_risky", 0.06/12), 0.06/12)
        self.sigma_risky = max(
            0.0,
            _safe_float(self._get(cfg, kwargs, "sigma_risky", 0.18/np.sqrt(12)), 0.18/np.sqrt(12))
        )
        self.r_safe_fix = _safe_float(self._get(cfg, kwargs, "r_safe", 0.02/12), 0.02/12)

        # --- hedge params ---
        self.hedge = str(self._get(cfg, kwargs, "hedge", "off") or "off").lower()
        self.hedge_mode = str(self._get(cfg, kwargs, "hedge_mode", "sigma") or "sigma").lower()
        self.hedge_sigma_k = float(np.clip(self._get(cfg, kwargs, "hedge_sigma_k", 0.50), 0.0, 1.0))

        premium_annual = self._get(cfg, kwargs, "hedge_premium", self._get(cfg, kwargs, "hedge_cost", 0.005))
        self.hedge_premium_annual = _safe_float(max(0.0, float(premium_annual)), 0.0)
        self.hedge_premium_m = self.hedge_premium_annual / self.steps_per_year
        self.hedge_cost   = self.hedge_premium_annual  # alias
        self.hedge_cost_m = self.hedge_premium_m

        self.hedge_tx_annual = _safe_float(self._get(cfg, kwargs, "hedge_tx", 0.0), 0.0)
        self.hedge_tx_m = self.hedge_tx_annual / self.steps_per_year

        # --- mortality / rf ---
        self.life_table: Optional[pd.DataFrame] = None
        self.mort_table_df: Optional[pd.DataFrame] = None
        self.r_f_real_annual: Optional[float] = None
        self._init_mortality_if_any(cfg, kwargs)

        # --- (옵션) 행동편향 사양 입력(효용-레이어) ---
        self._bh_spec = getattr(cfg, "behavioral_spec", None)
        self._prev_u_for_habit = 0.0  # 습관효용용 이전 효용(왜곡 후)

        # --- seeding / RNG ---
        seeds = self._get(cfg, kwargs, "seeds", [0]) or [0]
        seed_attr = self._get(cfg, kwargs, "seed", None)
        base = int(seed_attr) if seed_attr is not None else int(seeds[0])
        from numpy.random import SeedSequence, default_rng
        self._ss = SeedSequence(base)
        self.rng = default_rng(self._ss)
        self._path_counter = 0  # increments each reset

        # --- preload market arrays / injection support ---
        inj_ret = self._get(cfg, kwargs, "data_ret_series", None)
        inj_rf  = self._get(cfg, kwargs, "data_rf_series", None)
        inj_cpi = self._get(cfg, kwargs, "data_cpi", None)

        if inj_ret is not None and inj_rf is not None:
            self._risky    = _nan_guard_arr(inj_ret)
            self._safe     = _nan_guard_arr(inj_rf)
            self._cpi_rate = _nan_guard_arr(
                inj_cpi if inj_cpi is not None else np.zeros_like(self._risky)
            )
        elif self.market_mode == "bootstrap" and os.path.exists(self.market_csv):
            self._risky, self._safe, self._cpi_rate = _load_market_arrays(
                self.market_csv, self.use_real_rf
            )
        else:
            # IID 파라메트릭: 테스트 인자(mu_risky, sigma_risky, r_safe) 반영
            self._risky    = self.rng.normal(self.mu_risky, self.sigma_risky, size=6000)
            self._safe     = np.full(6000, self.r_safe_fix)
            self._cpi_rate = np.zeros(6000, dtype=float)

        # ---- yearly flags ----
        self.is_new_year: bool = True
        self.cpi_yoy: float = 0.0

        # --- [NEW] Terminal shortfall target (옵션) ---
        self.F_target = _safe_float(self._get(cfg, kwargs, "F_target", 0.0), 0.0)

        # 초기화 완료 후 기본 reset
        self.reset()

    # ----- mortality init -----
    def _init_mortality_if_any(self, cfg: Any, kwargs: dict):
        """
        생명표/실질 rf 로드.

        규칙:
        - mortality='on' 인 경우, mort_table 이 반드시 지정되어야 하며
          로딩 실패 시 예외를 발생시킨다 (무조건 반영).
        - mortality != 'on' 이면 life_table 을 사용하지 않는다.
        """
        mort_flag = str(self._get(cfg, kwargs, "mortality", "off") or "off").lower()
        mt = self._get(cfg, kwargs, "mort_table", None)

        self.life_table = None
        self.mort_table_df = None

        if mort_flag != "on":
            # mortality off → 생명표 미사용 (테스트/단순 실험용)
            rf_from_cfg = self._get(cfg, kwargs, "r_f_real_annual", 0.02)
            self.r_f_real_annual = _safe_float(rf_from_cfg, 0.02)
            return

        # 여기부터는 mortality='on' 이므로 mort_table 반드시 필요
        if not isinstance(mt, str) or not mt.strip():
            raise ValueError("[env:mort] mortality='on' 인데 mort_table 이 지정되지 않았습니다.")

        mt_str = str(mt).strip()
        loaded = False

        # 1) 문자열 토큰(BASE/COHORT/COHORT_YYYY) 직접 인식
        if mt_str.lower().startswith(("base", "cohort")) and not os.path.exists(mt_str):
            try:
                from ..annuity.mortality_gm import load_life_table  # type: ignore

                lt = load_life_table(
                    mt_str,
                    sex=str(self._get(cfg, kwargs, "sex", "M") or "M").upper(),
                    age0=int(self._get(cfg, kwargs, "age0", 55)),
                    horizon_years=int(self._get(cfg, kwargs, "horizon_years", 45)),
                    steps_per_year=int(self._get(cfg, kwargs, "steps_per_year", 12)),
                    annual_improvement=_safe_float(self._get(cfg, kwargs, "mort_imp", 0.01), 0.01),
                )
                self.life_table = lt.copy()
                self.mort_table_df = lt.copy()
                loaded = True
                print(f"[env:mort] life_table token='{mt_str}' loaded: rows={len(lt)}")
            except Exception as e:
                raise RuntimeError(f"[env:mort] token mort_table 로딩 실패({mt_str}): {e}") from e

        # 2) 파일 경로(csv)
        elif os.path.exists(mt_str):
            try:
                df = pd.read_csv(mt_str)
                has_age = "age" in df.columns
                has_px  = any(c in df.columns for c in ["px", "Px"])
                has_qx  = "qx" in df.columns
                has_mf  = all(c in df.columns for c in ["male", "female"])
                if not has_age:
                    raise ValueError("mort_table must contain 'age' column")

                if has_qx:
                    lt = df[["age", "qx"]].copy()
                elif has_px:
                    col = "px" if "px" in df.columns else "Px"
                    lt = df[["age", col]].rename(columns={col: "px"}).copy()
                elif has_mf:
                    sex = self._get(cfg, kwargs, "sex", "M")
                    col = "male" if str(sex).upper() == "M" else "female"
                    lt = df[["age", col]].rename(columns={col: "qx"}).copy()
                else:
                    raise ValueError(
                        "expected ('age' + 'qx') or ('age' + 'px/Px') or ('age' + 'male,female')"
                    )

                lt["age"] = lt["age"].astype(int)
                lt = lt.sort_values("age").reset_index(drop=True)
                self.life_table = lt
                self.mort_table_df = lt
                loaded = True
                print(f"[env:mort] life_table loaded: rows={len(lt)}, cols={list(lt.columns)}")
            except Exception as e:
                raise RuntimeError(
                    f"[env:mort] mort_table 파일 로딩/파싱 실패: path={mt_str}, err={e}"
                ) from e
        else:
            raise FileNotFoundError(
                f"[env:mort] mortality='on' 이지만 mort_table 경로/토큰을 인식할 수 없습니다: '{mt_str}'"
            )

        if not loaded or self.life_table is None or len(self.life_table) == 0:
            raise RuntimeError(
                f"[env:mort] mortality='on' 이지만 life_table 로딩에 실패했습니다 (mort_table={mt_str})."
            )

        # 실질 rf 설정
        rf_from_cfg = self._get(cfg, kwargs, "r_f_real_annual", 0.02)
        self.r_f_real_annual = _safe_float(rf_from_cfg, 0.02)

    # ----- market path builders -----
    def _bootstrap_path(
        self,
        T: int,
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """블록 부트스트랩으로 길이 T의 (risky, safe, cpi) 경로 생성."""
        N = len(self._risky)
        B = max(1, self.bootstrap_block)
        r = np.empty(T, float)
        s = np.empty(T, float)
        p = np.empty(T, float)
        t = 0
        hi = max(1, N - B + 1)
        while t < T:
            start = int(rng.integers(0, hi))
            take = min(B, T - t)
            r[t : t + take] = self._risky[start : start + take]
            s[t : t + take] = self._safe[start : start + take]
            p[t : t + take] = self._cpi_rate[start : start + take]
            t += take
        return _nan_guard_arr(r), _nan_guard_arr(s), _nan_guard_arr(p)

    # ----- returns drawer (테스트 monkeypatch 대상) -----
    def _draw_returns(self) -> Tuple[float, float]:
        """
        기본 구현: 사전 생성된 경로에서 현재 t의 (r_risky, r_safe)를 반환.
        테스트에서는 monkeypatch로 이 메서드를 교체한다.
        """
        rr = _safe_float(self.path_risky[self.t], 0.0)
        rf = _safe_float(self.path_safe[self.t], 0.0)
        return rr, rf

           # ----- annuity init at reset -----
    def _annuity_init_if_any(self):
        """
        ann_alpha > 0 이고 ann_on 이 'off' 가 아니면
        t=0에서 1회 종신연금을 매입하고, 매 스텝당 real 지급 y_ann 을 설정합니다.

        - life_table 이 있으면 생존확률을 사용
        - life_table 이 없으면 S_t = 1 (사망위험 무시)로 두고 단순 Yaari 스타일 annuity 구현
        """

        # 기본 상태 리셋
        self.y_ann = 0.0
        self.ann_purchased = False
        self.ann_P = 0.0
        self.ann_a_factor = 0.0

        # 1) annuity 사용 여부 체크 -----------------------------------
        alpha = _safe_float(getattr(self, "ann_alpha", 0.0), 0.0)
        if alpha <= 0.0:
            return

        mode = str(getattr(self, "ann_on", "auto") or "auto").lower()
        if mode == "off":
            return
        if mode == "auto":
            # alpha > 0 이면 자동 on
            mode = "on"
            setattr(self, "ann_on", "on")

        # 현재 t=0 투자계정 기준자산
        W0_now = float(
            _safe_float(
                getattr(self, "W", getattr(self, "W0", 0.0)),
                getattr(self, "W0", 0.0),
            )
        )
        if W0_now <= 0.0:
            return

        # 시간축 / 연령 ------------------------------------------------
        S = int(getattr(self, "steps_per_year", 12) or 12)
        T = int(getattr(self, "T", S * 30) or (S * 30))
        age0 = int(getattr(self, "age0", 55))
        horizon_years = int(math.ceil(T / float(S))) + 1

        # 2) 연간 생존확률 px_year 구축 ------------------------------
        lt = getattr(self, "life_table", None)
        px_year = None
        ages = None

        if isinstance(lt, pd.DataFrame) and not lt.empty and "age" in lt.columns:
            lt_use = lt.copy()
            lt_use["age"] = lt_use["age"].astype(int)
            lt_use = lt_use.sort_values("age").reset_index(drop=True)

            ages = lt_use["age"].to_numpy(dtype=int)
            if "px" in lt_use.columns:
                px_year = np.clip(
                    lt_use["px"].astype(float).to_numpy(), 0.0, 1.0
                )
            elif "qx" in lt_use.columns:
                qx_year = np.clip(
                    lt_use["qx"].astype(float).to_numpy(), 0.0, 1.0
                )
                px_year = 1.0 - qx_year

        if px_year is None or ages is None or px_year.size == 0:
            # mortality off 또는 생명표 구조 불일치 → px=1.0 가정
            ages = np.arange(age0, age0 + horizon_years, dtype=int)
            px_year = np.ones_like(ages, dtype=float)

        # age → px_year 맵 구성 --------------------------------------
        max_age_needed = age0 + math.ceil(T / S) + 1
        age_px_map: Dict[int, float] = {}
        j = 0
        last_px = 1.0
        for age in range(int(ages.min()), max_age_needed + 1):
            while j + 1 < len(ages) and ages[j + 1] <= age:
                j += 1
            last_px = float(px_year[j]) if j < len(px_year) else last_px
            age_px_map[age] = float(np.clip(last_px, 0.0, 1.0))

        # 3) 스텝별 생존확률 S_t (t=0..T), S_0=1 ----------------------
        step_surv = np.empty(T + 1, dtype=float)
        step_surv[0] = 1.0
        for t in range(1, T + 1):
            age_t = age0 + (t - 1) // S
            px_yr = float(age_px_map.get(age_t, 1.0))
            # 연간 px → 스텝별 q_m: px = (1 - q_m)^S
            q_m = 1.0 - px_yr ** (1.0 / float(S))
            q_m = float(np.clip(q_m, 0.0, 1.0))
            step_surv[t] = step_surv[t - 1] * (1.0 - q_m)

        # 4) 할인인자 (real rf 사용) ---------------------------------
        r_f_real = float(
            _safe_float(getattr(self, "r_f_real_annual", 0.02), 0.02)
        )
        if r_f_real <= -0.99:
            r_f_real = 0.0

        disc = np.array(
            [(1.0 + r_f_real) ** (-t / float(S)) for t in range(1, T + 1)],
            dtype=float,
        )

        # 지급시점별 EPV 가중치: 생존확률 * 할인인자 ------------------
        surv_pay = step_surv[1:]  # 길이 T
        ann_factor = float(np.sum(surv_pay * disc))
        if (not np.isfinite(ann_factor)) or ann_factor <= 0.0:
            return

        # 5) 공정가 조건으로 y_ann 및 초기 W 설정 --------------------
        P = float(alpha * W0_now)  # 종신연금 프리미엄
        if P <= 0.0:
            return

        y_step = float(P / ann_factor)  # 매 스텝당 연금지급액

        # 투자계정은 (1 - alpha) * W0_now 로 시작
        self.W = max(W0_now - P, 0.0)

        self.y_ann = max(y_step, 0.0)
        self.ann_purchased = self.y_ann > 0.0
        self.ann_P = P
        self.ann_a_factor = ann_factor

        # ann_index, phi_adval, ann_L 등은 현재 단계에서는 가격에 반영하지 않음
        # (annuity 효과가 확실히 보인 뒤, 필요 시 로딩/인덱싱 로직을 추가)
        return

    # ----- API -----
    def reset(self, W0: Optional[float] = None, seed: Optional[int] = None):
        """
        Supports reset(W0=...), reset(seed=...), reset(W0=..., seed=...).
        경로 RNG는 SeedSequence.spawn()으로 경로별 분기.
        """
        # 외부 seed가 주어지면 base 갱신
        if seed is not None:
            from numpy.random import SeedSequence, default_rng
            self._ss = SeedSequence(int(seed))
            self.rng = default_rng(self._ss)

        # 새 에피소드용 RNG 분기
        child = self._ss.spawn(1)[0]
        from numpy.random import default_rng
        self.rng = default_rng(child)
        self._path_counter += 1

        # 초기 상태
        self.t = 0
        self.W = _safe_float(self.W0 if W0 is None else W0, self.W0)
        self.age_years = float(self.age0)
        self.is_new_year = True
        self.cpi_yoy = 0.0

        # [ANN] 초기화
        #  - y_ann 포함해서 상태를 리셋하고,
        #  - 그 다음에 annuity overlay를 적용해 W0 → (1 - alpha)·W0, y_ann 설정
        self.y_ann = 0.0
        self.ann_purchased = False
        self.ann_P = 0.0
        self.ann_a_factor = 0.0

# --- ANN DEBUG: before annuity init ---
        try:
            tag = getattr(self, "tag", "NA")
            env_name = type(self).__name__
            ann_alpha_dbg = getattr(self, "ann_alpha", None)
            ann_on_dbg = getattr(self, "ann_on", None)
            life_loaded = getattr(self, "life_table", None) is not None

            print(
                f"[ANN-DBG-RESET][before] env={env_name} tag={tag} "
                f"ann_alpha={ann_alpha_dbg} ann_on={ann_on_dbg} "
                f"life_table_loaded={life_loaded} W_before={self.W}"
            )
        except Exception:
            pass

        # t=0 연금 매입(옵션)
        self._annuity_init_if_any()

        # --- ANN DEBUG: after annuity init ---
        try:
            tag = getattr(self, "tag", "NA")
            env_name = type(self).__name__
            print(
                f"[ANN-DBG-RESET][after ] env={env_name} tag={tag} "
                f"W_after={self.W} y_ann={getattr(self, 'y_ann', None)} "
                f"ann_P={getattr(self, 'ann_P', None)} "
                f"ann_a_factor={getattr(self, 'ann_a_factor', None)}"
            )
        except Exception:
            pass

        # (효용-레이어) 이전 효용 초기화
        self._prev_u_for_habit = 0.0

        # [NEW] 이론형 목적함수 관련 상태 초기화
        self._disc_factor = 1.0
        self._ruin_penalized = False

        # 경로 생성
        if self.market_mode == "bootstrap":
            self.path_risky, self.path_safe, self.path_cpi = self._bootstrap_path(self.T, self.rng)
        else:
            # IID 파라메트릭 (cfg의 mu/sigma/r_safe 반영)
            self.path_risky = self.rng.normal(self.mu_risky, self.sigma_risky, size=self.T)
            self.path_safe  = np.full(self.T, self.r_safe_fix)
            self.path_cpi   = np.zeros(self.T, dtype=float)

        # t=0 연금 매입(옵션)
        self._annuity_init_if_any()

        return self._obs().astype(np.float32)

    def _obs(self) -> np.ndarray:
        """정규화 시간과 현재 자산을 ndarray(float32)로 반환."""
        t_norm = (self.t / max(1, self.T - 1)) if self.T > 1 else 0.0
        return np.asarray(
            [_safe_float(t_norm, 0.0), _safe_float(self.W, 0.0)],
            dtype=np.float32,
        )

    # ▶ 테스트 호환: state 프로퍼티 (env.state.W 등 접근)
    @property
    def state(self):
        return SimpleNamespace(
            W=float(_safe_float(self.W, 0.0)),
            t=int(self.t),
            age_years=float(self.age_years),
        )

    # ----- floor helper -----
    def _q_min_now(self) -> float:
        """동적 소비하한: floor_on이면 f_min_real/W와 q_floor_base 중 큰 값을 적용."""
        if not self.floor_on:
            return float(self.q_floor_base or 0.0)
        W_now = max(_safe_float(self.W, 0.0), 0.0)
        if W_now <= 0.0:
            return 0.0
        spm = int(getattr(self, "steps_per_year", 12) or 12)
        f_min_per_step = _safe_float(self.f_min_real, 0.0) / spm  # 연간→월
        dyn = max(0.0, min(1.0, f_min_per_step / W_now))
        return max(float(self.q_floor_base or 0.0), dyn)

    # ----- hedge -----
    def _apply_hedge(self, r_risky_raw: float, r_safe: float, w: float) -> Tuple[float, bool]:
        """헤지 모드/강도에 따른 r_risky_eff와 hedge_active 플래그."""
        k = float(np.clip(_safe_float(getattr(self, "hedge_sigma_k", 0.0), 0.0), 0.0, 1.0))
        mode = str(getattr(self, "hedge_mode", "sigma")).lower()

        rr = _safe_float(r_risky_raw, 0.0)
        rf = _safe_float(r_safe, 0.0)
        hedge_active = False
        r_risky_eff = rr

        if str(getattr(self, "hedge", "off")).lower() == "on":
            if mode == "sigma":
                r_risky_eff = (1.0 - k) * rr + k * rf
                hedge_active = True
            elif mode in ("downside", "down"):
                if rr < 0.0 and k > 0.0:
                    r_risky_eff = (1.0 - k) * rr  # 하락만 완화
                    hedge_active = True

        # 상승 이득 증폭 금지 / 손실 뒤집기 금지
        if rr < 0:
            r_risky_eff = min(0.0, r_risky_eff)
        else:
            r_risky_eff = min(r_risky_eff, rr)

        return _safe_float(r_risky_eff, 0.0), bool(hedge_active)

    def step(self, *args, **kwargs):
        """
        Supports:
          - step(q=..., w=...)
          - step(q, w)
          - step([q, w]) / step((q, w)) / step(np.array([q, w]))
          - step({"q":..., "w":...})   ← dict 지원
        Returns: (obs, reward, done, info)  ← 항상 4-튜플
        """
        # ---- parse (q, w) ----
        if len(args) == 1 and not kwargs:
            act = args[0]
            if isinstance(act, dict):
                try:
                    q = float(act.get("q", 0.0))
                    w = float(act.get("w", 0.0))
                except Exception as e:
                    raise TypeError("step(dict) expects keys {'q','w'}") from e
            else:
                try:
                    q = float(act[0])
                    w = float(act[1])
                except Exception as e:
                    raise TypeError(
                        "step(action) expects sequence-like [q,w] or dict {'q','w'}"
                    ) from e
        elif len(args) >= 2:
            q = float(args[0])
            w = float(args[1])
        else:
            if "q" in kwargs and "w" in kwargs:
                q = float(kwargs["q"])
                w = float(kwargs["w"])
            else:
                raise TypeError("step requires (q, w) or action=[q,w] or dict {'q','w'}")

        q = _safe_float(q, 0.0)
        w = _safe_float(w, 0.0)

        # 에피소드 종료 후 호출 방지 → 4-튜플로 즉시 반환
        if self.t >= self.T:
            info = {
                "W_T": float(_safe_float(self.W, 0.0)),
                "done_reason": "already_ended",
                "truncated": False,
            }
            return self._obs(), 0.0, True, info

        # 현재 스텝 시작 자산(수수료 기준 보존)
        W_start = max(_safe_float(self.W, 0.0), 0.0)

        # 1) clipping ----------------------------------------------------
        q_min = self._q_min_now()
        q = max(q_min, _clip01(q))
        w = _clip01(min(w, _safe_float(getattr(self, "w_max", 1.0), 1.0)))

        # 2) consumption ----------------------------------------------------
        y_ann = _safe_float(getattr(self, "y_ann", 0.0), 0.0)
        c = _safe_float(y_ann + q * W_start, 0.0)
        W_after_c = max(W_start - q * W_start, 0.0)  # 연금은 외부 유입

        # 3) returns (+hedge) ----------------------------------------------
        r_risky_raw, r_safe = self._draw_returns()
        r_risky_eff, hedge_active = self._apply_hedge(r_risky_raw, r_safe, w)
        r_port = _safe_float(w * r_risky_eff + (1.0 - w) * r_safe, 0.0)

        # 헤지 비용/거래비용 (hedge_active일 때만)
        hc  = _safe_float(getattr(self, "hedge_cost_m", 0.0), 0.0)
        htx = _safe_float(getattr(self, "hedge_tx_m", 0.0), 0.0)
        if hedge_active and hc > 0.0:
            r_port -= w * hc
            if htx > 0.0:
                r_port -= w * htx

        gross = _safe_float(1.0 + r_port, 1.0)
        W_before_fee = _safe_float(W_after_c * gross, W_after_c)

        # 4) fee (월차감, **기준=W_after_c**) -------------------------------
        #  - ann_zero_fee_after_purchase == True 이고 ann_purchased면 수수료 0
        #  - 그렇지 않으면 기본 fee_m 적용
        fee_m_base = _safe_float(getattr(self, "fee_m", 0.0), 0.0)
        if self.ann_zero_fee_after_purchase and bool(getattr(self, "ann_purchased", False)):
            fee_m_eff = 0.0
        else:
            fee_m_eff = fee_m_base

        fee = _safe_float(fee_m_eff * W_after_c, 0.0)
        self.W = max(_safe_float(W_before_fee - fee, 0.0), 0.0)

        # ----- 효용(유틸리티) 계산 + 행동편향 훅 ---------------------------
        base_u = _crra_u(c, _safe_float(getattr(self, "crra_gamma", 3.0), 3.0))
        u_eff = base_u
        spec = getattr(self, "_bh_spec", None)
        if spec is not None and getattr(spec, "on", False):
            try:
                u1 = distort_utility(base_u, ref=0.0, spec=spec)
                u2 = habit_utility(u1, self._prev_u_for_habit, spec=spec)
                self._prev_u_for_habit = float(u1)
                u_eff = float(u2)
            except Exception:
                u_eff = base_u

        base_reward = (
            _safe_float(getattr(self, "u_scale", 0.0), 0.0)
            * _safe_float(u_eff, base_u)
            + _safe_float(getattr(self, "survive_bonus", 0.0), 0.0)
        )

        # advance time ------------------------------------------------------
        self.t += 1
        spm = int(getattr(self, "steps_per_year", 12) or 12)
        self.age_years = float(self.age0) + (self.t / max(1, spm))
        self.is_new_year = (self.t % spm == 0)

        # CPI YoY
        if self.t >= spm:
            window = _nan_guard_arr(self.path_cpi[self.t - spm : self.t], fill=0.0)
            try:
                self.cpi_yoy = float(np.prod(1.0 + window) - 1.0)
            except Exception:
                self.cpi_yoy = 0.0
        else:
            self.cpi_yoy = 0.0

        done = (self.t >= self.T) or (self.W <= 0.0)

        # ----- Bias 신호(액션-레이어 편향 모듈용) -------------------------
        recent_ret = float(r_risky_raw)
        if self.t >= 12:
            win = _nan_guard_arr(self.path_risky[self.t - 12 : self.t], fill=0.0)
            recent_vol = float(np.std(win))
        else:
            recent_vol = 0.0

        # info 구성 (항상 안전한 스칼라)
        info: Dict[str, Any] = {
            "consumption": float(c),
            "y_ann": float(y_ann),
            "ann_on": (self.ann_on == "on"),
            "ann_purchased": bool(self.ann_purchased),
            "ann_P": float(self.ann_P),
            "ann_a_factor": float(self.ann_a_factor),
            "W": float(_safe_float(self.W, 0.0)),
            "q": float(q),
            "q_min": float(q_min),
            "w": float(w),
            "r_risky": float(r_risky_raw),
            "r_risky_eff": float(r_risky_eff),
            "r_safe": float(r_safe),
            "hedge": str(getattr(self, "hedge", "off")).lower(),
            "hedge_mode": str(getattr(self, "hedge_mode", "sigma")).lower(),
            "hedge_active": bool(hedge_active),
            "hedge_k": float(getattr(self, "hedge_sigma_k", 0.0)),
            "fee_m_base": float(fee_m_base),
            "fee_m_eff": float(fee_m_eff),
            "fee": float(fee),
            "cpi_yoy": float(_safe_float(self.cpi_yoy, 0.0)),
            "is_new_year": bool(self.is_new_year),
            "age_years": float(self.age_years),
            "life_table": bool(self.life_table is not None),
            "recent_ret": recent_ret,
            "recent_vol": recent_vol,
            "truncated": False,  # 4-튜플 규약: trunc는 info에만 표시
        }

        L_term = 0.0
        if done:
            W_T = float(_safe_float(self.W, 0.0))
            info.setdefault("W_T", W_T)
            info.setdefault("terminal_wealth", W_T)
            info.setdefault(
                "done_reason",
                "wealth_depleted" if W_T <= 0.0 else "horizon",
            )
            _F = getattr(self, "F_target", None)
            try:
                if _F is not None:
                    F_val = float(_safe_float(_F, 0.0))
                    L_term = float(max(F_val - W_T, 0.0))
                    info.setdefault("L_term", L_term)
            except Exception:
                pass

        # ----- HJB형 패널티 및 시간 할인 반영 -----------------------------
        ruin_flag = bool(self.W <= 0.0)
        info["ruin_flag"] = ruin_flag

        penalty = 0.0

        # (1) 파산 패널티
        lam_r = _safe_float(getattr(self, "lambda_ruin", 0.0), 0.0)
        if lam_r > 0.0 and ruin_flag:
            if (not self.ruin_penalty_once) or (not getattr(self, "_ruin_penalized", False)):
                penalty -= lam_r
                if self.ruin_penalty_once:
                    self._ruin_penalized = True

        # (2) 만기 단기부족 패널티 (있다면)
        lam_s = _safe_float(getattr(self, "lambda_shortfall", 0.0), 0.0)
        if lam_s > 0.0 and L_term > 0.0:
            penalty -= lam_s * L_term

        # (3) 내부 시간 할인(beta) 적용
        beta = _safe_float(getattr(self, "beta", 1.0), 1.0)
        disc_factor = float(getattr(self, "_disc_factor", 1.0))

        reward = base_reward + penalty
        if beta != 1.0:
            reward = reward * disc_factor
            self._disc_factor = disc_factor * beta
        else:
            self._disc_factor = disc_factor  # 유지

        # reward 클리핑
        reward = float(np.clip(reward, -100.0, 100.0))

        # 모니터링용 정보 추가
        info["base_reward"] = float(base_reward)
        info["penalty"] = float(penalty)
        info["beta"] = float(beta)
        info["disc_factor"] = float(self._disc_factor)
 
        return self._obs(), float(reward), bool(done), info
