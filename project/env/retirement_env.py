# -*- coding: utf-8 -*-
"""
퇴직연금 DECUM 환경 (월 리밸런스)

- 상태(observation): obs = [t_norm, W_t]  (float32, 길이 2)
- 행동(action): (q, w) ∈ [0,1]^2
    q: 소비 비율
    w: 위험자산 비중

- 기본 보상:
    base_reward_t = u_scale * u_CRRA(c_t; gamma) + survive_bonus

- HJB/이론형 목적함수(옵션):
    reward_t = beta^t * ( base_reward_t
                          - lambda_ruin * I{ruin}
                          - lambda_shortfall * L_T )

  · L_term = max(F_target - W_T, 0)  (terminal shortfall)

주요 특징
---------
1) 연금(annuity) 오버레이 통합
   - project.annuity.overlay.init_from_sim_cfg 사용
   - mortality='on' & life_table 유효할 때만 t=0에서 annuity 매입
   - self.y_ann, self.ann_purchased, self.ann_P, self.ann_a_factor 설정
   - annuity 매입 후 계정자산은 W_after_ann으로 감소

2) annuity 구매 후 펀드보수 0 정책
   - ann_zero_fee_after_purchase='on' & ann_purchased=True 이면
     이후 fee_m_eff=0 (잔여 계정에는 펀드보수 미부과)

3) HJB형 패널티 및 시간할인
   - beta, lambda_ruin, lambda_shortfall, ruin_penalty_once 적용
   - terminal shortfall L_term 패널티

4) step API 일관화
   - step(q=..., w=...) / step(q, w) / step([q,w]) / step({"q":..,"w":..}) 모두 허용
   - 항상 (obs, reward, done, info) 4-튜플 반환
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ===== (옵션) 행동편향: 효용-레이어 훅 =====
try:
    from ..policy.behavioral import (  # type: ignore
        BehavioralSpec,                # type: ignore
        distort_utility, habit_utility, regret_utility  # type: ignore
    )
except Exception:  # pragma: no cover
    BehavioralSpec = None  # type: ignore

    def distort_utility(u: float, *, ref: float = 0.0, spec=None) -> float:  # type: ignore
        return float(u)

    def habit_utility(u: float, prev_u: float, *, spec=None) -> float:  # type: ignore
        return float(u)

    def regret_utility(u: float, c: float, c_ref: float, *, spec=None) -> float:  # type: ignore
        return float(u)


# [ANN] annuity overlay
try:
    from ..annuity.overlay import init_from_sim_cfg  # type: ignore
except Exception:  # pragma: no cover
    init_from_sim_cfg = None  # type: ignore


# ---------- helpers ----------

def _clip01(x: float) -> float:
    """[0,1] 구간 클리핑."""
    return max(0.0, min(1.0, float(x)))


def _crra_u(c: float, gamma: float) -> float:
    """
    CRRA 효용함수:

        u(c) = log(c)                     (gamma ≈ 1)
             = (c^{1-gamma} - 1)/(1-gamma) (그 외)

    c는 최소 1e-12로 바운드하여 수치 불안정 방지.
    """
    c = max(float(c), 1e-12)
    g = float(gamma)
    if abs(g - 1.0) < 1e-12:
        return math.log(c)
    return (c ** (1.0 - g) - 1.0) / (1.0 - g)


def _to_monthly_rate_like(x: np.ndarray) -> np.ndarray:
    """
    입력이 지수(level)면 전월대비율로, 이미 수익률이면 그대로 반환.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x

    # level-like heuristic
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
    clip: Tuple[float, float] | None = None,
) -> np.ndarray:
    """
    배열의 NaN/Inf를 정리하고, 필요 시 [lo,hi]로 클리핑.
    """
    arr = np.nan_to_num(np.asarray(a, dtype=float), nan=fill, posinf=fill, neginf=fill)
    if clip is not None:
        lo, hi = float(clip[0]), float(clip[1])
        arr = np.clip(arr, lo, hi)
    if not np.isfinite(arr).all():
        arr = np.zeros_like(arr, dtype=float)
    return arr


def _safe_float(x: Any, default: float = 0.0) -> float:
    """
    스칼라 NaN/Inf 방어용 캐스팅.
    """
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return float(default)
    except Exception:
        return float(default)


def _load_market_arrays(csv_path: str, use_real_rf: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    CSV에서 월간 risky/safe/CPI 시계열을 로드.

    요구 컬럼:
        - risky_nom
        - tbill_nom
        - cpi

    use_real_rf='on'이면 CPI를 사용하여 실질 수익률로 변환.
    """
    try:
        data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        names = {n.lower() for n in (data.dtype.names or ())}
        required = {"risky_nom", "tbill_nom", "cpi"}
        if not required.issubset(names):
            missing = sorted(required - names)
            raise ValueError(f"CSV missing columns: {missing}")

        risky_nom = np.asarray(data["risky_nom"], dtype=float)
        tbill_nom = np.asarray(data["tbill_nom"], dtype=float)
        cpi_col = np.asarray(data["cpi"], dtype=float)

        cpi_rate = _to_monthly_rate_like(np.nan_to_num(cpi_col, nan=0.0))

        if str(use_real_rf).lower() == "on":
            risky = (1.0 + np.nan_to_num(risky_nom, nan=0.0)) / (1.0 + cpi_rate) - 1.0
            safe = (1.0 + np.nan_to_num(tbill_nom, nan=0.0)) / (1.0 + cpi_rate) - 1.0
        else:
            risky = np.nan_to_num(risky_nom, nan=0.0)
            safe = np.nan_to_num(tbill_nom, nan=0.0)

        return _nan_guard_arr(risky), _nan_guard_arr(safe), _nan_guard_arr(cpi_rate)
    except Exception:
        # fallback: 간단한 i.i.d. 모형
        rng = np.random.default_rng(7)
        risky = rng.normal(0.06 / 12, 0.18 / np.sqrt(12), size=6000)
        safe = np.full(6000, 0.02 / 12)
        cpi_rate = np.zeros(6000, dtype=float)
        return risky, safe, cpi_rate


class RetirementEnv:
    """
    퇴직연금 인출(decumulation) 환경 (월 리밸런스).

    - CRRA 효용 + 행동편향(옵션)
    - HJB 스타일 시간할인/패널티 옵션(beta, lambda_ruin, lambda_shortfall)
    - 종신연금 오버레이(annuity overlay, ann_alpha > 0 & mortality='on' 시 t=0에서 매입)
    """

    # --- cfg/kwargs 통합 접근자 ---

    @staticmethod
    def _get(cfg: Any, kwargs: dict, name: str, default: Any) -> Any:
        """
        cfg와 kwargs를 통합하여 name에 해당하는 값을 가져온다.
        우선순위:
          1) kwargs[name]
          2) cfg.name / cfg.Name
          3) 언더스코어 제거/소문자 비교
          4) default
        """
        if kwargs and (name in kwargs):
            return kwargs[name]

        if cfg is not None:
            if hasattr(cfg, name):
                return getattr(cfg, name)

            if name:
                alt1 = name[0].upper() + name[1:]
                if hasattr(cfg, alt1):
                    return getattr(cfg, alt1)

            base = name.replace("_", "").lower()
            for attr in dir(cfg):
                if attr.startswith("_"):
                    continue
                a_low = attr.lower()
                if a_low == name.lower() or a_low == base:
                    return getattr(cfg, attr)

        return default

    # ------------------------------------------------------------------ #
    #   초기화
    # ------------------------------------------------------------------ #
    def __init__(self, cfg: Any = None, **kwargs):
        # --- 시간/초기자산/위험한도 ---
        self.steps_per_year = int(max(1, self._get(cfg, kwargs, "steps_per_year", 12)))
        horizon_years = int(max(1, self._get(cfg, kwargs, "horizon_years", 45)))
        self.T = horizon_years * self.steps_per_year

        self.W0 = _safe_float(self._get(cfg, kwargs, "W0", 1.0), 1.0)
        self.w_max = _safe_float(self._get(cfg, kwargs, "w_max", 1.0), 1.0)

        # 소비 floor 설정(q_floor, floor_on, f_min_real)
        _qf = self._get(cfg, kwargs, "q_floor", 0.0)
        self.q_floor_base = _safe_float(0.0 if _qf is None else _qf, 0.0) / self.steps_per_year
        self.floor_on = str(self._get(cfg, kwargs, "floor_on", "off") or "off").lower() == "on"
        self.f_min_real = _safe_float(self._get(cfg, kwargs, "f_min_real", 0.0), 0.0)

        # ---- 수수료 체계: 펀드보수(지속)/연금 부가보험료(1회) 분리 ----
        # (A) 포트폴리오 운용보수: 월차감 기반
        self.fee_annual = _safe_float(self._get(cfg, kwargs, "fee_annual", 0.004), 0.004)
        self.fee_m = self.fee_annual / self.steps_per_year

        # (B) 종신연금 front-loading: annuity overlay에서 φ_adval로 사용
        self.phi_adval = _safe_float(self._get(cfg, kwargs, "phi_adval", 0.0), 0.0)

        # 연금 전환 후 잔여 계정에 대해 펀드보수 0으로 처리할지 여부 (기본 on)
        self.ann_zero_fee_after_purchase = str(
            self._get(cfg, kwargs, "ann_zero_fee_after_purchase", "on") or "on"
        ).lower() == "on"

        # 효용 관련 파라미터
        self.survive_bonus = _safe_float(self._get(cfg, kwargs, "survive_bonus", 0.0), 0.0)
        self.u_scale = _safe_float(self._get(cfg, kwargs, "u_scale", 0.05), 0.05)
        self.gamma = _safe_float(self._get(cfg, kwargs, "crra_gamma", 3.0), 3.0)

        # [NEW] HJB형 목적함수 옵션
        self.beta = _safe_float(self._get(cfg, kwargs, "beta", 1.0), 1.0)
        self.lambda_ruin = _safe_float(self._get(cfg, kwargs, "lambda_ruin", 0.0), 0.0)
        self.lambda_shortfall = _safe_float(self._get(cfg, kwargs, "lambda_shortfall", 0.0), 0.0)
        self.ruin_penalty_once = str(
            self._get(cfg, kwargs, "ruin_penalty_once", "on") or "on"
        ).lower() == "on"

        self._disc_factor: float = 1.0
        self._ruin_penalized: bool = False

        # --- [ANN] annuity overlay params ---
        self.ann_on = str(self._get(cfg, kwargs, "ann_on", "auto") or "auto").lower()
        self.ann_alpha = _safe_float(self._get(cfg, kwargs, "ann_alpha", 0.0), 0.0)
        self.ann_L = _safe_float(self._get(cfg, kwargs, "ann_L", 0.0), 0.0)
        self.ann_d = int(self._get(cfg, kwargs, "ann_d", 0) or 0)
        self.ann_index = str(self._get(cfg, kwargs, "ann_index", "real") or "real")
        self.y_ann = max(0.0, _safe_float(self._get(cfg, kwargs, "y_ann", 0.0), 0.0))
        self.ann_purchased = False
        self.ann_P = 0.0
        self.ann_a_factor = 0.0
        # [FIX 2026-07] 상위(annuity_wiring.setup_annuity_overlay)에서 이미 매입이
        # 끝난 경우를 식별하기 위해 원본 cfg 값을 별도로 보관해 둔다(이중매입 방지용).
        self._cfg_y_ann_precomputed = _safe_float(self._get(cfg, kwargs, "y_ann", 0.0), 0.0)
        self._cfg_ann_P_precomputed = _safe_float(self._get(cfg, kwargs, "ann_P", 0.0), 0.0)
        self._cfg_ann_a_factor_precomputed = _safe_float(self._get(cfg, kwargs, "ann_a_factor", 0.0), 0.0)

        # --- meta ---
        self.age0 = int(self._get(cfg, kwargs, "age0", 55))
        self.age_years = float(self.age0)

        # --- 국민연금(소득대체율 ρ) 외생소득 [2026-07 신규] ---
        # hjb.py의 Y_sched와 동일한 계산식을 순방향 시뮬레이션(evaluate)에도 반영해야,
        # HJB가 "연금을 받을 걸 알고" 덜 인출한 정책을 평가할 때도 그 연금소득이 실제로
        # 소비/효용에 반영되어 정합성이 유지된다. y_ann(민간 즉시연금)과 동일한 방식으로
        # "포트폴리오에서 차감되지 않는 외부소득"으로 취급한다.
        self._pension_rho = float(self._get(cfg, kwargs, "pension_rho", 0.0) or 0.0)
        self._pension_income_mult = float(self._get(cfg, kwargs, "pension_income_mult", 3.692) or 3.692)
        self._pension_claim_age = float(self._get(cfg, kwargs, "pension_claim_age", 65.0) or 65.0)
        _pension_Y_month = 0.0
        if self._pension_rho > 0.0 and self._pension_income_mult > 0.0:
            _pension_Y_month = self._pension_rho / (self._pension_income_mult * 12.0)
        self._pension_Y_month = _pension_Y_month
        self._pension_claim_month_idx = max(
            0, int(round((self._pension_claim_age - self.age0) * self.steps_per_year))
        )

        # --- market sources ---
        self.market_mode = str(self._get(cfg, kwargs, "market_mode", "bootstrap") or "bootstrap").lower()
        self.market_csv = str(self._get(cfg, kwargs, "market_csv", "") or "")
        self.bootstrap_block = int(max(1, self._get(cfg, kwargs, "bootstrap_block", 24)))
        self.use_real_rf = str(self._get(cfg, kwargs, "use_real_rf", "on") or "on").lower()

        # --- IID 파라미터 ---
        # [FIX 2026-07] 기존 코드는 "mu_risky"/"sigma_risky"라는, cfg에 존재하지도 않는
        # 속성명을 찾다가 실패하면 무조건 하드코딩된 기본값(mu=6%/12, sigma=18%/sqrt(12))으로
        # 폴백했다. 그 결과 cfg.mu_annual/cfg.sigma_annual(자산별 프리셋: KR 20%, US 16%,
        # Gold 15% 등)이 iid 모드의 실제 시장 시뮬레이션에 전혀 반영되지 않는 버그가 있었다.
        # (--asset 플래그를 바꿔도 iid 모드 결과가 달라지지 않았던 근본 원인.)
        # 아래처럼 cfg.mu_annual/sigma_annual로부터 월간 mu_risky/sigma_risky 기본값을
        # 먼저 계산한 뒤, 그래도 명시적으로 mu_risky/sigma_risky가 주어지면(override) 그것을
        # 우선하도록 수정한다.
        mu_ann_default = float(self._get(cfg, kwargs, "mu_annual", 0.06) or 0.06)
        sigma_ann_default = float(self._get(cfg, kwargs, "sigma_annual", 0.20) or 0.20)
        _default_mu_risky = (1.0 + mu_ann_default) ** (1.0 / self.steps_per_year) - 1.0
        _default_sigma_risky = sigma_ann_default / np.sqrt(self.steps_per_year)
        self.mu_risky = _safe_float(self._get(cfg, kwargs, "mu_risky", _default_mu_risky), _default_mu_risky)
        self.sigma_risky = max(
            0.0,
            _safe_float(
                self._get(cfg, kwargs, "sigma_risky", _default_sigma_risky),
                _default_sigma_risky,
            ),
        )
        self.r_safe_fix = _safe_float(self._get(cfg, kwargs, "r_safe", 0.02 / 12), 0.02 / 12)

        # --- hedge params ---
        self.hedge = str(self._get(cfg, kwargs, "hedge", "off") or "off").lower()
        self.hedge_mode = str(self._get(cfg, kwargs, "hedge_mode", "sigma") or "sigma").lower()
        self.hedge_sigma_k = float(
            np.clip(self._get(cfg, kwargs, "hedge_sigma_k", 0.50), 0.0, 1.0)
        )

        premium_annual = self._get(
            cfg, kwargs, "hedge_premium", self._get(cfg, kwargs, "hedge_cost", 0.005)
        )
        self.hedge_premium_annual = _safe_float(max(0.0, float(premium_annual)), 0.0)
        self.hedge_premium_m = self.hedge_premium_annual / self.steps_per_year
        self.hedge_cost = self.hedge_premium_annual
        self.hedge_cost_m = self.hedge_premium_m

        self.hedge_tx_annual = _safe_float(self._get(cfg, kwargs, "hedge_tx", 0.0), 0.0)
        self.hedge_tx_m = self.hedge_tx_annual / self.steps_per_year

        # --- mortality / rf ---
        self.life_table: Optional[pd.DataFrame] = None
        self.mort_table_df: Optional[pd.DataFrame] = None
        self.r_f_real_annual: Optional[float] = None
        self._init_mortality_if_any(cfg, kwargs)

        # --- 행동편향 사양(효용 레이어) ---
        self._bh_spec = getattr(cfg, "behavioral_spec", None)
        self._prev_u_for_habit = 0.0

        # --- seeding / RNG ---
        seeds = self._get(cfg, kwargs, "seeds", [0]) or [0]
        seed_attr = self._get(cfg, kwargs, "seed", None)
        base_seed = int(seed_attr) if seed_attr is not None else int(seeds[0])

        from numpy.random import SeedSequence, default_rng

        self._ss = SeedSequence(base_seed)
        self.rng = default_rng(self._ss)
        self._path_counter = 0

        # --- market data injection / preloading ---
        inj_ret = self._get(cfg, kwargs, "data_ret_series", None)
        inj_rf = self._get(cfg, kwargs, "data_rf_series", None)
        inj_cpi = self._get(cfg, kwargs, "data_cpi", None)

        if inj_ret is not None and inj_rf is not None:
            self._risky = _nan_guard_arr(np.asarray(inj_ret, dtype=float))
            self._safe = _nan_guard_arr(np.asarray(inj_rf, dtype=float))
            if inj_cpi is not None:
                self._cpi_rate = _nan_guard_arr(np.asarray(inj_cpi, dtype=float))
            else:
                self._cpi_rate = np.zeros_like(self._risky)
        elif self.market_mode == "bootstrap" and os.path.exists(self.market_csv):
            self._risky, self._safe, self._cpi_rate = _load_market_arrays(
                self.market_csv, self.use_real_rf
            )
        else:
            self._risky = self.rng.normal(self.mu_risky, self.sigma_risky, size=6000)
            self._safe = np.full(6000, self.r_safe_fix)
            self._cpi_rate = np.zeros(6000, dtype=float)

        # --- yearly flags ---
        self.is_new_year: bool = True
        self.cpi_yoy: float = 0.0

        # --- terminal shortfall target(optional) ---
        self.F_target = _safe_float(self._get(cfg, kwargs, "F_target", 0.0), 0.0)

        # 초기화 후 기본 reset
        self.reset()

    # ------------------------------------------------------------------ #
    #   mortality init
    # ------------------------------------------------------------------ #
    def _init_mortality_if_any(self, cfg: Any, kwargs: dict) -> None:
        """
        mortality='on'이면 mort_table(토큰 또는 csv)을 반드시 로드.
        이 경우 life_table이 유효하지 않으면 예외 발생.
        mortality!='on'이면 life_table을 사용하지 않고, r_f_real_annual만 세팅.
        """
        mort_flag = str(self._get(cfg, kwargs, "mortality", "off") or "off").lower()
        mt = self._get(cfg, kwargs, "mort_table", None)

        self.life_table = None
        self.mort_table_df = None

        if mort_flag != "on":
            rf_from_cfg = self._get(cfg, kwargs, "r_f_real_annual", 0.02)
            self.r_f_real_annual = _safe_float(rf_from_cfg, 0.02)
            return

        if not isinstance(mt, str) or not mt.strip():
            raise ValueError("[env:mort] mortality='on' 인데 mort_table 이 지정되지 않았습니다.")

        mt_str = str(mt).strip()
        loaded = False

        # 1) 토큰(BASE/COHORT/COHORT_YYYY 등)
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
                has_px = any(c in df.columns for c in ["px", "Px"])
                has_qx = "qx" in df.columns
                has_mf = all(c in df.columns for c in ["male", "female"])
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

        rf_from_cfg = self._get(cfg, kwargs, "r_f_real_annual", 0.02)
        self.r_f_real_annual = _safe_float(rf_from_cfg, 0.02)

    # ------------------------------------------------------------------ #
    #   market path builders & returns
    # ------------------------------------------------------------------ #
    def _bootstrap_path(
        self,
        T: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        블록 부트스트랩으로 길이 T의 risky/safe/cpi 경로를 생성.
        """
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

    def _draw_returns(self) -> Tuple[float, float]:
        """
        사전 생성된 경로에서 현재 t의 (r_risky, r_safe)를 반환.
        테스트/디버깅 시 monkeypatch 대상.
        """
        rr = _safe_float(self.path_risky[self.t], 0.0)
        rf = _safe_float(self.path_safe[self.t], 0.0)
        return rr, rf

    # ------------------------------------------------------------------ #
    #   annuity overlay at reset
    # ------------------------------------------------------------------ #
    def _annuity_init_if_any(self) -> None:
        """
        ann_alpha > 0 이고 ann_on 이 'off' 가 아니면
        mortality='on' & life_table 유효할 때만 t=0에서 종신연금 매입.

        - annuity overlay(init_from_sim_cfg)를 사용하여
          W, y_ann, ann_P, ann_a_factor를 설정.

        [FIX 2026-07] 기존엔 runner.annuity_wiring.setup_annuity_overlay()가
        (HJB 액터 생성 전에) cfg.W0를 이미 (1-ann_alpha)만큼 줄여놓았는데도,
        여기서 다시 같은 ann_alpha로 한 번 더 매입을 수행하여 자산이 이중으로
        (예: 0.3+0.3 아니라 곱연산으로 0.7×0.7=0.49처럼) 줄어드는 버그가 있었다.
        그 결과 HJB가 가정한 잔여자산·연금소득과 실제 순방향 시뮬레이션의 값이
        서로 달라, θ(ann_alpha)>0 케이스의 평가지표가 왜곡되고 있었다.
        이제 cfg 쪽에서 이미 매입이 끝난 상태(ann_on='on' & y_ann>0)인지 먼저
        확인하고, 그렇다면 재매입하지 않고 그 값을 그대로 채택한다.
        """
        # 기본 상태 초기화
        self.y_ann = 0.0
        self.ann_purchased = False
        self.ann_P = 0.0
        self.ann_a_factor = 0.0
        self._ann_cfg = None
        self._ann_state = None

        alpha = _safe_float(getattr(self, "ann_alpha", 0.0), 0.0)
        mode = str(getattr(self, "ann_on", "auto") or "auto").lower()

        if alpha <= 0.0:
            return
        if mode == "off":
            return

        # --- [FIX] 이미 상위(setup_annuity_overlay)에서 매입이 끝난 경우 재매입 방지 ---
        upstream_y_ann = _safe_float(getattr(self, "_cfg_y_ann_precomputed", 0.0), 0.0)
        if mode == "on" and upstream_y_ann > 0.0:
            self.y_ann = upstream_y_ann
            self.ann_purchased = True
            self.ann_P = _safe_float(getattr(self, "_cfg_ann_P_precomputed", upstream_y_ann), upstream_y_ann)
            self.ann_a_factor = _safe_float(getattr(self, "_cfg_ann_a_factor_precomputed", 0.0), 0.0)
            # self.W(=self.W0)는 이미 setup_annuity_overlay에서 매입 후 잔여자산으로
            # 설정되어 있으므로 추가로 차감하지 않는다.
            return

        # auto → alpha>0 이면 on
        if mode == "auto":
            mode = "on"
            setattr(self, "ann_on", "on")

        if init_from_sim_cfg is None:
            return

        # mortality='on' & life_table 유효 + r_f_real_annual 필요
        if self.life_table is None or self.r_f_real_annual is None:
            return

        # 현재 t=0 기준 자산
        W0_now = float(
            _safe_float(getattr(self, "W", getattr(self, "W0", 0.0)), getattr(self, "W0", 0.0))
        )
        if W0_now <= 0.0:
            return

        try:
            W_after, ann_cfg, ann_state = init_from_sim_cfg(
                W0=W0_now,
                sim_cfg=self,
                life_table=self.life_table,
                r_f_real_annual=float(self.r_f_real_annual),
                steps_per_year=int(self.steps_per_year),
            )
        except Exception:
            return

        if not getattr(ann_state, "purchased", False):
            return

        self.W = max(_safe_float(W_after, W0_now), 0.0)
        self.y_ann = max(_safe_float(ann_state.y_ann, 0.0), 0.0)
        self.ann_purchased = True
        self.ann_P = _safe_float(ann_state.P, 0.0)
        self.ann_a_factor = _safe_float(ann_state.a_factor, 0.0)

        self._ann_cfg = ann_cfg
        self._ann_state = ann_state

    # ------------------------------------------------------------------ #
    #   reset / obs / state
    # ------------------------------------------------------------------ #
    def reset(self, W0: Optional[float] = None, seed: Optional[int] = None) -> np.ndarray:
        """
        reset(W0=...), reset(seed=...), reset(W0=..., seed=...) 지원.
        각 에피소드는 SeedSequence.spawn() 기반으로 경로 RNG를 분기.
        """
        # 외부에서 seed를 덮어쓸 경우, 전체 seed sequence 재설정
        if seed is not None:
            from numpy.random import SeedSequence, default_rng

            self._ss = SeedSequence(int(seed))
            self.rng = default_rng(self._ss)

        # path 별 child seed
        child = self._ss.spawn(1)[0]
        from numpy.random import default_rng

        self.rng = default_rng(child)
        self._path_counter += 1

        self.t = 0
        self.W = _safe_float(self.W0 if W0 is None else W0, self.W0)
        self.age_years = float(self.age0)
        self.is_new_year = True
        self.cpi_yoy = 0.0

        # [ANN] 초기화
        self.y_ann = 0.0
        self.ann_purchased = False
        self.ann_P = 0.0
        self.ann_a_factor = 0.0
        self._ann_cfg = None
        self._ann_state = None

        # ANN 디버그 (before)
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

        # ANN 디버그 (after)
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

        # 행동편향용 내부 상태
        self._prev_u_for_habit = 0.0

        # HJB용 할인 관련 상태
        self._disc_factor = 1.0
        self._ruin_penalized = False

        # 경로 생성
        if self.market_mode == "bootstrap":
            self.path_risky, self.path_safe, self.path_cpi = self._bootstrap_path(self.T, self.rng)
        else:
            self.path_risky = self.rng.normal(self.mu_risky, self.sigma_risky, size=self.T)
            self.path_safe = np.full(self.T, self.r_safe_fix)
            self.path_cpi = np.zeros(self.T, dtype=float)

        return self._obs().astype(np.float32)

    def _obs(self) -> np.ndarray:
        """
        관측값: [t_norm, W_t] (float32 ndarray)
        """
        t_norm = (self.t / max(1, self.T - 1)) if self.T > 1 else 0.0
        return np.asarray(
            [_safe_float(t_norm, 0.0), _safe_float(self.W, 0.0)],
            dtype=np.float32,
        )

    @property
    def state(self) -> SimpleNamespace:
        """테스트/디버깅 편의를 위한 간단 state 뷰."""
        return SimpleNamespace(
            W=float(_safe_float(self.W, 0.0)),
            t=int(self.t),
            age_years=float(self.age_years),
        )

    # ------------------------------------------------------------------ #
    #   floor helper
    # ------------------------------------------------------------------ #
    def _q_min_now(self) -> float:
        """
        동적 소비 하한:
          - floor_on=False: q_floor_base(연간 q_floor를 월로 나눈 값)
          - floor_on=True : max(q_floor_base, f_min_real/W) (연→월 환산 포함)
        """
        if not self.floor_on:
            return float(self.q_floor_base or 0.0)

        W_now = max(_safe_float(self.W, 0.0), 0.0)
        if W_now <= 0.0:
            return 0.0

        f_min_per_step = _safe_float(self.f_min_real, 0.0) / self.steps_per_year
        dyn = max(0.0, min(1.0, f_min_per_step / W_now))
        return max(float(self.q_floor_base or 0.0), dyn)

    # ------------------------------------------------------------------ #
    #   hedge
    # ------------------------------------------------------------------ #
    def _apply_hedge(self, r_risky_raw: float, r_safe: float, w: float) -> Tuple[float, bool]:
        """
        헤지 모드/강도에 따른 r_risky_eff와 hedge_active 플래그 계산.
        (현재 w는 비용 계산 등에는 직접 사용하지 않지만, 인터페이스 유지용 인자)
        """
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
                    r_risky_eff = (1.0 - k) * rr
                    hedge_active = True

        # 손실 뒤집기/상승 증폭 방지
        if rr < 0:
            r_risky_eff = min(0.0, r_risky_eff)
        else:
            r_risky_eff = min(r_risky_eff, rr)

        return _safe_float(r_risky_eff, 0.0), bool(hedge_active)

    # ------------------------------------------------------------------ #
    #   main step
    # ------------------------------------------------------------------ #
    def step(self, *args, **kwargs):
        """
        지원 형태:
          - step(q=..., w=...)
          - step(q, w)
          - step([q, w]) / step((q, w)) / step(np.array([q, w]))
          - step({"q":..., "w":...})

        반환:
          (obs, reward, done, info)  ← 항상 4-튜플
        """
        # --- action 파싱 ---
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
                    raise TypeError("step(action) expects [q,w] or dict {'q','w'}") from e
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

        # 이미 종료된 에피소드에 대한 호출 방어
        if self.t >= self.T:
            info = {
                "W_T": float(_safe_float(self.W, 0.0)),
                "done_reason": "already_ended",
                "truncated": False,
            }
            return self._obs(), 0.0, True, info

        W_start = max(_safe_float(self.W, 0.0), 0.0)

        # 1) clipping ----------------------------------------------------
        q_min = self._q_min_now()
        q = max(q_min, _clip01(q))
        w = _clip01(min(w, _safe_float(getattr(self, "w_max", 1.0), 1.0)))

        # 2) consumption -------------------------------------------------
        y_ann = _safe_float(getattr(self, "y_ann", 0.0), 0.0)
        pension_y = self._pension_Y_month if self.t >= self._pension_claim_month_idx else 0.0
        c = _safe_float(y_ann + pension_y + q * W_start, 0.0)
        # annuity/국민연금 지급은 계정 밖에서 이뤄지므로, 계정 차감은 q*W_start만 적용
        W_after_c = max(W_start - q * W_start, 0.0)

        # 3) returns (+hedge) --------------------------------------------
        r_risky_raw, r_safe = self._draw_returns()
        r_risky_eff, hedge_active = self._apply_hedge(r_risky_raw, r_safe, w)
        r_port = _safe_float(w * r_risky_eff + (1.0 - w) * r_safe, 0.0)

        hc = _safe_float(getattr(self, "hedge_cost_m", 0.0), 0.0)
        htx = _safe_float(getattr(self, "hedge_tx_m", 0.0), 0.0)
        if hedge_active and hc > 0.0:
            r_port -= w * hc
            if htx > 0.0:
                r_port -= w * htx

        gross = _safe_float(1.0 + r_port, 1.0)
        W_before_fee = _safe_float(W_after_c * gross, W_after_c)

        # 4) fee (월차감, 기준=W_after_c) -------------------------------
        fee_m_base = _safe_float(getattr(self, "fee_m", 0.0), 0.0)
        if self.ann_zero_fee_after_purchase and bool(getattr(self, "ann_purchased", False)):
            fee_m_eff = 0.0
        else:
            fee_m_eff = fee_m_base

        fee = _safe_float(fee_m_eff * W_after_c, 0.0)
        self.W = max(_safe_float(W_before_fee - fee, 0.0), 0.0)

        # ----- 효용 계산 + 행동편향 훅 ---------------------------------
        base_u = _crra_u(c, _safe_float(getattr(self, "crra_gamma", 3.0), 3.0))
        u_eff = base_u
        spec = getattr(self, "_bh_spec", None)
        if spec is not None and getattr(spec, "on", False):
            try:
                u1 = distort_utility(base_u, ref=0.0, spec=spec)
                u2 = habit_utility(u1, self._prev_u_for_habit, spec=spec)
                self._prev_u_for_habit = float(u1)
                # 후회(regret, 논문 식38): 기준소비 c*를 "4%룰 인출액(q4*W_start)"으로 설정.
                # 실제 소비(c)가 이보다 낮을 때만 페널티가 발생한다.
                q4_ref = 1.0 - (1.0 - 0.04) ** (1.0 / max(1, self.steps_per_year))
                c_ref = q4_ref * W_start
                u3 = regret_utility(u2, c, c_ref, spec=spec)
                u_eff = float(u3)
            except Exception:
                u_eff = base_u

        base_reward = (
            _safe_float(getattr(self, "u_scale", 0.0), 0.0)
            * _safe_float(u_eff, base_u)
            + _safe_float(getattr(self, "survive_bonus", 0.0), 0.0)
        )

        # time/age 업데이트 ----------------------------------------------
        self.t += 1
        self.age_years = float(self.age0) + (self.t / max(1, self.steps_per_year))
        self.is_new_year = (self.t % self.steps_per_year == 0)

        # CPI YoY
        if self.t >= self.steps_per_year:
            window = _nan_guard_arr(self.path_cpi[self.t - self.steps_per_year : self.t], fill=0.0)
            try:
                self.cpi_yoy = float(np.prod(1.0 + window) - 1.0)
            except Exception:
                self.cpi_yoy = 0.0
        else:
            self.cpi_yoy = 0.0

        done = (self.t >= self.T) or (self.W <= 0.0)

        # bias 레이어용 최근 수익/변동성 신호 ----------------------------
        recent_ret = float(r_risky_raw)
        if self.t >= 12:
            win = _nan_guard_arr(self.path_risky[self.t - 12 : self.t], fill=0.0)
            recent_vol = float(np.std(win))
        else:
            recent_vol = 0.0

        info: Dict[str, Any] = {
            "consumption": float(c),
            "y_ann": float(y_ann),
            "pension_y": float(pension_y),
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
            "truncated": False,
        }

        # ----- terminal info / shortfall L_term ------------------------
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

        # ----- HJB형 패널티 및 시간 할인 -------------------------------
        ruin_flag = bool(self.W <= 0.0)
        info["ruin_flag"] = ruin_flag

        penalty = 0.0

        lam_r = _safe_float(getattr(self, "lambda_ruin", 0.0), 0.0)
        if lam_r > 0.0 and ruin_flag:
            if (not self.ruin_penalty_once) or (not getattr(self, "_ruin_penalized", False)):
                penalty -= lam_r
                if self.ruin_penalty_once:
                    self._ruin_penalized = True

        lam_s = _safe_float(getattr(self, "lambda_shortfall", 0.0), 0.0)
        if lam_s > 0.0 and L_term > 0.0:
            penalty -= lam_s * L_term

        beta = _safe_float(getattr(self, "beta", 1.0), 1.0)
        disc_factor = float(getattr(self, "_disc_factor", 1.0))

        reward = base_reward + penalty
        if beta != 1.0:
            reward = reward * disc_factor
            self._disc_factor = disc_factor * beta
        else:
            self._disc_factor = disc_factor

        # 수치 폭주 방지용 클리핑
        reward = float(np.clip(reward, -100.0, 100.0))

        info["base_reward"] = float(base_reward)
        info["u_eff"] = float(_safe_float(u_eff, base_u))
        info["u_raw"] = float(_safe_float(base_u, 0.0))
        info["penalty"] = float(penalty)
        info["beta"] = float(beta)
        info["disc_factor"] = float(self._disc_factor)

        return self._obs(), float(reward), bool(done), info
