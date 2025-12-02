from dataclasses import dataclass
from typing import Tuple, MutableMapping, Any, Optional

import numpy as np
import pandas as pd


# =====================================================================
# Dataclasses
# =====================================================================


@dataclass
class AnnuityConfig:
    """
    종신연금(즉시연금) 설정 값.

    Parameters
    ----------
    on : bool
        연금 기능 사용 여부.
    alpha : float
        초기 자산 W0 중 연금에 투입하는 비율(θ).
    L : float
        부가보험료(load, φ_adval에 해당).
    d : int
        연금 개시 지연 기간(연/스텝 단위, MVP에서는 0).
    index : str
        'real' | 'nominal' – 연금 지급의 인덱싱 방식.
    """
    on: bool
    alpha: float
    L: float
    d: int
    index: str  # 'real' | 'nominal'


@dataclass
class AnnuityState:
    """
    초기화 이후 연금 상태.

    Attributes
    ----------
    purchased : bool
        실제로 연금을 매입했는지 여부.
    P : float
        t=0 시점 납입 보험료.
    y_ann : float
        스텝(예: 월)당 지급액.
    a_factor : float
        스텝 단위 종신연금 현가계수 a_x.
    t_star : int
        지급 개시 스텝(MVP: 0).
    """
    purchased: bool
    P: float
    y_ann: float
    a_factor: float
    t_star: int


# =====================================================================
# Survival and discounting helpers
# =====================================================================


def _monthly_survival_from_life_table(
    age0: int,
    lt: pd.DataFrame,
    S: int = 12,
    max_age: int = 110,
) -> np.ndarray:
    """
    생명표로부터 월 단위 생존확률을 생성 (UDD 가정).

    Parameters
    ----------
    age0 : int
        진입 연령(년).
    lt : DataFrame
        'age'와 'qx' 또는 'px'를 포함한 생명표.
    S : int
        연간 스텝 수(월 단위 12).
    max_age : int
        생존확률을 계산할 최대 연령.

    Returns
    -------
    np.ndarray
        월말 기준 생존확률 배열 (shape: (n_months,)).
    """
    lt = lt.copy()

    # px 열 보장
    if "px" not in lt.columns:
        qx = lt["qx"].to_numpy(dtype=float)
        px = 1.0 - np.clip(qx, 0.0, 1.0)
    else:
        px = lt["px"].to_numpy(dtype=float)

    ages = lt["age"].to_numpy(dtype=int)

    # UDD 하의 연 단위 사망강도: mu_x = -ln(px)
    mu = -np.log(np.clip(px, 1e-12, 1.0))

    S_m = []
    age_max = min(max_age, int(ages.max()))

    # age0에 해당하는 인덱스 정렬
    start_idx = int(np.searchsorted(ages, age0))
    start_idx = max(start_idx, 0)
    start_idx = min(start_idx, len(ages) - 1)

    alive = 1.0

    for a_idx in range(start_idx, len(ages)):
        mu_y = float(mu[a_idx])

        # 월간 생존확률 px_month = exp(-mu_y / S)
        for _ in range(S):
            p_m = float(np.exp(-mu_y / S))
            alive *= p_m
            S_m.append(alive)

        if ages[a_idx] >= age_max:
            break

    return np.array(S_m, dtype=float)


def compute_ax_real(
    age0_years: int,
    life_table: pd.DataFrame,
    r_f_real_annual: float,
    S: int = 12,
) -> float:
    """
    실질 기준 종신연금(즉시연금) 계수 a_x (스텝당 지급).

    첫 지급은 step=1에서 발생하는 annuity-immediate 형태.

    Parameters
    ----------
    age0_years : int
        가입 연령.
    life_table : DataFrame
        'age'와 'qx' 또는 'px'를 가진 생명표.
    r_f_real_annual : float
        연간 실질 무위험이자율.
    S : int
        연간 스텝 수(월 단위 12).

    Returns
    -------
    float
        per-step 지급을 전제로 한 a_x.
    """
    r = float(r_f_real_annual)
    if r > -0.9999:
        i_m = (1.0 + r) ** (1.0 / S) - 1.0
    else:
        # 극단적 음(-)금리 방어
        i_m = 1e-8

    v = 1.0 / (1.0 + i_m)

    surv = _monthly_survival_from_life_table(age0_years, life_table, S=S)
    if surv.size == 0:
        return 1e-9

    # annuity-immediate: sum_{m=1..} v^m * P(alive at end of month m)
    disc = v ** np.arange(1, len(surv) + 1, dtype=float)
    a = float(np.sum(disc * surv))

    return max(a, 1e-9)


# =====================================================================
# Core initializer
# =====================================================================


def init_annuity(
    W0: float,
    cfg: AnnuityConfig,
    age0_years: int,
    life_table: Optional[pd.DataFrame],
    r_f_real_annual: float,
    S: int = 12,
) -> Tuple[float, AnnuityState]:
    """
    t=0에서 종신연금을 1회 매입하는 초기화 로직.

    Parameters
    ----------
    W0 : float
        연금 매입 전 금융자산.
    cfg : AnnuityConfig
        연금 설정(on/alpha/load/index 등).
    age0_years : int
        가입 연령.
    life_table : DataFrame or None
        생명표. 없을 경우 연금 미적용.
    r_f_real_annual : float
        연간 실질 무위험이자율.
    S : int
        연간 스텝 수(월 단위 12).

    Returns
    -------
    W0_after : float
        연금 보험료 납입 후 금융자산.
    state : AnnuityState
        연금 상태(보험료, 지급액, 계수 등).
    """
    # 연금 off 또는 alpha=0이면 미적용
    if (not cfg.on) or (cfg.alpha <= 0.0):
        return W0, AnnuityState(False, 0.0, 0.0, 0.0, -1)

    # 생명표가 없으면 현재 MVP에서는 annuity 미적용
    if not isinstance(life_table, pd.DataFrame) or life_table.empty:
        return W0, AnnuityState(False, 0.0, 0.0, 0.0, -1)

    alpha = float(max(cfg.alpha, 0.0))
    load = float(max(cfg.L, -0.99))  # 1+L ≤ 0 방어

    P = (1.0 + load) * alpha * W0

    # 현재 엔진에서는 real/nominal 구분은 상위 레벨에서 처리
    a = compute_ax_real(age0_years, life_table, r_f_real_annual, S=S)
    y = P / a

    W_after = W0 - P
    state = AnnuityState(True, P, y, a, 0)
    return W_after, state


# =====================================================================
# Helper APIs for DECUM env / reporting
# =====================================================================


def _resolve_ann_load_from_sim_cfg(sim_cfg: Any) -> float:
    """
    시뮬레이션 설정(sim_cfg)에서 연금 load를 추론.

    우선순위:
      1) sim_cfg.ann_load
      2) sim_cfg.phi_adval
      3) 기본값 0.0
    """
    if hasattr(sim_cfg, "ann_load"):
        try:
            return float(getattr(sim_cfg, "ann_load"))
        except Exception:
            pass

    if hasattr(sim_cfg, "phi_adval"):
        try:
            return float(getattr(sim_cfg, "phi_adval"))
        except Exception:
            pass

    return 0.0


def init_from_sim_cfg(
    W0: float,
    sim_cfg: Any,
    life_table: Optional[pd.DataFrame],
    r_f_real_annual: float,
    steps_per_year: int = 12,
) -> Tuple[float, AnnuityConfig, AnnuityState]:
    """
    시뮬레이션 cfg에서 AnnuityConfig를 구성하고, 초기 연금 매입을 수행.

    Parameters
    ----------
    W0 : float
        연금 매입 전 자산.
    sim_cfg : Any
        시뮬레이션 설정 객체. 예상 속성:
        - ann_on (bool 또는 {'on','off','auto'})
        - ann_alpha (float)
        - ann_load (float, optional)
        - phi_adval (float, optional)
        - ann_index (str, optional)
        - age0 (int, optional)
    life_table : DataFrame or None
        생명표.
    r_f_real_annual : float
        연간 실질 무위험이자율.
    steps_per_year : int
        연간 스텝 수(월 단위 12).

    Returns
    -------
    W_after : float
        연금 보험료 납입 후 자산.
    cfg : AnnuityConfig
        생성된 연금 설정.
    state : AnnuityState
        연금 상태.
    """
    raw_on = getattr(sim_cfg, "ann_on", False)
    if isinstance(raw_on, str):
        mode = raw_on.lower()
        on_flag = mode == "on"
    else:
        on_flag = bool(raw_on)
        mode = "on" if on_flag else "off"

    ann_alpha = float(getattr(sim_cfg, "ann_alpha", 0.0))
    ann_load = _resolve_ann_load_from_sim_cfg(sim_cfg)
    ann_index = str(getattr(sim_cfg, "ann_index", "real"))
    age0 = int(getattr(sim_cfg, "age0", 55))

    cfg = AnnuityConfig(
        on=on_flag,
        alpha=ann_alpha,
        L=ann_load,
        d=0,
        index=ann_index,
    )

    W_after, state = init_annuity(
        W0=W0,
        cfg=cfg,
        age0_years=age0,
        life_table=life_table,
        r_f_real_annual=r_f_real_annual,
        S=steps_per_year,
    )
    return W_after, cfg, state


def write_annuity_metrics(
    metrics: MutableMapping[str, float],
    cfg: AnnuityConfig,
    state: AnnuityState,
) -> None:
    """
    metrics(dict-like)에 연금 관련 지표를 써 넣는 헬퍼.

    - ann_on / ann_alpha / ann_load / ann_index / ann_defer
    - ann_purchased / P / y_ann / a_factor / ann_a_factor
    """
    # config-level
    metrics["ann_on"] = float(bool(cfg.on))       # 0/1
    metrics["ann_alpha"] = float(cfg.alpha)
    metrics["ann_load"] = float(cfg.L)
    metrics["ann_index"] = str(cfg.index)
    metrics["ann_defer"] = float(cfg.d)

    # state-level
    purchased = bool(getattr(state, "purchased", False))
    metrics["ann_purchased"] = float(purchased)

    if purchased:
        metrics["P"] = float(state.P)
        metrics["y_ann"] = float(state.y_ann)
        metrics["a_factor"] = float(state.a_factor)
        metrics["ann_a_factor"] = float(state.a_factor)
    else:
        metrics["P"] = 0.0
        metrics["y_ann"] = 0.0
        metrics["a_factor"] = np.nan
        metrics["ann_a_factor"] = np.nan
