# project/annuity/overlay.py
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
    Static configuration for an immediate life annuity.

    Parameters
    ----------
    on : bool
        Whether the annuity feature is enabled.
    alpha : float
        Purchase fraction of initial wealth W0 (θ).
    L : float
        Load (additional expense / margin). Interpreted as φ_adval.
    d : int
        Deferral in years/steps (MVP: 0 = immediate).
        Currently not used to shift payments in time but kept for future use.
    index : str
        'real' | 'nominal' – indexation convention of the annuity payout.
    """
    on: bool
    alpha: float
    L: float
    d: int
    index: str  # 'real' | 'nominal'


@dataclass
class AnnuityState:
    """
    Runtime state of the annuity after initialization.

    Attributes
    ----------
    purchased : bool
        Whether an annuity was actually purchased.
    P : float
        Premium paid at t=0.
    y_ann : float
        Per-step payout (e.g., monthly).
    a_factor : float
        Annuity-immediate factor (per-step).
    t_star : int
        Start step (MVP: 0).
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
    Build monthly survival probabilities from a life table under UDD.

    Parameters
    ----------
    age0 : int
        Entry age (years).
    lt : DataFrame
        Life table with columns 'age' and either 'qx' or 'px'.
    S : int
        Number of steps per year (12 for monthly).
    max_age : int
        Maximum age to consider in survival projection.

    Returns
    -------
    np.ndarray
        Array of survival probabilities at the end of each month.
        Shape: (n_months,)
    """
    lt = lt.copy()

    # ensure px column
    if "px" not in lt.columns:
        qx = lt["qx"].to_numpy(dtype=float)
        px = 1.0 - np.clip(qx, 0.0, 1.0)
    else:
        px = lt["px"].to_numpy(dtype=float)

    ages = lt["age"].to_numpy(dtype=int)

    # yearly force of mortality under UDD: mu_x = -ln(px)
    mu = -np.log(np.clip(px, 1e-12, 1.0))

    S_m = []
    age_max = min(max_age, int(ages.max()))

    # align index of age0 (if age0 < min(ages), we start from the first age)
    start_idx = int(np.searchsorted(ages, age0))
    start_idx = max(start_idx, 0)
    start_idx = min(start_idx, len(ages) - 1)

    alive = 1.0

    for a_idx in range(start_idx, len(ages)):
        mu_y = float(mu[a_idx])

        # monthly px under UDD: exp(-mu_y / S)
        # keep S-month loop for flexibility (steps_per_year != 12)
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
    Real annuity-immediate factor with per-step payments (e.g. monthly).

    First payment is at step 1.

    Parameters
    ----------
    age0_years : int
        Entry age in years.
    life_table : DataFrame
        Life table with 'age' and 'qx'/'px'.
    r_f_real_annual : float
        Annual real risk-free rate.
    S : int
        Steps per year (12 for monthly).

    Returns
    -------
    float
        Present value factor a_x with per-step payments.
    """
    # guard against pathological or zero rates
    r = float(r_f_real_annual)
    if r > -0.9999:
        i_m = (1.0 + r) ** (1.0 / S) - 1.0
    else:
        # extremely negative rate → approximate with very small positive monthly rate
        i_m = 1e-8

    v = 1.0 / (1.0 + i_m)

    surv = _monthly_survival_from_life_table(age0_years, life_table, S=S)
    if surv.size == 0:
        return 1e-9

    # annuity-immediate: sum_{m=1..} v^m * P(alive at end of month m)
    disc = v ** np.arange(1, len(surv) + 1, dtype=float)
    a = float(np.sum(disc * surv))

    # numerical guard
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
    Initialize an immediate life annuity at t=0.

    Parameters
    ----------
    W0 : float
        Initial financial wealth before annuity purchase.
    cfg : AnnuityConfig
        Annuity configuration (on/alpha/load/index).
    age0_years : int
        Entry age at t=0.
    life_table : DataFrame or None
        Life table used for survival probabilities.
    r_f_real_annual : float
        Annual real risk-free rate (for discounting in real terms).
    S : int
        Steps per year (12 for monthly).

    Returns
    -------
    W0_after : float
        Wealth after paying annuity premium at t=0.
    state : AnnuityState
        Resulting annuity state (premium, payout, factor).
    """
    # annuity off or no purchase
    if (not cfg.on) or (cfg.alpha <= 0.0):
        return W0, AnnuityState(False, 0.0, 0.0, 0.0, -1)

    # safety: no life table → do not apply annuity (MVP behavior)
    if not isinstance(life_table, pd.DataFrame) or life_table.empty:
        return W0, AnnuityState(False, 0.0, 0.0, 0.0, -1)

    # premium and factor
    alpha = float(max(cfg.alpha, 0.0))
    load = float(max(cfg.L, -0.99))  # guard against 1 + L <= 0

    P = (1.0 + load) * alpha * W0

    # 현재 엔진에서는 "real" vs "nominal" 구분을
    # 전체 시뮬레이션 레벨에서 처리하므로 여기서는 항상 real factor 사용.
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
    Resolve annuity load from simulation config.

    우선순위:
      1) sim_cfg.ann_load
      2) sim_cfg.phi_adval
      3) 기본값 0.0
    """
    # 1) explicit ann_load
    if hasattr(sim_cfg, "ann_load"):
        try:
            return float(getattr(sim_cfg, "ann_load"))
        except Exception:
            pass

    # 2) CLI에서 넘어온 phi_adval을 로딩으로 해석 (run_opt_design에서 세팅)
    if hasattr(sim_cfg, "phi_adval"):
        try:
            return float(getattr(sim_cfg, "phi_adval"))
        except Exception:
            pass

    # 3) default
    return 0.0


def init_from_sim_cfg(
    W0: float,
    sim_cfg: Any,
    life_table: Optional[pd.DataFrame],
    r_f_real_annual: float,
    steps_per_year: int = 12,
) -> Tuple[float, AnnuityConfig, AnnuityState]:
    """
    Convenience initializer that builds AnnuityConfig from a simulation config.

    Parameters
    ----------
    W0 : float
        Initial wealth before annuity purchase.
    sim_cfg : Any
        Simulation config object; expected attributes:
        - ann_on (bool)
        - ann_alpha (float)
        - ann_load (float, optional)  ← if missing, phi_adval fallback
        - phi_adval (float, optional)
        - ann_index (str, optional)
        - age0 (int, entry age; default 55)
    life_table : DataFrame or None
        Life table used for survival probabilities.
    r_f_real_annual : float
        Annual real risk-free rate.
    steps_per_year : int
        Number of steps per year (e.g. 12 for monthly).

    Returns
    -------
    W_after : float
        Wealth after annuity premium at t=0.
    cfg : AnnuityConfig
        Derived annuity configuration.
    state : AnnuityState
        Resulting annuity state.
    """
    ann_on = bool(getattr(sim_cfg, "ann_on", False))
    ann_alpha = float(getattr(sim_cfg, "ann_alpha", 0.0))

    # ann_load를 우선 사용, 없으면 phi_adval을 로딩으로 사용
    ann_load = _resolve_ann_load_from_sim_cfg(sim_cfg)

    ann_index = str(getattr(sim_cfg, "ann_index", "real"))
    age0 = int(getattr(sim_cfg, "age0", 55))

    cfg = AnnuityConfig(
        on=ann_on,
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
    Write annuity-related fields into a metrics mapping (for CSV, logs, etc.).

    Parameters
    ----------
    metrics : MutableMapping[str, float]
        Target metrics dict-like object.
    cfg : AnnuityConfig
        Annuity configuration used in the run.
    state : AnnuityState
        Realized annuity state after init.
    """
    # config-level fields
    metrics["ann_on"] = float(cfg.on)  # store as 0/1
    metrics["ann_alpha"] = float(cfg.alpha)
    metrics["ann_load"] = float(cfg.L)
    metrics["ann_index"] = str(cfg.index)
    metrics["ann_defer"] = float(cfg.d)

    # state-level fields
    metrics["ann_purchased"] = float(bool(getattr(state, "purchased", False)))

    if getattr(state, "purchased", False):
        metrics["P"] = float(state.P)
        metrics["y_ann"] = float(state.y_ann)
        metrics["a_factor"] = float(state.a_factor)
    else:
        # for non-purchase cases, keep consistent keys with neutral values
        metrics["P"] = 0.0
        metrics["y_ann"] = 0.0
        metrics["a_factor"] = np.nan
