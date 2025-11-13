# project/annuity/overlay.py
from dataclasses import dataclass
from typing import Tuple, MutableMapping, Any, Optional

import numpy as np
import pandas as pd


@dataclass
class AnnuityConfig:
    """Static configuration for an immediate life annuity."""
    on: bool          # whether annuity feature is enabled
    alpha: float      # purchase fraction of W0 (θ)
    L: float          # load (additional expense / margin)
    d: int            # deferral (in years or steps, MVP: 0 = immediate)
    index: str        # 'real' | 'nominal'


@dataclass
class AnnuityState:
    """Runtime state of the annuity after init."""
    purchased: bool
    P: float         # premium paid at t=0
    y_ann: float     # per-step payout (e.g., monthly)
    a_factor: float  # annuity-immediate factor (per-step)
    t_star: int      # start step (MVP: 0)


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

    # align index of age0
    start_idx = int(np.searchsorted(ages, age0))
    alive = 1.0

    for a_idx in range(start_idx, len(ages)):
        mu_y = float(mu[a_idx])

        for _ in range(S):
            # monthly px under UDD: exp(-mu_y / S)
            p_m = float(np.exp(-mu_y / S))
            S_m.append(alive * p_m)
            alive *= p_m

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
    i_m = (1.0 + float(r_f_real_annual)) ** (1.0 / S) - 1.0
    v = 1.0 / (1.0 + i_m)

    surv = _monthly_survival_from_life_table(age0_years, life_table, S=S)
    # annuity-immediate: sum_{m=1..} v^m * P(alive at end of month m)
    disc = v ** np.arange(1, len(surv) + 1, dtype=float)
    a = float(np.sum(disc * surv))

    # numerical guard
    return max(a, 1e-9)


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
        Annual real risk-free rate (for discounting).
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
    load = float(max(cfg.L, -0.99))  # guard against 1+L <= 0

    P = (1.0 + load) * alpha * W0
    a = compute_ax_real(age0_years, life_table, r_f_real_annual, S=S)
    y = P / a

    W_after = W0 - P
    state = AnnuityState(True, P, y, a, 0)
    return W_after, state


# === Helper APIs for DECUM env / reporting ==================================


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
        - ann_load (float, optional)
        - ann_index (str, optional)
        - age0 (int, entry age)
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
    ann_load = float(getattr(sim_cfg, "ann_load", 0.0))
    ann_index = str(getattr(sim_cfg, "ann_index", "real"))
    age0 = int(getattr(sim_cfg, "age0", 65))

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
    metrics["ann_on"] = float(cfg.on)  # store as 0/1
    metrics["ann_alpha"] = float(cfg.alpha)
    metrics["ann_load"] = float(cfg.L)

    if state.purchased:
        metrics["P"] = float(state.P)
        metrics["y_ann"] = float(state.y_ann)
        metrics["a_factor"] = float(state.a_factor)
    else:
        # for non-purchase cases, keep consistent keys with neutral values
        metrics["P"] = 0.0
        metrics["y_ann"] = 0.0
        metrics["a_factor"] = np.nan
