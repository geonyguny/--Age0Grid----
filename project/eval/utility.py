# project/eval/utility.py
from __future__ import annotations

from typing import Iterable, Optional, Any
import numpy as np

# 행동편향(효용-레이어) 스펙 & 연산 (있으면 사용)
try:
    from ..policy.behavioral import BehavioralSpec, distort_utility, habit_utility  # type: ignore
except Exception:
    BehavioralSpec = Any  # type: ignore

    def distort_utility(u: float, *, ref: float = 0.0, spec: Any) -> float:  # type: ignore
        # 안전 폴백: 그대로 통과
        return float(u)

    def habit_utility(u: float, prev_u: float, *, spec: Any) -> float:  # type: ignore
        # 안전 폴백: 습관효용 미적용
        return float(u)


# ---------- base utility ----------
def crra_u(c: float, gamma: float) -> float:
    """CRRA utility (gamma≈1 → log)."""
    c = max(float(c), 1e-12)
    g = float(gamma)
    if abs(g - 1.0) < 1e-12:
        return float(np.log(c))
    return float((c ** (1.0 - g) - 1.0) / (1.0 - g))


def monthly_discount_from_annual(delta_annual: Optional[float], steps_per_year: int) -> float:
    """
    연간 할인계수 delta_annual ∈ (0,1] → 월간 할인계수로 변환.
    None이면 1.0 반환(무할인).
    """
    if delta_annual is None:
        return 1.0
    try:
        d = float(delta_annual)
        d = max(1e-12, min(1.0, d))
        spm = int(max(1, steps_per_year))
        return float(d ** (1.0 / spm))
    except Exception:
        return 1.0


# ---------- behavioral-aware EU on a path ----------
def path_expected_utility(
    c_hist: Iterable[float],
    *,
    gamma: float,
    u_scale: float,
    delta_m: float,
    bh_spec: Optional[BehavioralSpec] = None
) -> float:
    """
    한 경로의 소비열에 대해 기대효용 합을 계산.
      - CRRA 효용 후 u_scale 스케일
      - 습관효용 φ: U_t' = u_t - φ·u_{t-1}
      - 손실가중/현재편향 λ, β: distort_utility()로 적용
      - 시간할인은 월 할인계수 delta_m^t 로 곱함
    """
    eu = 0.0
    prev_u_raw = 0.0

    for t_idx, c_t in enumerate(c_hist):
        try:
            c = float(c_t)
        except Exception:
            continue
        if not np.isfinite(c):
            continue

        # 1) 기본 효용
        u_raw = u_scale * crra_u(c, gamma)

        # 2) 습관효용(있으면)
        u_habit = habit_utility(u_raw, prev_u_raw, spec=bh_spec) if bh_spec is not None else u_raw

        # 3) 손실가중/현재편향(있으면)
        u_behav = distort_utility(u_habit, ref=0.0, spec=bh_spec) if bh_spec is not None else u_habit

        # 4) 할인
        eu += (delta_m ** t_idx) * u_behav

        prev_u_raw = u_raw

    return float(eu)
