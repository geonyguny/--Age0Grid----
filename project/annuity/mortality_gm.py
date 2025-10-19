# project/annuity/mortality_gm.py
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple

# 간단 파라미터(예시값): 국내 생명표 대체용 근사치
PARAMS = {
    "M": dict(A=5e-4, B=5.0e-5, c=1.085),  # 남
    "F": dict(A=4e-4, B=3.5e-5, c=1.082),  # 여
}

def gm_hazard(age: np.ndarray, sex: str = "M") -> np.ndarray:
    p = PARAMS.get(sex.upper(), PARAMS["M"])
    A, B, c = p["A"], p["B"], p["c"]
    return A + B * (c ** (age - 40.0))  # 기준연령 40 근사

def monthly_survival_curve(age0: int, horizon_years: int, steps_per_year: int = 12, sex: str = "M") -> Dict[str, np.ndarray]:
    T = int(horizon_years * steps_per_year)
    dt = 1.0 / steps_per_year
    ages = age0 + np.arange(T) * dt
    h_ann = gm_hazard(ages, sex=sex)
    # 포아송 근사: S_{t+1} = S_t * exp(-h*dt)
    S = np.empty(T+1, dtype=float); S[0] = 1.0
    for t in range(T):
        S[t+1] = S[t] * np.exp(-h_ann[t] * dt)
    q = 1.0 - (S[1:] / S[:-1])  # 월별 사망확률
    return dict(S=S, q=q, age=ages, hazard=h_ann)
