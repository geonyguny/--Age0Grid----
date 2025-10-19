# project/annuity/annuity_stream.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterable, Tuple, Literal
import numpy as np
import pandas as pd

IndexMode = Literal["real", "nominal"]

@dataclass
class AnnuitySpec:
    """연금 지급 사양(월지급 기준)."""
    steps_per_year: int = 12
    index_mode: IndexMode = "real"      # 'real'이면 실질 기준, 'nominal'이면 명목 기준
    r_f_annual: float = 0.02            # (실질 or 명목) 무위험 연율
    first_payment_immediate: bool = True  # 즉시연금(True) / 1개월 후(False)

def _to_monthly_rate(r_annual: float, spm: int) -> float:
    spm = max(1, int(spm))
    return (1.0 + float(r_annual)) ** (1.0 / spm) - 1.0

def _survival_probs_from_table(age0: int, horizon_years: int, life_table: pd.DataFrame, spm: int) -> np.ndarray:
    """
    생존확률 월 시퀀스 S_t 생성.
    life_table: 최소 ['age','qx'] 필요(qx=해당 연령에서 1년 내 사망확률).
    """
    assert {"age", "qx"}.issubset(set(life_table.columns)), "life_table needs columns: age,qx"
    lt = life_table.set_index("age").sort_index()
    ages = np.arange(age0, age0 + horizon_years + 1, dtype=int)
    qy = np.clip(lt.loc[ages, "qx"].to_numpy(dtype=float, copy=True), 0.0, 1.0)  # yearly q_x

    # 연간 qx → 월간 q_m (균등위험 가정: 1 - (1 - qy)^(1/12))
    q_m = 1.0 - (1.0 - qy) ** (1.0 / spm)
    q_m = np.repeat(q_m[:-1], spm)  # 마지막 연령 끝은 horizon 경계

    Tm = int(horizon_years * spm)
    q_m = q_m[:Tm] if q_m.size >= Tm else np.pad(q_m, (0, Tm - q_m.size), constant_values=q_m[-1] if q_m.size else 0.0)

    # 생존확률 S_t (t=0..Tm-1 직전 생존 확률), S_0=1
    log_surv = np.cumsum(np.log1p(-np.clip(q_m, 0.0, 1.0)))
    S = np.exp(np.insert(log_surv, 0, 0.0))[:-1]
    return S  # length Tm

def annuity_factor(
    *,
    W0: float,
    age0: int,
    horizon_years: int,
    life_table: Optional[pd.DataFrame],
    spec: AnnuitySpec,
) -> Tuple[float, float]:
    """
    (a_factor, r_m)를 반환.
    a_factor: '월 1 단위 지급의 현재가치 합' (index_mode에 맞는 할인/생존 고려)
    r_m     : 사용된 월 무위험률
    """
    spm = max(1, int(spec.steps_per_year))
    r_m = _to_monthly_rate(spec.r_f_annual, spm)

    Tm = int(horizon_years * spm)
    disc = (1.0 / (1.0 + r_m)) ** np.arange(0 if spec.first_payment_immediate else 1, Tm + (0 if spec.first_payment_immediate else 1))
    disc = disc[:Tm]  # 방어

    if life_table is not None and not life_table.empty:
        S = _survival_probs_from_table(age0, horizon_years, life_table, spm)
    else:
        # 생존표 없으면 S=1(생존) 근사
        S = np.ones(Tm, dtype=float)

    # real vs nominal: real이면 이미 실질금리 r_m 를 넣는다고 보면 OK(지급 자체는 실질 고정)
    # nominal 모드여도 여기서는 r_m만 다르게 오면 동일 수식으로 PV를 얻는다.
    a_factor = float(np.sum(S * disc))
    return a_factor, r_m

def level_payment_from_alpha(
    *,
    W0: float,
    alpha: float,
    a_factor: float,
) -> float:
    """
    초기자산 W0 중 α 비율을 즉시연금에 투입한다고 할 때,
    월 고정지급액 P 를 반환. P = α * W0 / a_factor
    """
    alpha = max(0.0, min(1.0, float(alpha)))
    if a_factor <= 0.0:
        return 0.0
    return float(alpha * W0 / a_factor)

def next_indexed_payment(P_real: float, cpi_m: float, index_mode: IndexMode) -> float:
    """
    직전 월 실질기준 지급 P_real 과 CPI 월변화율 cpi_m을 받아
    명목/실질 모드에 맞춰 다음월 지급을 리턴.
    - real : 여전히 P_real (실질 고정)
    - nominal : P_real*(1+cpi_m) (CPI 연동)
    """
    if index_mode == "nominal":
        return float(P_real * (1.0 + float(cpi_m)))
    return float(P_real)
