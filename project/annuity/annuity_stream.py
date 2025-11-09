# project/annuity/annuity_stream.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Dict
import numpy as np
import pandas as pd

IndexMode = Literal["real", "nominal"]

__all__ = [
    "IndexMode",
    "AnnuitySpec",
    "annuity_factor",
    "annuity_income_from_pv",
    "level_payment_from_alpha",
    "next_indexed_payment",
]

# ─────────────────────────────────────────────────────────
# Spec
# ─────────────────────────────────────────────────────────
@dataclass
class AnnuitySpec:
    """
    즉시연금(월지급) 사양.

    - index_mode:
        'real'    : 실질 지급 고정(물가와 무관하게 실질액 P 유지)
        'nominal' : 명목 지급(물가상승률(cpi_m)에 따라 명목 P가 변동)
      ※ 본 모듈의 PV 계산은 '할인율 r_f_annual'의 체계에 따릅니다.
        index_mode는 '지급 업데이트' 단계(next_indexed_payment)에서 사용합니다.

    - r_f_annual  : (실질 또는 명목) 무위험 연율
    - phi_adval   : 가입 시 비례 로딩(예: 0.03 = 3%)
    - ann_L       : 연 기준 추가 마진(예: 0.02 = 2%); 월 할인률에 가산
    - first_payment_immediate : 즉시연금(True)이면 첫 지급이 t=0 시점
    """
    steps_per_year: int = 12
    index_mode: IndexMode = "real"
    r_f_annual: float = 0.02
    first_payment_immediate: bool = True

    # 현실화 파라미터
    phi_adval: float = 0.0      # 가입 로딩(ad-valorem)
    ann_L: float = 0.0          # 추가 마진(연 기준, 월 할인률에 가산)


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────
def _to_monthly_rate(r_annual: float, spm: int) -> float:
    spm = max(1, int(spm))
    return (1.0 + float(r_annual)) ** (1.0 / spm) - 1.0


def _apply_margin_monthly(r_m: float, ann_L: float, spm: int) -> float:
    """
    월 할인률 r_m에 연 마진 ann_L를 월 기준으로 가산.
    단순 가산이 아니라 월복리 일관성을 위해 곱 형태로 변환.
    """
    if ann_L and ann_L > 0.0:
        # r_m' = (1+r_m) * (1 + ann_L/spm) - 1
        r_m = (1.0 + r_m) * (1.0 + float(ann_L) / max(1, spm)) - 1.0
    return float(r_m)


def _survival_probs_from_table(
    age0: int, horizon_years: int, life_table: pd.DataFrame, spm: int
) -> np.ndarray:
    """
    생존확률 월 시퀀스 S_t 생성.
    life_table: 최소 ['age','qx'] 필요(qx=해당 연령에서 1년 내 사망확률).
    반환 S 길이 = horizon_years * spm, 각 원소는 '해당 월 직전 생존확률'.
    """
    assert {"age", "qx"}.issubset(set(life_table.columns)), "life_table needs columns: age,qx"
    lt = life_table.set_index("age").sort_index()

    ages = np.arange(age0, age0 + horizon_years + 1, dtype=int)
    qy = np.clip(lt.loc[ages, "qx"].to_numpy(dtype=float, copy=True), 0.0, 1.0)

    # 연간 qx → 월간 q_m (균등 위험 가정)
    q_m_year = 1.0 - (1.0 - qy) ** (1.0 / spm)
    q_m = np.repeat(q_m_year[:-1], spm)  # 마지막 연령 구간은 horizon 경계로 자름

    Tm = int(horizon_years * spm)
    if q_m.size < Tm:
        pad_val = q_m[-1] if q_m.size else 0.0
        q_m = np.pad(q_m, (0, Tm - q_m.size), constant_values=pad_val)
    else:
        q_m = q_m[:Tm]

    # 생존확률 S_t (S_0=1, 각 월 직전 생존확률)
    # S_t = prod_{i=0..t-1}(1 - q_m[i])
    with np.errstate(divide="ignore", invalid="ignore"):
        log_surv = np.cumsum(np.log1p(-np.clip(q_m, 0.0, 1.0)))
        S = np.exp(np.insert(log_surv, 0, 0.0))[:-1]  # 길이 Tm
    return S


# ─────────────────────────────────────────────────────────
# Core PV / Factor
# ─────────────────────────────────────────────────────────
def annuity_factor(
    *,
    W0: float,  # 유지(호환 목적), 실제 계산에는 미사용
    age0: int,
    horizon_years: int,
    life_table: Optional[pd.DataFrame],
    spec: AnnuitySpec,
) -> Tuple[float, float]:
    """
    (a_factor, r_m)를 반환.
      a_factor: '월 1 단위 지급의 현재가치 합'(생존·할인 고려)
      r_m     : 사용된 월 무위험률(마진 반영 후)
    - index_mode는 지급 업데이트 로직(next_indexed_payment)에서 사용하므로
      본 PV 계산은 할인 체계(r_f_annual + ann_L)와 생존만 반영합니다.
    """
    spm = max(1, int(spec.steps_per_year))
    # 무위험 월리 변환 + 마진 가산
    r_m = _to_monthly_rate(spec.r_f_annual, spm)
    r_m = _apply_margin_monthly(r_m, spec.ann_L, spm)

    Tm = int(horizon_years * spm)

    # 지급 타이밍: 즉시(True)면 t=0 포함, 1개월후(False)면 t=1부터
    start = 0 if spec.first_payment_immediate else 1
    stop = Tm + (0 if spec.first_payment_immediate else 1)
    disc = (1.0 / (1.0 + r_m)) ** np.arange(start, stop)
    disc = disc[:Tm]  # 방어

    if life_table is not None and not life_table.empty:
        S = _survival_probs_from_table(age0, horizon_years, life_table, spm)
    else:
        S = np.ones(Tm, dtype=float)

    a_factor = float(np.sum(S * disc))
    return a_factor, float(r_m)


def annuity_income_from_pv(
    *,
    pv: float,
    age0: int,
    horizon_years: int,
    life_table: Optional[pd.DataFrame],
    spec: AnnuitySpec,
) -> Dict[str, float]:
    """
    투입원금 pv에서 즉시연금 월지급액을 산출.
    로딩(phi_adval)은 투입원금에서 차감하여 순PV(pv_net)를 만들고,
    계수 a_factor로 나눠 지급액을 결정합니다.

    반환:
      {
        'income'    : 월 지급액,
        'ann_factor': a_factor,
        'pv_net'    : pv * (1 - phi_adval),
        'r_m_used'  : 월 할인률(마진 반영 후)
      }
    """
    pv = float(pv)
    if pv <= 0.0:
        return {"income": 0.0, "ann_factor": 0.0, "pv_net": 0.0, "r_m_used": 0.0}

    a_factor, r_m_used = annuity_factor(
        W0=0.0,
        age0=age0,
        horizon_years=horizon_years,
        life_table=life_table,
        spec=spec,
    )

    pv_net = pv * (1.0 - max(0.0, float(spec.phi_adval)))
    income = pv_net / a_factor if a_factor > 0.0 else 0.0
    return {
        "income": float(income),
        "ann_factor": float(a_factor),
        "pv_net": float(pv_net),
        "r_m_used": float(r_m_used),
    }


# ─────────────────────────────────────────────────────────
# Convenience API (호환)
# ─────────────────────────────────────────────────────────
def level_payment_from_alpha(*, W0: float, alpha: float, a_factor: float) -> float:
    """
    초기자산 W0 중 α 비율을 즉시연금에 투입한다고 할 때,
    월 고정지급액 P 를 반환. (로딩·마진 미반영 레거시 도우미)
    실무 사용 시 annuity_income_from_pv(pv=alpha*W0, ...) 사용을 권장합니다.
    """
    alpha = max(0.0, min(1.0, float(alpha)))
    if a_factor <= 0.0:
        return 0.0
    return float(alpha * float(W0) / float(a_factor))


def next_indexed_payment(P_real: float, cpi_m: float, index_mode: IndexMode) -> float:
    """
    직전 월 '실질 기준' 지급 P_real과 CPI 월변화율 cpi_m을 받아 다음월 지급을 리턴.
      - real    : P_real(변동 없음; 실질고정)
      - nominal : P_real * (1 + cpi_m) (명목 인덱싱)
    """
    if index_mode == "nominal":
        return float(P_real) * (1.0 + float(cpi_m))
    return float(P_real)
