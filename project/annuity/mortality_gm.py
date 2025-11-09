# project/annuity/mortality_gm.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Literal, Optional

__all__ = [
    "PARAMS",
    "gm_hazard",
    "monthly_survival_curve",              # (GM 기반 월 생존곡선) 레거시 호환
    "make_base_life_table_gm",            # BASE 생명표(DataFrame: age, qx)
    "apply_cohort_improvement",           # 코호트 개선률 적용(annual imp.)
    "load_life_table",                    # 모드 문자열로 로드: BASE / COHORT_YYYY
    "discretize_qx_to_monthly_survival",  # (옵션) qx 테이블 → 월 S_t
]

# 간단 파라미터(예시값): 국내 생명표 대체용 근사치 (Gompertz–Makeham)
PARAMS = {
    "M": dict(A=5.0e-4, B=5.0e-5, c=1.085),  # 남
    "F": dict(A=4.0e-4, B=3.5e-5, c=1.082),  # 여
}

Sex = Literal["M", "F"]


# ─────────────────────────────────────────────────────────
# GM hazard & 월 생존곡선(레거시 호환)
# ─────────────────────────────────────────────────────────
def gm_hazard(age: np.ndarray, sex: Sex = "M") -> np.ndarray:
    """
    Gompertz–Makeham 형태의 순간사망강도 h(age).
    h(x) = A + B * c^(x - 40)  (기준연령 40 근사)
    """
    p = PARAMS.get(sex.upper(), PARAMS["M"])
    A, B, c = float(p["A"]), float(p["B"]), float(p["c"])
    return A + B * (c ** (np.asarray(age, dtype=float) - 40.0))


def monthly_survival_curve(
    age0: int,
    horizon_years: int,
    steps_per_year: int = 12,
    sex: Sex = "M",
) -> Dict[str, np.ndarray]:
    """
    (레거시) GM hazard로부터 월 생존곡선을 생성.
    반환: dict(S, q, age, hazard)
      - S: 길이 T+1 의 생존함수(월 기준, S[0]=1)
      - q: 길이 T 의 월 사망확률
    """
    T = int(horizon_years * steps_per_year)
    dt = 1.0 / steps_per_year
    ages = age0 + np.arange(T) * dt
    h_ann = gm_hazard(ages, sex=sex)
    # 포아송 근사: S_{t+1} = S_t * exp(-h*dt)
    S = np.empty(T + 1, dtype=float)
    S[0] = 1.0
    for t in range(T):
        S[t + 1] = S[t] * np.exp(-h_ann[t] * dt)
    q = 1.0 - (S[1:] / S[:-1])  # 월별 사망확률
    return dict(S=S, q=q, age=ages, hazard=h_ann)


# ─────────────────────────────────────────────────────────
# 생명표(DataFrame: age, qx) 생성/로딩 헬퍼
# ─────────────────────────────────────────────────────────
def _annual_qx_from_gm(age_grid: np.ndarray, sex: Sex = "M", steps_per_year: int = 12) -> np.ndarray:
    """
    GM hazard로부터 '연간 사망확률 q_x'를 수치적으로 얻는다.
    1년 구간을 월 12회로 분할하여 생존을 적분(포아송 근사).
    """
    age_grid = np.asarray(age_grid, dtype=float)
    spm = max(1, int(steps_per_year))
    dt = 1.0 / spm

    qx = np.zeros_like(age_grid, dtype=float)
    for i, x in enumerate(age_grid):
        # [x, x+1) 구간을 월 단위로 적분
        ages_month = x + np.arange(spm) * dt
        h = gm_hazard(ages_month, sex=sex)
        S = 1.0
        for k in range(spm):
            S *= np.exp(-h[k] * dt)
        qx[i] = 1.0 - S  # 1년 내 사망확률
    return qx


def make_base_life_table_gm(
    sex: Sex = "M",
    age_min: int = 20,
    age_max: int = 120,
    steps_per_year: int = 12,
) -> pd.DataFrame:
    """
    GM 파라미터로부터 BASE(기간) 생명표를 구성한다.
    반환: DataFrame[['age','qx']]
    """
    ages = np.arange(int(age_min), int(age_max) + 1, dtype=int)
    qx = _annual_qx_from_gm(ages, sex=sex, steps_per_year=steps_per_year)
    return pd.DataFrame({"age": ages, "qx": np.clip(qx, 0.0, 1.0)})


def apply_cohort_improvement(
    base_table: pd.DataFrame,
    *,
    age0: int,
    horizon_years: int,
    annual_improvement: float = 0.01,
) -> pd.DataFrame:
    """
    코호트 개선률(연간 사망개선율)을 단순 적용한 근사 코호트 생명표를 만든다.

    아이디어:
      - 연금 개시 시점의 연령을 age0라 하고, t년 뒤 도달연령은 age0 + t.
      - 해당 시점의 사망확률을 base q_x에 (1 - imp)^t 를 곱해 개선(감소)한다.
      - 즉, '미래로 갈수록 q_x가 매년 일정 비율로 낮아진다'는 코호트 근사.

    반환: DataFrame[['age','qx_cohort']]
    주의: 진짜 코호트 생명표(연도별 q_{x,y})가 있으면 그것을 쓰는 것이 최선이고,
         본 함수는 데이터 부재 시 합리적 근사로 사용한다.
    """
    if base_table is None or base_table.empty:
        raise ValueError("base_table is empty")
    if "age" not in base_table or "qx" not in base_table:
        raise ValueError("base_table must have columns ['age','qx']")

    ages = base_table["age"].to_numpy(dtype=int)
    qx_base = base_table["qx"].to_numpy(dtype=float)

    # 필요한 연령대만 추출 (age0 ~ age0 + horizon_years)
    a_min = int(age0)
    a_max = int(age0 + horizon_years)
    mask = (ages >= a_min) & (ages <= a_max)
    ages_sel = ages[mask]
    qx_sel = qx_base[mask]

    t_years = (ages_sel - a_min).astype(float)  # 개시로부터 경과연수
    imp = float(max(0.0, annual_improvement))
    factor = (1.0 - imp) ** t_years
    qx_cohort = np.clip(qx_sel * factor, 0.0, 1.0)

    return pd.DataFrame({"age": ages_sel, "qx": qx_cohort})


def load_life_table(
    mode: str,
    *,
    sex: Sex = "M",
    age0: int = 65,
    horizon_years: int = 35,
    steps_per_year: int = 12,
    annual_improvement: float = 0.01,
) -> pd.DataFrame:
    """
    생명표 로더(간단 문자열 인터페이스).
      - mode="BASE"               : 기간표(코호트 개선 미적용)
      - mode="COHORT"             : 코호트 근사(개선율 annual_improvement 적용)
      - mode="cohort_YYYY" / etc. : 접두사 인식해서 코호트로 취급(YYYY는 정보용)

    반환: DataFrame[['age','qx']]
    """
    mode_l = str(mode).strip().lower()
    # BASE
    if mode_l == "base":
        return make_base_life_table_gm(sex=sex, steps_per_year=steps_per_year)
    # COHORT 계열
    if mode_l.startswith("cohort"):
        base = make_base_life_table_gm(sex=sex, steps_per_year=steps_per_year)
        return apply_cohort_improvement(
            base,
            age0=age0,
            horizon_years=horizon_years,
            annual_improvement=annual_improvement,
        )
    # 기본: BASE로 처리
    return make_base_life_table_gm(sex=sex, steps_per_year=steps_per_year)


# ─────────────────────────────────────────────────────────
# (옵션) qx 테이블 → 월 단위 생존함수로 이산화
# ─────────────────────────────────────────────────────────
def discretize_qx_to_monthly_survival(
    life_table: pd.DataFrame,
    *,
    age0: int,
    horizon_years: int,
    steps_per_year: int = 12,
) -> Dict[str, np.ndarray]:
    """
    DataFrame(age, qx)을 받아 월 단위 생존함수 S_t를 만든다.
    연간 qx를 월 q_m으로 변환(균등위험 근사): q_m = 1 - (1 - qx)^(1/spm)
    """
    assert {"age", "qx"}.issubset(life_table.columns), "life_table needs columns: age,qx"
    spm = max(1, int(steps_per_year))
    T = int(horizon_years * spm)

    lt = life_table.set_index("age").sort_index()
    ages = np.arange(age0, age0 + horizon_years + 1, dtype=int)
    qy = np.clip(lt.loc[ages, "qx"].to_numpy(dtype=float, copy=True), 0.0, 1.0)

    q_m_year = 1.0 - (1.0 - qy) ** (1.0 / spm)     # 연→월
    q_m = np.repeat(q_m_year[:-1], spm)            # 마지막 연은 경계 제외
    if q_m.size < T:
        q_m = np.pad(q_m, (0, T - q_m.size), constant_values=q_m[-1] if q_m.size else 0.0)
    else:
        q_m = q_m[:T]

    # S_t: 길이 T+1, S[0]=1
    S = np.empty(T + 1, dtype=float); S[0] = 1.0
    for t in range(T):
        S[t + 1] = S[t] * (1.0 - np.clip(q_m[t], 0.0, 1.0))
    q_month = 1.0 - (S[1:] / S[:-1])

    return dict(S=S, q=q_month, q_month=q_m, q_year=qy)
