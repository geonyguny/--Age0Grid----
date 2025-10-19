# project/runner/annuity_wiring.py
from __future__ import annotations
from typing import Optional, Any, Dict
import numpy as _np
import pandas as pd

from ..config import SimConfig
from ..env.retirement_env import RetirementEnv  # type: ignore
from .helpers import get_life_table_from_env

# (있으면 사용) 경량 유틸: annuity factor 계산/지급액 산출
try:
    from ..annuity.annuity_stream import (
        AnnuitySpec, annuity_factor, level_payment_from_alpha
    )  # type: ignore
    _ANN_STREAM = True
except Exception:
    _ANN_STREAM = False


# ─────────────────────────────────────────
# small utils
# ─────────────────────────────────────────
def _f(x: Any, d: float = 0.0) -> float:
    """float 캐스팅 + NaN/Inf 방호."""
    try:
        v = float(x)
        return v if _np.isfinite(v) else float(d)
    except Exception:
        return float(d)


def _off(cfg: SimConfig) -> None:
    """cfg를 'annuity off' 상태로 정리."""
    setattr(cfg, "ann_on", "off")
    setattr(cfg, "y_ann", 0.0)
    setattr(cfg, "ann_P", 0.0)
    setattr(cfg, "ann_a_factor", 0.0)


def _avg_monthly_rf(cfg: SimConfig, env_probe: Any) -> float:
    """
    월 무위험률 r_m 추정:
      - cfg.data_rf_series 가 우선
      - 없으면 env.path_safe 평균
      - 전부 없으면 0.0
    """
    r_series = getattr(cfg, "data_rf_series", None)
    if isinstance(r_series, (list, _np.ndarray)) and len(r_series) > 0:
        return _f(_np.nanmean(_np.asarray(r_series, dtype=float)), 0.0)

    try:
        ps = getattr(env_probe, "path_safe", None)
        if isinstance(ps, _np.ndarray) and ps.size > 0:
            return _f(_np.nanmean(ps.astype(float)), 0.0)
    except Exception:
        pass
    return 0.0


def _avg_monthly_cpi(cfg: SimConfig) -> float:
    """월 CPI율 평균 추정(지수 또는 월률 혼재 가능 → 차분/비율로 처리). 실패 시 0."""
    cpi = getattr(cfg, "data_cpi", None)
    if isinstance(cpi, (list, _np.ndarray)) and len(cpi) > 1:
        arr = _np.asarray(cpi, dtype=float)
        with _np.errstate(all="ignore"):
            # 지수로 가정하고 전월 대비 비율. 이미 월률이면 크기상 영향 적음.
            r = arr[1:] / arr[:-1] - 1.0
        r = _np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        return _f(_np.nanmean(r), 0.0)
    return 0.0


def _annuity_factor_fallback(
    *, age0: int, horizon_years: int, spm: int, r_annual: float, life_table: Optional[pd.DataFrame], immediate: bool
) -> float:
    """
    annuity_stream 모듈이 없을 때의 최소 구현:
      - 생존표 있으면 연간 q_x → 월 q_m 균등위험 가정으로 변환하여 S_t 구성
      - 없으면 S_t = 1 (조심스런 상한)
      - 즉시/한달후 지급 반영
    """
    r_m = (1.0 + float(r_annual)) ** (1.0 / spm) - 1.0
    Tm = int(horizon_years * spm)

    # 생존확률 S_t
    if isinstance(life_table, pd.DataFrame) and not life_table.empty and {"age", "qx"}.issubset(life_table.columns):
        lt = life_table.set_index("age").sort_index()
        ages = _np.arange(age0, age0 + horizon_years + 1, dtype=int)
        qy = _np.clip(lt.loc[ages, "qx"].to_numpy(dtype=float, copy=True), 0.0, 1.0)  # yearly
        q_m = 1.0 - (1.0 - qy) ** (1.0 / spm)
        q_m = _np.repeat(q_m[:-1], spm)  # 마지막 해는 경계
        if q_m.size < Tm:
            q_m = _np.pad(q_m, (0, Tm - q_m.size), constant_values=(q_m[-1] if q_m.size else 0.0))
        log_surv = _np.cumsum(_np.log1p(-_np.clip(q_m[:Tm], 0.0, 1.0)))
        S = _np.exp(_np.insert(log_surv, 0, 0.0))[:-1]
    else:
        S = _np.ones(Tm, dtype=float)

    # 할인계수
    start = 0 if immediate else 1
    disc = (1.0 / (1.0 + r_m)) ** _np.arange(start, start + Tm)
    disc = disc[:Tm]

    return float(_np.sum(S * disc))


# ─────────────────────────────────────────
# main
# ─────────────────────────────────────────
def setup_annuity_overlay(cfg: SimConfig, args) -> Optional[Dict[str, float | int | str | bool]]:
    """
    annuity overlay 사전 배선(자급식 계산 버전).
      - args.ann_on != 'on' 또는 ann_alpha<=0 → OFF
      - life table이 없어도 동작(보수적 S=1), 가능하면 생존표 사용
      - 성공 시 cfg에 (W0_after, y_ann, ann_P, ann_a_factor) 주입
    반환: {'y_ann', 'P', 'a_factor', 'W0_after', 'life_table_used'} 또는 None
    """
    # 1) 토글/파라미터
    ann_on_flag = str(getattr(args, "ann_on", "off") or "off").lower() == "on"
    ann_alpha = _f(getattr(args, "ann_alpha", 0.0), 0.0)
    if (not ann_on_flag) or (ann_alpha <= 0.0):
        _off(cfg)
        return None

    spm = int(getattr(cfg, "steps_per_year", 12) or 12)
    age0 = int(getattr(cfg, "age0", 65) or 65)
    horizon_years = int(getattr(cfg, "horizon_years", 35) or 35)
    index_mode = str(getattr(args, "ann_index", getattr(cfg, "ann_index", "real")) or "real").lower()
    first_immediate = bool(int(getattr(args, "ann_d", getattr(cfg, "ann_d", 0)) or 0) == 0)

    # 2) probe로 보조 정보 확보
    probe = RetirementEnv(cfg)
    life_df = get_life_table_from_env(probe)  # 있을수록 정확
    W0 = _f(getattr(probe, "W0", getattr(cfg, "W0", 1.0)), 1.0)

    # 3) (실질/명목) 연율 추정 → y_ann
    r_m = _avg_monthly_rf(cfg, probe)
    if index_mode == "nominal":
        r_m = r_m + _avg_monthly_cpi(cfg)  # CPI 연동 시 명목화
    y_ann = (1.0 + r_m) ** spm - 1.0

    # 4) a_factor & 지급액 계산
    if _ANN_STREAM:
        spec = AnnuitySpec(
            steps_per_year=spm,
            index_mode=("real" if index_mode == "real" else "nominal"),
            r_f_annual=float(y_ann),
            first_payment_immediate=first_immediate,
        )
        a_fac, _ = annuity_factor(W0=W0, age0=age0, horizon_years=horizon_years, life_table=life_df, spec=spec)
        P = level_payment_from_alpha(W0=W0, alpha=ann_alpha, a_factor=a_fac)
    else:
        a_fac = _annuity_factor_fallback(
            age0=age0, horizon_years=horizon_years, spm=spm, r_annual=float(y_ann),
            life_table=life_df, immediate=first_immediate,
        )
        P = (ann_alpha * W0 / a_fac) if a_fac > 0 else 0.0

    # 남는 초기자산(즉시연금 매입): 단순 (1-α)W0 (L/d 옵션은 후속 반영)
    W0_after = float(max(0.0, W0 * (1.0 - ann_alpha)))

    # 5) cfg 주입
    setattr(cfg, "ann_on", "on")
    setattr(cfg, "W0", W0_after)
    setattr(cfg, "y_ann", _f(y_ann, 0.0))
    setattr(cfg, "ann_P", _f(P, 0.0))
    setattr(cfg, "ann_a_factor", _f(a_fac, 0.0))

    # 러너가 메타를 기록할 수 있도록 도우미 dict 반환
    return {
        "y_ann": float(y_ann),
        "P": float(P),
        "a_factor": float(a_fac),
        "W0_after": float(W0_after),
        "life_table_used": bool(isinstance(life_df, pd.DataFrame) and not life_df.empty),
        "index_mode": ("real" if index_mode == "real" else "nominal"),
        "spm": int(spm),
    }
