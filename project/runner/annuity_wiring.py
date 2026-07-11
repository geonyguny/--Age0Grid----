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
        AnnuitySpec,
        annuity_factor,
        level_payment_from_alpha,
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
    setattr(cfg, "ann_alpha", 0.0)
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
    *,
    age0: int,
    horizon_years: int,
    spm: int,
    r_annual: float,
    life_table: Optional[pd.DataFrame],
    immediate: bool,
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
    if (
        isinstance(life_table, pd.DataFrame)
        and not life_table.empty
        and {"age", "qx"}.issubset(life_table.columns)
    ):
        lt = life_table.set_index("age").sort_index()
        ages = _np.arange(age0, age0 + horizon_years + 1, dtype=int)
        qy = _np.clip(
            lt.loc[ages, "qx"].to_numpy(dtype=float, copy=True),
            0.0,
            1.0,
        )  # yearly
        q_m = 1.0 - (1.0 - qy) ** (1.0 / spm)
        q_m = _np.repeat(q_m[:-1], spm)  # 마지막 해는 경계
        if q_m.size < Tm:
            q_m = _np.pad(
                q_m,
                (0, Tm - q_m.size),
                constant_values=(q_m[-1] if q_m.size else 0.0),
            )
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
def setup_annuity_overlay(
    cfg: SimConfig, args
) -> Optional[Dict[str, float | int | str | bool]]:
    """
    annuity overlay 사전 배선(자급식 계산 버전).

    정책:
      - ann_on 해석
        * ann_on 이 'on'  → 무조건 ON
        * ann_on 이 'off' → 무조건 OFF
        * ann_on 이 없거나 'auto' → ann_alpha > 0 이면 ON, 아니면 OFF
      - ann_alpha 는 args → cfg 순으로 확보
      - life table 이 없어도 동작(보수적 S=1), 가능하면 생존표 사용
      - 성공 시 cfg 에 (W0_after, y_ann, ann_P, ann_a_factor, ann_on='on') 주입

    반환:
      {'y_ann', 'P', 'a_factor', 'W0_after', 'life_table_used', ...} 또는 None
    """

    # 1) ann_alpha 확보 (args 우선, 없으면 cfg)
    def _get_ann_alpha(_cfg: SimConfig, _args: Any) -> float:
        for src in (_args, _cfg):
            try:
                val = getattr(src, "ann_alpha", None)
                if val is not None:
                    return _f(val, 0.0)
            except Exception:
                continue
        return 0.0

    ann_alpha = _get_ann_alpha(cfg, args)

    # 2) ann_on 플래그 해석
    raw_flag = getattr(args, "ann_on", None)
    flag_s = str(raw_flag).strip().lower() if raw_flag is not None else ""

    if flag_s in ("", "auto"):
        # auto 모드: ann_alpha > 0 이면 ON
        ann_on_flag = ann_alpha > 0.0
    else:
        # 명시 모드: on/off 그대로 사용
        ann_on_flag = (flag_s == "on")

    if (not ann_on_flag) or (ann_alpha <= 0.0):
        _off(cfg)
        return None

    # cfg 에 최종 ann_alpha 반영
    setattr(cfg, "ann_alpha", float(ann_alpha))

    # 3) 기본 파라미터
    spm = int(getattr(cfg, "steps_per_year", 12) or 12)
    age0 = int(getattr(cfg, "age0", 55) or 55)
    horizon_years = int(getattr(cfg, "horizon_years", 35) or 35)

    index_mode = str(
        getattr(args, "ann_index", getattr(cfg, "ann_index", "real"))
        or "real"
    ).lower()
    first_immediate = bool(
        int(getattr(args, "ann_d", getattr(cfg, "ann_d", 0)) or 0) == 0
    )

    # 4) probe 환경 생성 및 보조 정보 확보
    probe = RetirementEnv(cfg)
    life_df = get_life_table_from_env(probe)  # 있을수록 정확
    W0 = _f(getattr(probe, "W0", getattr(cfg, "W0", 1.0)), 1.0)

    # 5) (실질/명목) 연율 추정 → annuity_factor 계산용 할인율(r_ann_rate)
    #    [FIX 2026-07] 이 지역변수는 "연금 계수 계산용 할인율"일 뿐 실제 지급액이
    #    아닌데, 기존 코드는 변수명이 똑같이 "y_ann"이라 아래 7)에서 실수로
    #    cfg.y_ann에 이 할인율(예: 연 2%대)을 그대로 대입하고 있었다. 그 결과
    #    HJB/평가 파이프라인이 실제 지급액(P, 보통 연 1~2%대 수준으로 훨씬 작음)
    #    대신 훨씬 큰 가짜 소득을 "월 연금지급액"으로 착각하는 버그가 있었다.
    #    (HJB가 실제보다 훨씬 큰 연금소득을 믿고 개인자산을 과도하게 인출하도록
    #    유도해, θ>0 케이스에서 EU가 오히려 급격히 악화되는 현상의 원인이었음.)
    r_m = _avg_monthly_rf(cfg, probe)
    if index_mode == "nominal":
        # CPI 연동 시 명목화
        r_m = r_m + _avg_monthly_cpi(cfg)
    r_ann_rate = (1.0 + r_m) ** spm - 1.0

    # 6) a_factor & 지급액 계산
    if _ANN_STREAM:
        spec = AnnuitySpec(
            steps_per_year=spm,
            index_mode=("real" if index_mode == "real" else "nominal"),
            r_f_annual=float(r_ann_rate),
            first_payment_immediate=first_immediate,
        )
        a_fac, _ = annuity_factor(
            W0=W0,
            age0=age0,
            horizon_years=horizon_years,
            life_table=life_df,
            spec=spec,
        )
        P = level_payment_from_alpha(
            W0=W0,
            alpha=ann_alpha,
            a_factor=a_fac,
        )
    else:
        a_fac = _annuity_factor_fallback(
            age0=age0,
            horizon_years=horizon_years,
            spm=spm,
            r_annual=float(r_ann_rate),
            life_table=life_df,
            immediate=first_immediate,
        )
        P = (ann_alpha * W0 / a_fac) if a_fac > 0 else 0.0

    # 남는 초기자산(즉시연금 매입): 단순 (1-α)W0
    W0_after = float(max(0.0, W0 * (1.0 - ann_alpha)))

    # 7) cfg 주입
    # [FIX 2026-07] cfg.y_ann은 "실제 스텝당 연금 지급액"이어야 하므로 P를 대입한다
    # (기존엔 위의 r_ann_rate가 잘못 들어가고 있었음).
    setattr(cfg, "ann_on", "on")
    setattr(cfg, "W0", W0_after)
    setattr(cfg, "y_ann", _f(P, 0.0))
    setattr(cfg, "ann_P", _f(P, 0.0))
    setattr(cfg, "ann_a_factor", _f(a_fac, 0.0))
    setattr(cfg, "ann_index", "real" if index_mode == "real" else "nominal")

    # 러너가 메타를 기록할 수 있도록 도우미 dict 반환
    return {
        "y_ann": float(P),
        "P": float(P),
        "a_factor": float(a_fac),
        "W0_after": float(W0_after),
        "life_table_used": bool(
            isinstance(life_df, pd.DataFrame) and not life_df.empty
        ),
        "index_mode": ("real" if index_mode == "real" else "nominal"),
        "spm": int(spm),
    }


def add_annuity_metrics(
    metrics: Dict[str, Any],
    cfg: SimConfig,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    metrics dict 에 annuity 관련 필드를 주입하는 헬퍼.
    - cfg: run 시점에 사용한 SimConfig (ann_on, ann_alpha, ann_load, ann_index 등)
    - meta: setup_annuity_overlay(...) 가 반환한 dict (선택사항)
    """
    # 기본 설정값
    metrics["ann_on"] = getattr(cfg, "ann_on", "off")
    metrics["ann_alpha"] = float(getattr(cfg, "ann_alpha", 0.0))
    # ann_load가 따로 없으면 phi_adval 사용
    metrics["ann_load"] = float(
        getattr(cfg, "ann_load", getattr(cfg, "phi_adval", 0.0))
    )
    metrics["ann_index"] = getattr(cfg, "ann_index", "real")

    if meta is not None:
        metrics.setdefault("y_ann", float(meta.get("y_ann", 0.0)))
        metrics.setdefault("ann_P", float(meta.get("P", 0.0)))
        metrics.setdefault(
            "ann_a_factor", float(meta.get("a_factor", 0.0))
        )
        metrics.setdefault(
            "ann_W0_after",
            float(meta.get("W0_after", getattr(cfg, "W0", 0.0))),
        )
    else:
        metrics.setdefault("y_ann", float(getattr(cfg, "y_ann", 0.0)))
        metrics.setdefault("ann_P", float(getattr(cfg, "ann_P", 0.0)))
        metrics.setdefault(
            "ann_a_factor", float(getattr(cfg, "ann_a_factor", 0.0))
        )
