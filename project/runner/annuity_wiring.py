# project/runner/annuity_wiring.py
from __future__ import annotations
from typing import Optional, Any
import numpy as _np

# 러너 내 다른 파일들과 동일한 경로 규칙 유지
from ..env.retirement_env import RetirementEnv  # type: ignore
from ..config import SimConfig
from .helpers import get_life_table_from_env

# overlay는 있을 수도/없을 수도 있으므로 로드 자체를 방호
try:
    from ..annuity.overlay import AnnuityConfig, init_annuity  # type: ignore
    _ANN_AVAILABLE = True
except Exception:
    AnnuityConfig = object  # type: ignore
    init_annuity = None     # type: ignore
    _ANN_AVAILABLE = False


def _f(x: Any, d: float = 0.0) -> float:
    """float 캐스팅 + NaN/Inf 방호."""
    try:
        v = float(x)
        if _np.isfinite(v):
            return v
        return float(d)
    except Exception:
        return float(d)


def _off(cfg: SimConfig) -> None:
    """cfg를 'annuity off' 상태로 정리."""
    setattr(cfg, "ann_on", "off")
    setattr(cfg, "y_ann", 0.0)
    setattr(cfg, "ann_P", 0.0)
    setattr(cfg, "ann_a_factor", 0.0)


def setup_annuity_overlay(cfg: SimConfig, args) -> Optional[object]:
    """
    annuity overlay 사전 배선:
      - args.ann_on != 'on' 또는 ann_alpha<=0 → 즉시 OFF
      - life table/overlay 모듈 없으면 OFF
      - 성공 시 cfg에 (W0_after, y_ann, ann_P, ann_a_factor) 주입
    반환: overlay state (성공 시) / None (OFF 또는 실패)
    """
    # 1) 토글/파라미터 확인
    ann_on_flag = str(getattr(args, "ann_on", "off") or "off").lower() == "on"
    ann_alpha = _f(getattr(args, "ann_alpha", 0.0), 0.0)
    if (not ann_on_flag) or (ann_alpha <= 0.0) or (not _ANN_AVAILABLE):
        _off(cfg)
        return None

    # 2) 생명표 필수 — 없으면 OFF
    probe = RetirementEnv(cfg)
    life_df = get_life_table_from_env(probe)
    if life_df is None:
        _off(cfg)
        return None

    # 3) 할인율(실질 무위험 연율) 결정
    if getattr(probe, "r_f_real_annual", None) is not None:
        r_annual = _f(getattr(probe, "r_f_real_annual"), 0.02)
    else:
        spm = int(getattr(cfg, "steps_per_year", 12) or 12)
        r_m = _f(_np.mean(getattr(probe, "path_safe", _np.array([0.0], dtype=float))), 0.0)
        r_annual = (1.0 + r_m) ** spm - 1.0

    # 4) overlay 구성 및 실행(실패시 OFF)
    try:
        age0 = int(getattr(cfg, "age0", 65) or 65)
        ann_cfg = AnnuityConfig(  # type: ignore[call-arg]
            on=True,
            alpha=ann_alpha,
            L=_f(getattr(args, "ann_L", 0.0), 0.0),
            d=int(getattr(args, "ann_d", 0) or 0),
            index=str(getattr(args, "ann_index", "real") or "real"),
        )
        W0 = _f(getattr(probe, "W0", getattr(cfg, "W0", 1.0)), 1.0)
        # init_annuity(W0_before, config, age0, life_table, r_f_real_annual?)
        W0_after, st = init_annuity(W0, ann_cfg, age0, life_df, r_annual)  # type: ignore[misc]

        # 5) cfg에 결과 주입
        setattr(cfg, "ann_on", "on")
        setattr(cfg, "W0", _f(W0_after, W0))
        setattr(cfg, "y_ann", _f(getattr(st, "y_ann", 0.0), 0.0))
        setattr(cfg, "ann_P", _f(getattr(st, "P", 0.0), 0.0))
        setattr(cfg, "ann_a_factor", _f(getattr(st, "a_factor", 0.0), 0.0))
        return st
    except Exception:
        _off(cfg)
        return None
