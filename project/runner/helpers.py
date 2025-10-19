# project/runner/helpers.py
from __future__ import annotations
import hashlib
from typing import Optional, Tuple, Any
import numpy as _np
import pandas as pd


def arrhash(a: Any) -> str:
    """
    안정적인 배열 해시.
    - dtype/shape/바이트를 모두 반영해 충돌을 줄임
    - NaN/Inf를 고정 패턴으로 정규화(플랫폼별 NaN 비트차이 영향 최소화)
    - None이면 'none'
    """
    if a is None:
        return "none"
    arr = _np.asarray(a)

    # 수치 안정성: float은 float32로 통일 + 비유한/비정상값 정규화
    if _np.issubdtype(arr.dtype, _np.floating):
        arr = arr.astype(_np.float32, copy=False)
        # NaN/Inf를 고정 상수로 정규화(해시 일관성 ↑)
        if _np.isnan(arr).any() or _np.isinf(arr).any():
            arr = arr.copy()
            _np.nan_to_num(arr, copy=False, nan=_np.float32('nan'), posinf=_np.float32(3.4028235e38), neginf=_np.float32(-3.4028235e38))

    h = hashlib.md5()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(str(tuple(arr.shape)).encode("utf-8"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def auto_eta_grid(cfg: Any, requested_n: int | None = None) -> None:
    """
    HJB에서 사용하는 RU-dual η 그리드를 자동 구성.
    - lambda_term <= 0 → (0.0,) 고정
    - 그렇지 않으면 [0, F_target] 구간을 균등분할
      · 점 개수: requested_n 또는 cfg.hjb_eta_n 또는 41
      · F_target<=0이면 보수적으로 1.0 사용
      · 수치안정: 단조증가/중복제거/형변환 보장
    """
    lam = float(getattr(cfg, "lambda_term", 0.0) or 0.0)
    cur = getattr(cfg, "hjb_eta_grid", ())

    if lam <= 0.0:
        if not cur or len(cur) <= 1:
            cfg.hjb_eta_grid = (0.0,)
        return

    # 그리드 포인트 수
    n = int(requested_n or getattr(cfg, "hjb_eta_n", 41) or 41)
    n = max(2, n)

    # 기준선 F
    F = float(getattr(cfg, "F_target", 0.0) or 0.0)
    if not _np.isfinite(F) or F <= 0.0:
        F = 1.0

    try:
        grid = _np.linspace(0.0, F, n, dtype=_np.float64)
    except Exception:
        step = F / max(n - 1, 1)
        grid = _np.array([0.0 + i * step for i in range(n)], dtype=_np.float64)

    # 단조/중복/범위 보정
    grid = _np.clip(grid, 0.0, max(F, 0.0))
    grid = _np.unique(grid)  # 중복 제거
    if grid.size < 2:
        grid = _np.array([0.0, float(F)], dtype=_np.float64)

    cfg.hjb_eta_grid = tuple(float(x) for x in grid)


def get_life_table_from_env(env: Any) -> Optional[pd.DataFrame]:
    """
    RetirementEnv 유사 객체에서 생명표 DataFrame을 얻는다.
    우선순위: env.life_table → env.mort_table_df → None
    """
    lt = getattr(env, "life_table", None)
    if isinstance(lt, pd.DataFrame) and not lt.empty:
        return lt
    lt2 = getattr(env, "mort_table_df", None)
    if isinstance(lt2, pd.DataFrame) and not lt2.empty:
        return lt2
    return None


def monthly_from_cfg(cfg: Any) -> Tuple[float, float]:
    """
    (g_m, p_m)를 반환.
    - cfg.monthly()가 있으면 그 값을 신뢰하되 키 없으면 0.0으로 보정
    - 없으면 연 인자에서 월 인자로 변환:
        g_m = (1 + g_ann)^(1/spm) - 1
        p_m = 1 - (1 - p_ann)^(1/spm)   ← 소비 '비율'의 월 등가
    """
    # 우선 사용자 제공 monthly() 사용
    if hasattr(cfg, "monthly") and callable(getattr(cfg, "monthly")):
        try:
            m = cfg.monthly()
            if isinstance(m, dict):
                g_m = float(m.get("g_m", 0.0) or 0.0)
                p_m = float(m.get("p_m", 0.0) or 0.0)
                # 비정상값 가드
                if not _np.isfinite(g_m): g_m = 0.0
                if not _np.isfinite(p_m): p_m = 0.0
                return g_m, p_m
        except Exception:
            pass

    spm = int(getattr(cfg, "steps_per_year", 12) or 12)
    spm = max(1, spm)

    g_ann = float(getattr(cfg, "g_real_annual", 0.0) or 0.0)
    p_ann = float(getattr(cfg, "p_annual", 0.0) or 0.0)

    # 성장률: 표준 기하 변환
    g_m = (1.0 + g_ann) ** (1.0 / spm) - 1.0
    # 소비비율: 연간 소비비율 → 월 등가
    p_ann = float(_np.clip(p_ann, 0.0, 0.999999))  # 수치 안정화
    p_m = 1.0 - (1.0 - p_ann) ** (1.0 / spm)

    return float(g_m), float(p_m)
