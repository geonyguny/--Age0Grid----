# project/runner/cvar_utils.py
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional

import numpy as np

# (과거 호환용; 현재는 사용하지 않지만 남겨둠)
try:
    from ..utils.metrics_utils import terminal_losses, cvar_alpha  # type: ignore
except Exception:
    terminal_losses = None  # type: ignore
    cvar_alpha = None       # type: ignore


# -------------------------
# Low-level CVaR fallback
# -------------------------
def cvar_fallback(losses: Iterable[float], alpha: float) -> float:
    """
    L의 CVaR_α를 계산 (정렬-꼬리 평균; 선형 보간 포함).
    - NaN/Inf 제거
    - α 경계 안정화
    """
    L = np.asarray(list(losses), dtype=float)
    # 정화
    L = L[np.isfinite(L)]
    n = L.size
    if n == 0:
        return 0.0

    a = float(alpha)
    a = max(min(a, 1.0 - 1e-12), 1e-12)

    L.sort()
    j = int(math.floor(n * a))
    j = min(max(j, 0), n - 1)
    theta = n * a - j  # [0,1)

    Lj1 = float(L[j])
    tail_sum = float(L[j + 1:].sum()) if (j + 1) < n else 0.0
    ES = ((1.0 - theta) * Lj1 + tail_sum) / (n * (1.0 - a))
    return float(ES)


# -------------------------
# WT extraction
# -------------------------
_WT_KEYS = [
    "eval_WT", "W_T", "WT", "terminal_wealth", "terminal_wealths",
    "paths_WT", "wt_paths", "wealth_terminal", "wealth_T",
]

_CONTAINER_KEYS = ["metrics", "extra", "extras", "eval", "payload", "data", "result", "details"]


def maybe_extract_WT(candidate: Any) -> Optional[Iterable[float]]:
    """
    다양한 포맷의 객체(dict/tuple(list)/ndarray/obj)에서 WT 시퀀스를 최대한 찾아낸다.
    - (metrics, extras) 튜플도 지원
    - dict: 위계 탐색(_CONTAINER_KEYS), WT 유력 키(_WT_KEYS)
    - ndarray: tolist()
    - 객체 속성 접근
    """
    if candidate is None:
        return None

    # numpy array
    try:
        import numpy as _np  # noqa: F401
        if isinstance(candidate, _np.ndarray):
            return candidate.tolist()  # type: ignore
    except Exception:
        pass

    # list/tuple
    if isinstance(candidate, (list, tuple)):
        # (metrics, extras) 스타일: extras 안을 먼저 본다
        if len(candidate) >= 2 and isinstance(candidate[1], (dict, list, tuple)):
            wt = maybe_extract_WT(candidate[1])
            if wt is not None:
                return wt
        # 숫자 리스트 자체가 WT일 수도 있음
        if candidate and all(isinstance(x, (int, float)) for x in candidate):
            return candidate  # type: ignore

    # dict: 우선 직접 키, 그다음 컨테이너 키 순회
    if isinstance(candidate, dict):
        for k in _WT_KEYS:
            if k in candidate and candidate[k] is not None:
                return candidate[k]  # type: ignore
        for k in _CONTAINER_KEYS:
            if k in candidate and isinstance(candidate[k], (dict, list, tuple)):
                wt = maybe_extract_WT(candidate[k])
                if wt is not None:
                    return wt

    # 객체 속성
    for attr in _WT_KEYS:
        try:
            wt = getattr(candidate, attr)
            if wt is not None:
                return wt  # type: ignore
        except Exception:
            pass

    # extra/eval-like 속성도 훑기
    for attr in ["extra", "extras", "eval", "payload", "data", "result", "details"]:
        try:
            sub = getattr(candidate, attr)
            if isinstance(sub, (dict, list, tuple)):
                wt = maybe_extract_WT(sub)
                if wt is not None:
                    return wt
        except Exception:
            pass
    return None


# -------------------------
# F_target resolver
# -------------------------
def resolve_F_for_cvar(args, out: Dict[str, Any]) -> float:
    """
    우선순위:
      1) out.cvar_calibration.selected_F_target/F_selected
      2) out.F_target
      3) args.F_target
      (없으면 0.0)
    """
    try:
        cc = out.get("cvar_calibration", {}) if isinstance(out, dict) else {}
        sf = cc.get("selected_F_target") or cc.get("F_selected")
        if isinstance(sf, (int, float)):
            return float(sf)
    except Exception:
        pass
    try:
        ft = out.get("F_target") if isinstance(out, dict) else None
        if isinstance(ft, (int, float)):
            return float(ft)
    except Exception:
        pass
    try:
        return float(getattr(args, "F_target", 0.0) or 0.0)
    except Exception:
        return 0.0


# -------------------------
# Main fixer
# -------------------------
def fixup_metrics_with_cvar(args, out: Dict[str, Any]) -> Dict[str, Any]:
    """
    es_mode=loss + F_target>0인 경우, WT가 있으면 ES95/EL을 재계산해 메트릭을 보강한다.
    단, evaluate가 이미 계산했음을 나타내는 es95_source가 있으면 재계산을 생략.
    """
    metrics = out["metrics"] if "metrics" in out and isinstance(out["metrics"], dict) else out
    es_mode = str(getattr(args, "es_mode", "wealth")).lower()
    F_target = resolve_F_for_cvar(args, out if isinstance(out, dict) else {})
    alpha_v = float(getattr(args, "alpha", 0.95) or 0.95)

    # 모드/기준선 체크
    if es_mode != "loss" or F_target <= 0.0:
        metrics["es_mode"] = es_mode
        metrics.setdefault("F_target_used", F_target if F_target > 0 else None)
        return out

    # 이미 evaluate에서 책임지고 계산했으면 존중
    if isinstance(metrics, dict):
        src = str(metrics.get("es95_source", "") or "")
        if "computed_in_evaluate" in src or "path_level_cvar" in src:
            metrics["es_mode"] = es_mode
            metrics.setdefault("F_target_used", F_target if F_target > 0 else None)
            return out

    # WT 탐색
    WT = None
    for cand in (out, metrics):
        WT = maybe_extract_WT(cand)
        if WT is not None:
            break

    if WT is None:
        # WT 없으면 손댈 수 없음. 그래도 힌트 남겨두기
        try:
            EW = float(metrics.get("EW", 0.0) or metrics.get("mean_WT", 0.0) or 0.0)
            ES_old = float(metrics.get("ES95", 0.0) or 0.0)
            if abs((EW + ES_old) - F_target) < 1e-9 and ES_old > 0:
                metrics["es95_note"] = "ES95 looks like (F_target - EW). No W_T to recompute; please expose path-level W_T from evaluate."
        except Exception:
            pass
        metrics["es_mode"] = es_mode
        metrics.setdefault("F_target_used", F_target if F_target > 0 else None)
        return out

    # 재계산
    try:
        WT_arr = np.asarray(list(WT), dtype=float)
        # 정화
        WT_arr = WT_arr[np.isfinite(WT_arr)]

        if WT_arr.size == 0:
            raise ValueError("empty WT after cleaning")

        if terminal_losses is not None and cvar_alpha is not None:
            L = terminal_losses(WT_arr, F_target)
            ES = cvar_alpha(L, alpha=alpha_v)
            EL = float(np.mean(L))
        else:
            L = np.maximum(F_target - WT_arr, 0.0)
            ES = cvar_fallback(L, alpha=alpha_v)
            EL = float(np.mean(L))

        # EW/mean_WT 보강
        EW = float(metrics.get("EW", 0.0) or metrics.get("mean_WT", 0.0) or float(np.mean(WT_arr)))
        metrics["EW"] = EW
        metrics["mean_WT"] = EW

        # 손실 메트릭
        metrics["EL"] = EL
        metrics["ES95"] = float(ES)

        # 기타
        try:
            metrics["Ruin"] = float((WT_arr <= 0.0).mean())
        except Exception:
            pass

        metrics["es_mode"] = es_mode
        metrics["es95_source"] = "path_level_cvar"
        metrics["F_target_used"] = float(F_target)
    except Exception as e:
        metrics["es_mode"] = es_mode
        metrics["F_target_used"] = float(F_target)
        metrics["es95_note"] = f"failed to recompute ES95: {type(e).__name__}"
    return out
