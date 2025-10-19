# project/runner/cvar_utils.py
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

# (과거 호환용; 현재는 사용하지 않지만 남겨둠)
try:
    from ..utils.metrics_utils import terminal_losses, cvar_alpha  # type: ignore
except Exception:
    terminal_losses = None  # type: ignore
    cvar_alpha = None       # type: ignore


# ─────────────────────────────────────────────────────────
# Low-level: CVaR(=ES_α) fallback
# ─────────────────────────────────────────────────────────
def cvar_fallback(losses: Iterable[float], alpha: float) -> float:
    """
    L ~ 손실벡터의 CVaR_α(= ES_α) 계산.
    - NaN/Inf 제거, α를 (0,1) 내부로 클램프
    - 정렬 후 α-분위수 지점의 선형보간 + 꼬리평균
    - 반환: float (loss 단위)
    """
    L = np.asarray(list(losses), dtype=float)
    L = L[np.isfinite(L)]
    n = L.size
    if n == 0:
        return 0.0

    a = float(alpha)
    # (0,1) 개방구간으로 안정화
    eps = 1e-12
    a = min(max(a, eps), 1.0 - eps)

    L.sort()
    # j = floor(n*a) (0-index), theta ∈ [0,1)
    j = int(math.floor(n * a))
    j = min(max(j, 0), n - 1)
    theta = n * a - j

    Lj = float(L[j])
    tail_sum = float(L[j + 1 :].sum()) if (j + 1) < n else 0.0
    denom = n * (1.0 - a)
    # Lj 가중 + 잔여 꼬리합 평균
    ES = ((1.0 - theta) * Lj + tail_sum) / max(denom, eps)
    return float(ES)


# ─────────────────────────────────────────────────────────
# WT extraction (best-effort)
# ─────────────────────────────────────────────────────────
_WT_KEYS: Sequence[str] = (
    "eval_WT", "W_T", "WT", "terminal_wealth", "terminal_wealths",
    "paths_WT", "wt_paths", "wealth_terminal", "wealth_T",
)
_CONTAINER_KEYS: Sequence[str] = ("metrics", "extra", "extras", "eval", "payload", "data", "result", "details")


def maybe_extract_WT(candidate: Any) -> Optional[Iterable[float]]:
    """
    다양한 포맷(dict/tuple/list/ndarray/객체)에서 WT 시퀀스를 최대한 찾아냄.
    - (metrics, extras) 튜플 패턴(2번째 dict) 우선
    - dict: 직접 키 → 컨테이너 키 순회
    - ndarray: tolist()
    - 객체: 속성 접근
    """
    if candidate is None:
        return None

    # numpy array
    try:
        if isinstance(candidate, np.ndarray):
            return candidate.tolist()  # type: ignore[return-value]
    except Exception:
        pass

    # list/tuple
    if isinstance(candidate, (list, tuple)):
        # (metrics, extras) 스타일: extras에 우선 존재 가능
        if len(candidate) >= 2 and isinstance(candidate[1], (dict, list, tuple, np.ndarray)):
            wt = maybe_extract_WT(candidate[1])
            if wt is not None:
                return wt
        # 숫자 리스트 자체가 WT일 수도 있음
        if candidate and all(isinstance(x, (int, float, np.floating)) for x in candidate):
            return candidate  # type: ignore[return-value]

    # dict: 우선 직접 키, 그다음 컨테이너 키 순회
    if isinstance(candidate, dict):
        for k in _WT_KEYS:
            if k in candidate and candidate[k] is not None:
                return candidate[k]  # type: ignore[return-value]
        for k in _CONTAINER_KEYS:
            if k in candidate and isinstance(candidate[k], (dict, list, tuple, np.ndarray)):
                wt = maybe_extract_WT(candidate[k])
                if wt is not None:
                    return wt

    # 객체 속성
    for attr in _WT_KEYS:
        try:
            wt = getattr(candidate, attr)
            if wt is not None:
                return wt  # type: ignore[return-value]
        except Exception:
            pass

    # extra/eval-like 속성도 훑기
    for attr in ("extra", "extras", "eval", "payload", "data", "result", "details"):
        try:
            sub = getattr(candidate, attr)
            if isinstance(sub, (dict, list, tuple, np.ndarray)):
                wt = maybe_extract_WT(sub)
                if wt is not None:
                    return wt
        except Exception:
            pass
    return None


# ─────────────────────────────────────────────────────────
# F_target resolver
# ─────────────────────────────────────────────────────────
def resolve_F_for_cvar(args, out: Dict[str, Any]) -> float:
    """
    F_target 결정 우선순위:
      1) out.cvar_calibration.selected_F_target / F_selected
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


# ─────────────────────────────────────────────────────────
# Main fixer
# ─────────────────────────────────────────────────────────
def fixup_metrics_with_cvar(args, out: Dict[str, Any]) -> Dict[str, Any]:
    """
    es_mode='loss'이고 F_target>0인 경우, WT가 있으면 ES95/EL을 재계산해 메트릭 보강.
    - evaluate가 이미 CVaR을 계산해 'es95_source'를 남긴 경우는 존중(스킵).
    - EW/mean_WT, Ruin도 함께 보강(가능 시).
    반환: 입력 dict(out)를 제자리 갱신 후 그대로 반환.
    """
    metrics = out["metrics"] if "metrics" in out and isinstance(out["metrics"], dict) else out
    es_mode = str(getattr(args, "es_mode", "wealth")).lower()
    F_target = resolve_F_for_cvar(args, out if isinstance(out, dict) else {})
    alpha_v = float(getattr(args, "alpha", 0.95) or 0.95)

    # 모드/기준선 체크
    if es_mode != "loss" or F_target <= 0.0:
        metrics["es_mode"] = es_mode
        if F_target > 0:
            metrics.setdefault("F_target_used", F_target)
        return out

    # 이미 evaluate에서 계산했다면 존중
    if isinstance(metrics, dict):
        src = str(metrics.get("es95_source", "") or "")
        if "computed_in_evaluate" in src or "path_level_cvar" in src:
            metrics["es_mode"] = es_mode
            metrics.setdefault("F_target_used", F_target)
            return out

    # WT 탐색
    WT = None
    for cand in (out, metrics):
        WT = maybe_extract_WT(cand)
        if WT is not None:
            break

    if WT is None:
        # WT 없으면 계산 불가. 힌트 남기고 종료
        try:
            EW0 = float(metrics.get("EW", metrics.get("mean_WT", 0.0)))
            ES0 = float(metrics.get("ES95", 0.0))
            if abs((F_target - EW0) - ES0) < 1e-9 and ES0 > 0:
                metrics["es95_note"] = (
                    "ES95 looks like F_target - EW (no W_T available). "
                    "Expose path-level W_T from evaluate to recompute."
                )
        except Exception:
            pass
        metrics["es_mode"] = es_mode
        metrics.setdefault("F_target_used", F_target)
        return out

    # 재계산
    try:
        WT_arr = np.asarray(list(WT), dtype=float)
        WT_arr = WT_arr[np.isfinite(WT_arr)]
        if WT_arr.size == 0:
            raise ValueError("empty WT after cleaning")

        # 손실 벡터 L = max(F - WT, 0)
        if terminal_losses is not None and cvar_alpha is not None:
            L = terminal_losses(WT_arr, F_target)
            ES = cvar_alpha(L, alpha=alpha_v)
            EL = float(np.mean(L))
        else:
            L = np.maximum(F_target - WT_arr, 0.0)
            ES = cvar_fallback(L, alpha=alpha_v)
            EL = float(np.mean(L))

        # EW/mean_WT 보강
        EW = float(metrics.get("EW", metrics.get("mean_WT", float(np.mean(WT_arr)))))
        metrics["EW"] = EW
        metrics["mean_WT"] = EW

        # 손실 메트릭
        metrics["EL"] = EL
        metrics["ES95"] = float(ES)

        # Ruin(가능 시)
        try:
            metrics["Ruin"] = float((WT_arr <= 0.0).mean())
        except Exception:
            pass

        # 태그
        metrics["es_mode"] = es_mode
        metrics["es95_source"] = "path_level_cvar"
        metrics["F_target_used"] = float(F_target)
    except Exception as e:
        metrics["es_mode"] = es_mode
        metrics["F_target_used"] = float(F_target)
        metrics["es95_note"] = f"failed to recompute ES95: {type(e).__name__}"

    return out
