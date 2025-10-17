# project/metrics/es.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np

__all__ = ["es95_wealth", "es_alpha_wealth", "cvar_alpha_loss"]

def _finite1d(x) -> np.ndarray:
    """입력을 1D float 배열로 캐스팅하고 비유한값 제거."""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        return arr
    m = np.isfinite(arr)
    return arr[m]

def _quantile(arr: np.ndarray, q: float) -> float:
    """NumPy 버전 차이를 흡수한 분위수 계산(빈/단일 샘플도 방어)."""
    a = _finite1d(arr)
    if a.size == 0:
        return float("nan")
    q = float(min(max(q, 0.0), 1.0))
    try:
        # NumPy >= 1.22
        return float(np.quantile(a, q, method="higher"))
    except TypeError:
        # NumPy < 1.22
        return float(np.quantile(a, q))

def es_alpha_wealth(wealth_terminal: np.ndarray, alpha: float = 0.95) -> float:
    """
    ES_α(wealth): 하위 (1-α) 구간(왼쪽 꼬리)의 평균 자산.
    - wealth가 낮을수록 위험하므로, ES 값이 클수록 '더 안전'.
    - NaN/Inf/빈 입력을 방어.
    """
    wt = _finite1d(wealth_terminal)
    if wt.size == 0:
        return 0.0
    q = _quantile(wt, 1.0 - float(alpha))  # 예: α=0.95 → 하위 5% 경계
    if not np.isfinite(q):
        # 모두 동일 NaN 등 이슈가 있으면 평균으로 폴백
        return float(np.nanmean(wt)) if wt.size else 0.0
    tail = wt[wt <= q]
    if tail.size == 0:
        # 극단적으로 모든 값이 동일해 tail이 비면 q 자체를 사용
        return float(q)
    return float(np.mean(tail))

def es95_wealth(wealth_terminal: np.ndarray, alpha: float = 0.95) -> float:
    """호환용 별칭: ES95(wealth)."""
    return es_alpha_wealth(wealth_terminal, alpha=alpha)

def cvar_alpha_loss(losses: np.ndarray, alpha: float = 0.95) -> float:
    """
    CVaR_α(loss): VaR_α 이상의 손실의 평균.
    - 손실분포 L에서 VaR_α = quantile(L, α)
    - ES(CVaR) = E[L | L ≥ VaR_α]
    """
    L = _finite1d(losses)
    if L.size == 0:
        return 0.0
    q = _quantile(L, float(alpha))
    if not np.isfinite(q):
        return float(np.nanmean(L)) if L.size else 0.0
    tail = L[L >= q]
    return float(np.mean(tail)) if tail.size > 0 else float(q)
