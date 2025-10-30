# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from typing import Iterable, Literal, Tuple, Optional

Tail = Literal["left", "right"]

def _as_array(x: Iterable[float]) -> np.ndarray:
    a = np.asarray(list(x), dtype=float)
    if a.ndim != 1:
        a = a.reshape(-1)
    return a[~np.isnan(a)]

def var_at_p(values: Iterable[float], p: float = 0.95, tail: Tail = "left") -> float:
    """
    VaR_p: 왼쪽 꼬리(tail='left')면 p-quantile (예: 95%면 왼쪽 5% 경계)
    오른쪽 꼬리는 반대로 처리. 금융 관례상 손실은 왼쪽 꼬리.
    """
    v = _as_array(values)
    if v.size == 0:
        return float("nan")
    q = 1.0 - p if tail == "right" else p
    return float(np.quantile(v, q, interpolation="linear"))

def es_at_p(values: Iterable[float], p: float = 0.95, tail: Tail = "left") -> Tuple[float, int]:
    """
    ES_p (Expected Shortfall): VaR 경계 바깥(더 나쁜) 구간의 평균.
    tail='left'면 값이 VaR 이하인 샘플들의 평균.
    반환: (ES, 사용표본수)
    """
    v = _as_array(values)
    if v.size == 0:
        return float("nan"), 0
    var = var_at_p(v, p=p, tail=tail)
    if tail == "left":
        mask = v <= var
    else:
        mask = v >= var
    sel = v[mask]
    if sel.size == 0:
        return float("nan"), 0
    return float(sel.mean()), int(sel.size)

def es95_loss_from_wealth(wealth: Iterable[float]) -> Tuple[float, int]:
    """
    'wealth'로부터 손실 기준 ES95 계산.
    관례: 손실 L = -wealth (wealth가 낮을수록 손실 큼) → 왼쪽 꼬리 ES.
    """
    w = _as_array(wealth)
    loss = -w
    es, n = es_at_p(loss, p=0.95, tail="left")
    return es, n
