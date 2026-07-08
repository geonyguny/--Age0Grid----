# project/stats/ci.py
from __future__ import annotations
import numpy as np
from typing import Tuple, Literal, Callable

ArrayLike = np.ndarray

def bootstrap_ci(
    arr: ArrayLike, q: Tuple[float, float] = (0.025, 0.975), B: int = 2000, seed: int | None = None,
    stat: Literal["mean","median"] = "mean"
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(arr, dtype=float)
    n = arr.shape[0]
    if n == 0:
        return (np.nan, np.nan)
    fn: Callable[[ArrayLike], float] = np.nanmean if stat=="mean" else np.nanmedian
    boots = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boots[b] = fn(arr[idx])
    lo, hi = np.quantile(boots, q)
    return float(lo), float(hi)

def compare_diff_ci(
    arr_a: ArrayLike, arr_b: ArrayLike, B: int = 5000, seed: int | None = None,
    stat: Literal["mean","median"] = "mean", q: Tuple[float,float]=(0.025,0.975)
) -> tuple[float,float,float]:
    """return (diff_point, lo, hi) for stat(A)-stat(B)"""
    rng = np.random.default_rng(seed)
    A, Bv = np.asarray(arr_a, float), np.asarray(arr_b, float)
    fn: Callable[[ArrayLike], float] = np.nanmean if stat=="mean" else np.nanmedian
    nA, nB = A.shape[0], Bv.shape[0]
    boots = np.empty(B, dtype=float)
    for b in range(B):
        iA = rng.integers(0, nA, size=nA)
        iB = rng.integers(0, nB, size=nB)
        boots[b] = fn(A[iA]) - fn(Bv[iB])
    lo, hi = np.quantile(boots, q)
    point = fn(A) - fn(Bv)
    return float(point), float(lo), float(hi)

def perm_test_diff(
    arr_a: ArrayLike, arr_b: ArrayLike, B: int = 5000, seed: int | None = None,
    stat: Literal["mean","median"] = "mean", alternative: Literal["two-sided","greater","less"]="two-sided"
) -> float:
    """Permutation test p-value of stat(A)-stat(B)."""
    rng = np.random.default_rng(seed)
    A, Bv = np.asarray(arr_a, float), np.asarray(arr_b, float)
    fn: Callable[[ArrayLike], float] = np.nanmean if stat=="mean" else np.nanmedian
    obs = fn(A) - fn(Bv)
    pool = np.concatenate([A, Bv])
    nA = A.shape[0]
    count = 0
    for b in range(B):
        rng.shuffle(pool)
        sA = fn(pool[:nA]); sB = fn(pool[nA:])
        diff = sA - sB
        if alternative == "two-sided":
            if abs(diff) >= abs(obs): count += 1
        elif alternative == "greater":
            if diff >= obs: count += 1
        else:
            if diff <= obs: count += 1
    p = (count + 1.0) / (B + 1.0)
    return float(p)
