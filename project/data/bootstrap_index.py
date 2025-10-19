# project/data/bootstrap_index.py
from __future__ import annotations
import numpy as np
from typing import Tuple

def make_block_bootstrap_indices(T: int, block: int, total_len: int, seed: int = 0) -> np.ndarray:
    """
    길이 T의 원본 시계열에서 블록 길이 'block'로 이어 붙여 total_len 길이의 인덱스 시퀀스를 생성.
    반환 shape = (total_len,)
    """
    T = int(T); block = int(block); total_len = int(total_len)
    if T <= 0 or block <= 0 or total_len <= 0:
        raise ValueError("T, block, total_len must be positive")
    rng = np.random.default_rng(seed)
    out = np.empty(total_len, dtype=int)
    pos = 0
    while pos < total_len:
        start = int(rng.integers(0, max(1, T - block)))
        seg_len = min(block, total_len - pos)
        out[pos:pos+seg_len] = np.arange(start, start + seg_len) % T
        pos += seg_len
    return out

def apply_bootstrap(series: np.ndarray, idx: np.ndarray) -> np.ndarray:
    if series.ndim == 1:
        return series[idx]
    # 열 여러 개일 때 행 기준 인덱싱
    return series[idx, ...]
