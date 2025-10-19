# project/data/schema_validator.py
from __future__ import annotations
import os, json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

REQUIRED_MIN = ["date", "ret_kr_eq", "cpi_kr", "rf_kr_nom"]

def _is_month_str(s: str) -> bool:
    return isinstance(s, str) and len(s) >= 7 and s[:4].isdigit() and s[4] == "-" and s[5:7].isdigit()

def _as_month(s) -> str:
    try:
        return pd.to_datetime(str(s)).strftime("%Y-%m")
    except Exception:
        s = str(s)[:7]
        if _is_month_str(s): return s
        raise

def _rate_like(arr: np.ndarray) -> bool:
    if arr.size == 0: return True
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0: return True
    return (np.nanmax(np.abs(a)) < 5.0) and (np.nanmedian(np.abs(a)) < 0.5)

def _fail(msg: str):
    raise ValueError(f"[schema] {msg}")

def validate_market_csv(path: str, *, enforce_required: bool = True) -> Dict:
    """CSV 스키마/값 검증. 문제 발견 시 ValueError."""
    if not os.path.exists(path):
        _fail(f"not found: {path}")

    df = pd.read_csv(path)
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols

    if enforce_required:
        for c in REQUIRED_MIN:
            if c not in cols:
                _fail(f"missing column '{c}' (required: {REQUIRED_MIN})")

    # date 정규화 및 정렬 검사
    try:
        mm = df["date"].map(_as_month)
    except Exception:
        _fail("column 'date' must be parseable to YYYY-MM")
    df["date"] = mm
    if (mm != mm.sort_values().values).any():
        _fail("date must be non-decreasing (sort by month)")

    # 타입/결측 검사(핵심 수치열만)
    num_cols = [c for c in ["ret_kr_eq","cpi_kr","rf_kr_nom",
                            "ret_us_eq_krw","ret_us_eq_usd","ret_gold_krw","ret_gold_usd","usdkrw","rf_kr_real"]
                if c in cols]
    for c in num_cols:
        v = pd.to_numeric(df[c], errors="coerce")
        null_rate = float(v.isna().mean())
        if null_rate > 0.05:
            _fail(f"too many NaNs in '{c}' (>{null_rate:.1%})")
        if c == "cpi_kr":
            # CPI는 지수/률 모두 허용 → 판별만 기록
            pass
        else:
            # 극단치 기본 가드
            x = v.to_numpy()
            x = x[np.isfinite(x)]
            if x.size and np.nanmax(np.abs(x)) > 100.0:
                _fail(f"abs({c}) has extreme values (>100)")

    # 메타: CPI가 지수형인지(rate-like가 아닌지) 휴리스틱
    cpi_like = None
    if "cpi_kr" in df.columns:
        cpi_like = "rate" if _rate_like(pd.to_numeric(df["cpi_kr"], errors="coerce").to_numpy()) else "index"

    return {
        "n_rows": int(len(df)),
        "have": {c: True for c in cols},
        "cpi_mode": cpi_like,  # "index" | "rate" | None
        "required_ok": all(c in cols for c in REQUIRED_MIN),
    }
