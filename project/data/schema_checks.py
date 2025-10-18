# project/data/schema_checks.py
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import pandas as pd


# ─────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────
def assert_market_csv_valid(
    path: str,
    required_cols: Optional[List[str]] = None,
    date_col: str = "date",
    allow_extra_cols: bool = True,
    sample_n: int = 3,
) -> None:
    """
    Validate market CSV schema (v1).
    - Required columns exist
    - 'date' column parseable to datetime and monotonic ascending (after sort)
    - No all-NaN columns in required set
    - Numeric columns have numeric dtype (after coercion)
    - No duplicate dates
    - Basic sanity: at least 12 rows

    Args
    ----
    path: CSV file path
    required_cols: list of required columns excluding the date column. If None, defaults to:
        ['risky_nom', 'tbill_nom', 'cpi']
    date_col: name of the date column
    allow_extra_cols: if False, fail when unknown columns exist (besides required ones)
    sample_n: number of row samples to show in error messages

    Raises
    ------
    ValueError on validation failure with a human-readable message
    """
    _raise_if(not os.path.isfile(path), f"[schema] file not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"[schema] failed to read CSV: {path}\n- {e}")

    # Defaults
    if required_cols is None:
        required_cols = ["risky_nom", "tbill_nom", "cpi"]

    # 1) column existence
    cols = list(df.columns)
    _raise_if(date_col not in cols, _fmt(
        "[schema] missing date column '{date_col}'. got columns={cols}",
        date_col=date_col, cols=cols,
    ))
    missing = [c for c in required_cols if c not in cols]
    _raise_if(missing, _fmt(
        "[schema] missing required columns: {missing}. got columns={cols}",
        missing=missing, cols=cols,
    ))

    # 2) extra column policy
    if not allow_extra_cols:
        allowed = set([date_col] + required_cols)
        extra = [c for c in cols if c not in allowed]
        _raise_if(extra, _fmt(
            "[schema] unknown columns not allowed: {extra}. allowed={allowed}",
            extra=extra, allowed=sorted(list(allowed))
        ))

    # 3) parse date and basic checks
    df = df.copy()
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors="raise")
    except Exception as e:
        _fail_with_sample(df, f"[schema] failed to parse '{date_col}' to datetime: {e}", sample_n)

    # drop duplicates check
    dup_mask = df.duplicated(subset=[date_col], keep=False)
    _raise_if(bool(dup_mask.any()), _fmt(
        "[schema] duplicate '{date_col}' found for rows: {rows}",
        date_col=date_col,
        rows=_row_indices(dup_mask),
    ))

    # Sort by date (won't mutate caller)
    df = df.sort_values(date_col, kind="mergesort").reset_index(drop=True)

    # 4) basic length
    _raise_if(len(df) < 12, _fmt(
        "[schema] too few rows: {n}. need >= 12", n=len(df)
    ))

    # 5) required numeric columns: coerce and check all-NaN / dtype
    numeric_cols = required_cols[:]  # copy
    coerced = {}
    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        coerced[c] = s
        _raise_if(s.isna().all(), _fmt(
            "[schema] column '{c}' is all-NaN after numeric coercion", c=c
        ))
        # small sample of NaN rows if any
        nan_idx = s.isna()
        if nan_idx.any():
            _warn_with_sample(df.loc[nan_idx, [date_col, c]], _fmt(
                "[schema] column '{c}' has NaNs after numeric coercion (showing up to {k})",
                c=c, k=sample_n
            ), sample_n)

    # 6) no all-NaN required columns (already checked) and at least one non-NaN row across all
    any_ok = False
    for c in numeric_cols:
        if (~coerced[c].isna()).any():
            any_ok = True
            break
    _raise_if(not any_ok, "[schema] all required numeric columns are NaN. data unusable.")

    # Passed
    return None


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────
def _fmt(s: str, **kw) -> str:
    try:
        return s.format(**kw)
    except Exception:
        return s

def _raise_if(cond: bool, msg: str) -> None:
    if cond:
        raise ValueError(msg)

def _row_indices(mask: pd.Series, max_n: int = 10) -> List[int]:
    idx = list(mask[mask].index[:max_n])
    if len(mask[mask]) > max_n:
        idx.append(-1)  # indicate more
    return idx

def _fail_with_sample(df_like: pd.DataFrame, msg: str, sample_n: int) -> None:
    sample = df_like.head(sample_n)
    raise ValueError(f"{msg}\n[sample top {sample_n} rows]\n{sample.to_string(index=False)}")

def _warn_with_sample(df_like: pd.DataFrame, msg: str, sample_n: int) -> None:
    # Non-fatal: print to stdout (runner logger will capture)
    sample = df_like.head(sample_n)
    print(f"{msg}\n[sample top {sample_n} rows]\n{sample.to_string(index=False)}")
