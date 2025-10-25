# project/data/schema_checks.py
from __future__ import annotations

import os
from typing import List, Optional, Dict, Iterable

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
    Validate market CSV schema (v1) with alias support.
    - Required columns exist (any-one-of alias set is accepted)
    - 'date' column parseable to datetime and monotonic (post-sort)
    - No all-NaN columns in required set (after numeric coercion)
    - Numeric columns are coercible to numbers (NaN rows are warned)
    - No duplicate dates
    - Basic sanity: at least 12 rows

    NOTE:
    This function is validation-only and returns None. It does not mutate the CSV.
    Callers that need canonical column names should normalize downstream if necessary.
    """
    _raise_if(not os.path.isfile(path), f"[schema] file not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"[schema] failed to read CSV: {path}\n- {e}")

    # Defaults (kept for backward-compat)
    if required_cols is None:
        required_cols = ["risky_nom", "tbill_nom", "cpi"]

    # ---- alias sets (any-one-of present is accepted) ----
    # 키는 호출자가 요구하는 '논리 컬럼 이름'(required_cols에 올 수 있는 값)
    # 값은 해당 컬럼을 충족시킬 수 있는 '실제 CSV 컬럼명의 후보 집합'
    alias_any: Dict[str, Iterable[str]] = {
        # legacy <-> new
        "cpi": {"cpi", "cpi_kr"},
        "cpi_kr": {"cpi_kr", "cpi"},

        "rf_nom": {"rf_nom", "rf_kr_nom"},
        "rf_kr_nom": {"rf_kr_nom", "rf_nom"},

        # 종종 쓰는 FX 표기
        "ret_fx": {"ret_fx", "ret_fx_usdkrw"},
        "ret_fx_usdkrw": {"ret_fx_usdkrw", "ret_fx"},

        # 프로젝트 내에서 관측되는 자산 컬럼들(그 자체로 필수일 수 있음)
        "ret_kr_eq": {"ret_kr_eq"},
        "ret_us_eq_krw": {"ret_us_eq_krw"},
        "ret_gold_krw": {"ret_gold_krw"},
        "rf_real": {"rf_real"},
        "rf_kr_real": {"rf_kr_real", "rf_real"},
        "tbill_nom": {"tbill_nom", "rf_nom", "rf_kr_nom"},  # 호환 목적
        "risky_nom": {"risky_nom", "ret_kr_eq"},            # 호환 목적
    }

    cols = list(df.columns)

    # 1) date column
    _raise_if(date_col not in cols, _fmt(
        "[schema] missing date column '{date_col}'. got columns={cols}",
        date_col=date_col, cols=cols,
    ))

    # 2) required existence (with alias)
    missing_logical = []
    present_map: Dict[str, str] = {}  # logical -> actual column used
    for logical in required_cols:
        candidates = alias_any.get(logical, {logical})
        actual = _choose_first_present(candidates, cols)
        if actual is None:
            missing_logical.append(logical)
        else:
            present_map[logical] = actual

    _raise_if(bool(missing_logical), _fmt(
        "[schema] missing required columns (any-one-of accepted): {missing}. got columns={cols}",
        missing=missing_logical, cols=cols,
    ))

    # 3) extra column policy
    if not allow_extra_cols:
        allowed = set([date_col] + list({present_map[k] for k in present_map}))
        extra = [c for c in cols if c not in allowed]
        _raise_if(bool(extra), _fmt(
            "[schema] unknown columns not allowed: {extra}. allowed={allowed}",
            extra=extra, allowed=sorted(list(allowed))
        ))

    # 4) parse date
    df = df.copy()
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors="raise")
    except Exception as e:
        _fail_with_sample(df, f"[schema] failed to parse '{date_col}' to datetime: {e}", sample_n)

    # 5) duplicates by date
    dup_mask = df.duplicated(subset=[date_col], keep=False)
    _raise_if(bool(dup_mask.any()), _fmt(
        "[schema] duplicate '{date_col}' found for rows: {rows}",
        date_col=date_col,
        rows=_row_indices(dup_mask),
    ))

    # stable sort by date
    df = df.sort_values(date_col, kind="mergesort").reset_index(drop=True)

    # 6) basic length
    _raise_if(len(df) < 12, _fmt(
        "[schema] too few rows: {n}. need >= 12", n=len(df)
    ))

    # 7) numeric coercion checks on the *actual* columns we will use
    any_ok = False
    for logical, actual in present_map.items():
        s = pd.to_numeric(df[actual], errors="coerce")
        _raise_if(s.isna().all(), _fmt(
            "[schema] column '{actual}' (for logical '{logical}') is all-NaN after numeric coercion",
            actual=actual, logical=logical
        ))

        # warn a small sample of NaN rows if any
        nan_idx = s.isna()
        if nan_idx.any():
            _warn_with_sample(
                df.loc[nan_idx, [date_col, actual]],
                _fmt(
                    "[schema] column '{actual}' (for logical '{logical}') has NaNs after numeric coercion (showing up to {k})",
                    actual=actual, logical=logical, k=sample_n
                ),
                sample_n
            )

        if (~s.isna()).any():
            any_ok = True

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

def _choose_first_present(candidates: Iterable[str], cols: List[str]) -> Optional[str]:
    colset = set(cols)
    for c in candidates:
        if c in colset:
            return c
    return None
