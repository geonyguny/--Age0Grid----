# project/report/scoring.py
from __future__ import annotations

import math
import pandas as pd
from typing import Dict
from .constants import WEIGHTS, SCORING_VERSION

def _safe(v):
    try:
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            return None
        return float(v)
    except Exception:
        return None

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalize EW (max), ES95 (min), RuinPct (min), WinRate (max)."""
    out = df.copy()
    # Prepare columns
    for c in ["EW","ES95","RuinPct","WinRate"]:
        if c not in out.columns:
            out[c] = None
    # Coerce
    for c in ["EW","ES95","RuinPct","WinRate"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    # Normalize
    def _norm(col: pd.Series, invert: bool=False) -> pd.Series:
        x = col.astype(float)
        lo, hi = x.min(skipna=True), x.max(skipna=True)
        if pd.isna(lo) or pd.isna(hi) or lo==hi:
            return pd.Series([0.5 if not invert else 0.5]*len(x), index=x.index)  # neutral if degenerate
        z = (x - lo) / (hi - lo)
        return 1.0 - z if invert else z
    out["EW_n"] = _norm(out["EW"], invert=False)
    out["ES95_n"] = _norm(out["ES95"], invert=True)      # lower is better
    out["RuinPct_n"] = _norm(out["RuinPct"], invert=True)
    # WinRate may be precomputed 0..1
    if "WinRate" in out.columns:
        wr = pd.to_numeric(out["WinRate"], errors="coerce").clip(0,1)
        out["WinRate_n"] = wr.fillna(0.5)
    else:
        out["WinRate_n"] = 0.5
    return out

def composite_score(df_norm: pd.DataFrame) -> pd.Series:
    w = WEIGHTS
    # Missing columns default neutral 0.5 in normalize_columns
    s = (
        w["EW"] * df_norm["EW_n"] +
        w["ES95"] * df_norm["ES95_n"] +
        w["RuinPct"] * df_norm["RuinPct_n"] +
        w["WinRate"] * df_norm["WinRate_n"]
    )
    return s

def apply_scoring(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_columns(df)
    out["CompositeScore"] = composite_score(out)
    out["scoring_version"] = SCORING_VERSION
    return out
