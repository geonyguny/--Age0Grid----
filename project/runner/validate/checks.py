# project/runner/validate/checks.py
from __future__ import annotations
from typing import Dict, Any

DEFAULT_THRESHOLDS = {
    "Ruin_max": 0.05,       # <= 5%
    "ES95_ratio_vs_HJB": 1.20,  # rule/RL ES95 ≤ 1.2 × HJB
}

def basic_checks(metrics_row: Dict[str, Any], *, hjb_ref: Dict[str, float] | None = None) -> Dict[str, Any]:
    """간단 PASS/FAIL. 반환: {'pass': bool, 'notes': [...]}"""
    notes = []
    ok = True

    ruin = float(metrics_row.get("Ruin", 0.0) or 0.0)
    if ruin > DEFAULT_THRESHOLDS["Ruin_max"]:
        ok = False
        notes.append(f"Ruin {ruin:.3f} > {DEFAULT_THRESHOLDS['Ruin_max']}")

    if hjb_ref and "ES95" in metrics_row:
        es = float(metrics_row["ES95"] or 0.0)
        hjb_es = float(hjb_ref.get("ES95", 0.0) or 0.0)
        if hjb_es > 0 and es > DEFAULT_THRESHOLDS["ES95_ratio_vs_HJB"] * hjb_es:
            ok = False
            notes.append(f"ES95 {es:.4f} worse than 1.2×HJB ({hjb_es:.4f})")

    return {"pass": ok, "notes": notes}

def pick_hjb_ref(summary_df) -> Dict[str, float] | None:
    try:
        d = summary_df.loc[summary_df["method"].str.lower()=="hjb"].head(1)
        if len(d):
            return {"EW": float(d["EW_avg"].iloc[0]), "ES95": float(d["ES95_avg"].iloc[0])}
    except Exception:
        pass
    return None
