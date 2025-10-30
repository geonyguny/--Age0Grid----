# project/report/constants.py
"""
Centralized constants and schema definitions for the reporting pipeline.
Edit weights below to tune CompositeScore.
"""

from __future__ import annotations

# === Scoring Weights (feel free to tune) ===
SCORING_VERSION = "v1.0.0"
WEIGHTS = {
    "EW": +0.50,          # higher is better
    "ES95": -0.35,        # lower tail risk is better (negative weight)
    "RuinPct": -0.10,     # lower is better
    "WinRate": +0.05,     # higher is better
}

# === Required Columns for pairwise schema ===
PAIRWISE_MIN_COLUMNS = [
    "panel_id", "method", "comparator", "delta", "is_win",
    "es_metric", "tag"
]

# === Required Columns for summary_scored schema ===
SUMMARY_SCORED_MIN_COLUMNS = [
    "panel_id", "method", "EW", "ES95", "RuinPct", "WinRate",
    "CompositeScore", "es_metric", "tag"
]
