# project/report/aggregate_pairwise.py
from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path
from typing import Dict
from .constants import PAIRWISE_MIN_COLUMNS, SUMMARY_SCORED_MIN_COLUMNS
from .scoring import apply_scoring
from .utils import make_panel_id

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"missing file: {path}")
    return pd.read_csv(path)

def load_or_build_summary_scored(root: Path) -> pd.DataFrame:
    """
    Prefer _summary_scored.csv if present; else derive minimal table from _summary.csv.
    """
    p_scored = root / "_summary_scored.csv"
    if p_scored.exists():
        df = pd.read_csv(p_scored)
        return df
    # fallback
    p_sum = root / "_summary.csv"
    df = pd.read_csv(p_sum)
    # minimal reshape
    need = ["panel_id","method","EW","ES95","RuinPct","WinRate","es_metric","tag"]
    for c in need:
        if c not in df.columns:
            df[c] = None
    df = df[need]
    df = apply_scoring(df)
    return df

def build_pairwise(df_scored: pd.DataFrame, group_cols=("panel_id","es_metric","tag")) -> pd.DataFrame:
    """
    For each (panel, es_metric, tag), find the best CompositeScore and compute deltas.
    delta = CompositeScore(method) - best
    is_win = (delta >= -1e-12)
    """
    g = df_scored.groupby(list(group_cols), dropna=False, as_index=False)
    rows = []
    for _, sub in g:
        if len(sub)==0: 
            continue
        best = sub["CompositeScore"].max()
        best_method = sub.loc[sub["CompositeScore"].idxmax(), "method"]
        for _, r in sub.iterrows():
            rows.append({
                "panel_id": r.get("panel_id"),
                "method": r.get("method"),
                "comparator": best_method,
                "delta": float(r.get("CompositeScore", 0.0)) - float(best),
                "is_win": float(r.get("CompositeScore", 0.0)) >= float(best) - 1e-12,
                "es_metric": r.get("es_metric"),
                "tag": r.get("tag"),
            })
    out = pd.DataFrame(rows)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="directory containing *_summary*.csv")
    ap.add_argument("--out", default="_pairwise_vs_best.csv")
    args = ap.parse_args()

    root = Path(args.root)
    df_scored = load_or_build_summary_scored(root)
    # Ensure required columns exist
    for c in SUMMARY_SCORED_MIN_COLUMNS:
        if c not in df_scored.columns:
            df_scored[c] = None
    # Pairwise
    df_pw = build_pairwise(df_scored)
    # Save
    out_path = root / args.out
    df_pw.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path}  (rows={len(df_pw)})")

if __name__ == "__main__":
    main()
