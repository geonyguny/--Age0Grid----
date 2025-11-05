# scripts/find_optima.py
from __future__ import annotations
import argparse, json, os, pathlib as p
import pandas as pd
import numpy as np
import re

def _infer_from_tag(df: pd.DataFrame) -> pd.DataFrame:
    """tag에서 us, h 파싱 → mix_us, hedge_sigma_k 채우기."""
    if "tag" not in df.columns:
        return df
    pat = re.compile(r".*?_2D_us(?P<u>[0-9.]+)_h(?P<h>[0-9.]+)")
    def _one(tag: str):
        m = pat.match(str(tag))
        if not m:
            return {}
        try:
            u = float(m.group("u")); h = float(m.group("h"))
        except Exception:
            return {}
        return {"mix_us": u, "hedge_sigma_k": h}
    extra = df["tag"].map(_one)
    extra_df = pd.DataFrame(list(extra))
    for c in ("mix_us","hedge_sigma_k"):
        if c not in df.columns and c in extra_df.columns:
            df[c] = extra_df[c]
    return df

def main():
    ap = argparse.ArgumentParser(description="Pick optimal (x,y) by CompositeScore")
    ap.add_argument("--src", required=True, help="snapshot csv")
    ap.add_argument("--tag_startswith", required=True, help="e.g., DEV_2D_")
    ap.add_argument("--x", default="mix_us")
    ap.add_argument("--y", default="hedge_sigma_k")
    ap.add_argument("--z", default="CompositeScore")
    ap.add_argument("--tiebreak", default="ES95", help="lower is better")
    ap.add_argument("--agg", choices=["median","mean"], default="median")
    ap.add_argument("--method", default="", help="optional filter")
    ap.add_argument("--es_mode", default="", help="optional filter")
    ap.add_argument("--out_csv", default=r".\outputs\paper\tables\optimal_points.csv")
    ap.add_argument("--out_json", default=r".\outputs\figs\optimal_points.json")
    args = ap.parse_args()

    df = pd.read_csv(args.src)
    df = df[df["tag"].astype(str).str.startswith(args.tag_startswith)]
    if args.method:
        df = df[df.get("method","").astype(str).str.lower()==args.method.lower()]
    if args.es_mode:
        df = df[df.get("es_mode","").astype(str).str.lower()==args.es_mode.lower()]
    if df.empty:
        raise SystemExit("[find_optima] no rows after filter")

    df = _infer_from_tag(df)
    for c in (args.x, args.y, args.z, args.tiebreak):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[args.x, args.y, args.z])

    # 동일 (x,y) 가 여러 seed/method로 존재 → 집계
    grp_cols = [args.x, args.y]
    aggfunc = dict(median="median", mean="mean")[args.agg]
    agg_df = df.groupby(grp_cols, as_index=False).agg({
        args.z: aggfunc,
        args.tiebreak: aggfunc
    })

    # 정렬 규칙: z desc, tiebreak asc, |x-0.6| asc (선호중심 0.6은 필요 시 변경)
    agg_df["_abs_x_center"] = (agg_df[args.x] - 0.6).abs()
    agg_df = agg_df.sort_values(
        by=[args.z, args.tiebreak, "_abs_x_center"],
        ascending=[False, True, True]
    ).reset_index(drop=True)

    best = agg_df.iloc[0].to_dict()

    # 저장물
    out_csv = p.Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    agg_df.rename(columns={args.z:"score", args.tiebreak:"es"}).to_csv(out_csv, index=False)

    out_json = p.Path(args.out_json); out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "points":[
            {"x": float(best[args.x]), "y": float(best[args.y]),
             "label": f"best: {best[args.x]:.2f},{best[args.y]:.2f}",
             "score": float(best[args.z]), "es": float(best[args.tiebreak])}
        ],
        "x": args.x, "y": args.y
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[find_optima] best ({args.x},{args.y}) = ({best[args.x]}, {best[args.y]}) "
          f"| score={best[args.z]:.4f}, {args.tiebreak}={best[args.tiebreak]:.4f}")
    print(f"[OK] {out_csv}")
    print(f"[OK] {out_json}")

if __name__ == "__main__":
    main()
