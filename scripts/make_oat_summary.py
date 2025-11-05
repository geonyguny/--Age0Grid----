# scripts/make_oat_summary.py
import argparse, os
import numpy as np
import pandas as pd

def iqr(s): 
    return np.nanpercentile(s, 75) - np.nanpercentile(s, 25)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--tag_startswith", required=True)
    p.add_argument("--method", required=True, choices=["hjb","rl","rule"])
    p.add_argument("--es_mode", default="wealth", choices=["wealth","cons"])
    p.add_argument("--mode", required=True, choices=["oat","2d"])
    p.add_argument("--var", default=None, help="OAT: variable column (e.g., bias_loss_aversion)")
    p.add_argument("--x", default=None, help="2D: x column")
    p.add_argument("--y", default=None, help="2D: y column")
    p.add_argument("--metrics", default="EW,ES95,CompositeScore")
    p.add_argument("--agg", default="median", choices=["median"])
    p.add_argument("--out", required=True, help=r"outputs\paper\tables\xxx.xlsx")
    p.add_argument("--csv_out", default=None)
    return p.parse_args()

def ensure_exists(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)

def pick(df, tag_prefix, method, es_mode):
    d = df.copy()
    d = d[d["tag"].astype(str).str.startswith(tag_prefix)]
    d = d[d["method"].astype(str)==method]
    d = d[d["es_mode"].astype(str)==es_mode]
    return d

def ensure_alias(df, name):
    if name is None: return df, None
    # minimal alias for loss_aversion
    if name not in df.columns:
        for a in ["bias_loss_aversion","la_k","la"]:
            if a in df.columns:
                df = df.copy()
                df[name] = df[a]
                break
    return df, name

def main():
    args = parse_args()
    ensure_exists(args.out)
    df = pd.read_csv(args.src)
    df = pick(df, args.tag_startswith, args.method, args.es_mode)
    if df.empty:
        raise SystemExit(f"[ERR] no rows after filter. tag={args.tag_startswith}, method={args.method}, es_mode={args.es_mode}")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    keep = ["tag","seed","method","es_mode"] + metrics + [c for c in df.columns if c in ["mix_us","hedge_sigma_k","bias_loss_aversion","loss_aversion","la_k"]]
    df = df[[c for c in keep if c in df.columns]].copy()

    out_tables = {}

    if args.mode=="oat":
        df, var = ensure_alias(df, args.var)
        if var is None or var not in df.columns:
            raise SystemExit(f"[ERR] var column not found: {args.var}")
        grp = df.groupby(var)
        for m in metrics:
            if m not in df.columns: continue
            out_tables[f"{m}_median"] = grp[m].median().rename(m)
            out_tables[f"{m}_std"]    = grp[m].std(ddof=1).rename(m)
            out_tables[f"{m}_iqr"]    = grp[m].agg(iqr).rename(m)

    else:  # 2d
        if args.x not in df.columns or args.y not in df.columns:
            raise SystemExit(f"[ERR] 2D requires columns --x {args.x}, --y {args.y}")
        grp = df.groupby([args.x, args.y])
        for m in metrics:
            if m not in df.columns: continue
            out_tables[f"{m}_median"] = grp[m].median().unstack(args.y)
            out_tables[f"{m}_std"]    = grp[m].std(ddof=1).unstack(args.y)
            out_tables[f"{m}_iqr"]    = grp[m].agg(iqr).unstack(args.y)

    # write excel
    with pd.ExcelWriter(args.out, engine="xlsxwriter") as xw:
        for name, tbl in out_tables.items():
            tbl.to_excel(xw, sheet_name=name[:31])

    if args.csv_out:
        # simple dump of median rows for quick glance
        med_keys = [k for k in out_tables if k.endswith("_median")]
        if med_keys:
            if args.mode=="oat":
                out_tables[med_keys[0]].to_csv(args.csv_out, index=True, encoding="utf-8-sig")
            else:
                # pick first metric median to csv (pivot)
                out_tables[med_keys[0]].to_csv(args.csv_out, encoding="utf-8-sig")

    print(f"[OK] wrote {args.out}" + (f" and {args.csv_out}" if args.csv_out else ""))

if __name__=="__main__":
    main()
