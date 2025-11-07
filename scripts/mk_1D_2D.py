# -*- coding: utf-8 -*-
# mk_1D_2D.py — Make 1D/2D tables & figures for paper from snapshot/summary.
import argparse, re, os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_df(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)

TAG_1D_PATTERNS = {
    # DEF1D_ann_{mort}_{sex}_{a}
    "ann":   re.compile(r"^(?P<prefix>.+)1D_ann_(?P<mort>[^_]+)_(?P<sex>[^_]+)_(?P<val>[-\d\.eE]+)$"),
    # DEF1D_wmax_{mort}_{sex}_{w}
    "wmax":  re.compile(r"^(?P<prefix>.+)1D_wmax_(?P<mort>[^_]+)_(?P<sex>[^_]+)_(?P<val>[-\d\.eE]+)$"),
    # DEF1D_fee_{mort}_{sex}_{f}
    "fee":   re.compile(r"^(?P<prefix>.+)1D_fee_(?P<mort>[^_]+)_(?P<sex>[^_]+)_(?P<val>[-\d\.eE]+)$"),
    # DEF1D_hedge_{mort}_{sex}_{h}
    "hfx":   re.compile(r"^(?P<prefix>.+)1D_hedge_(?P<mort>[^_]+)_(?P<sex>[^_]+)_(?P<val>[-\d\.eE]+)$"),
    # DEF1D_vpw_{mort}_{sex}_{c}
    "vpw":   re.compile(r"^(?P<prefix>.+)1D_vpw_(?P<mort>[^_]+)_(?P<sex>[^_]+)_(?P<val>[-\d\.eE]+)$"),
    # DEF1D_mix_{name}_{mort}_{sex}
    "mix":   re.compile(r"^(?P<prefix>.+)1D_mix_(?P<mix>[^_]+)_(?P<mort>[^_]+)_(?P<sex>[^_]+)$"),
}

TAG_2D_PATTERNS = {
    # DEF2D_ann_wmax_{mort}_{sex}_a{a}_w{w}
    "ann_wmax": re.compile(r"^(?P<prefix>.+)2D_ann_wmax_(?P<mort>[^_]+)_(?P<sex>[^_]+)_a(?P<a>[-\d\.eE]+)_w(?P<w>[-\d\.eE]+)$"),
    # DEF2D_wmax_hfx_{mort}_{sex}_w{w}_h{h}
    "wmax_hfx": re.compile(r"^(?P<prefix>.+)2D_wmax_hfx_(?P<mort>[^_]+)_(?P<sex>[^_]+)_w(?P<w>[-\d\.eE]+)_h(?P<h>[-\d\.eE]+)$"),
}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag_prefix", default="DEF")
    ap.add_argument("--metrics_csv", required=True)
    ap.add_argument("--summary_csv", default="")
    ap.add_argument("--outdir", default="./outputs/figs")
    ap.add_argument("--weights", default="0.6,0.4", help="EW,ES95 weights for fallback composite")
    return ap.parse_args()

def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan

def make_composite(group_df, w_ew=0.6, w_es=0.4):
    # min-max normalize within group for EW (higher better) and ES95 (higher better)
    df = group_df.copy()
    for col in ["EW","ES95"]:
        v = df[col].astype(float)
        vmin, vmax = np.nanmin(v), np.nanmax(v)
        if np.isfinite(vmin) and np.isfinite(vmax) and vmax>vmin:
            df[col+"_norm"] = (v - vmin) / (vmax - vmin)
        else:
            df[col+"_norm"] = np.nan
    df["Composite_fallback"] = w_ew*df["EW_norm"] + w_es*df["ES95_norm"]
    return df

def line_plot(x, y, title, xlabel, ylabel, path):
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=240)
    plt.close()

def heatmap_plot(X, xvals, yvals, title, xlabel, ylabel, path):
    plt.figure()
    im = plt.imshow(X, aspect="auto", origin="lower",
                    extent=[min(xvals), max(xvals), min(yvals), max(yvals)])
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.colorbar(im,label="Composite (benefit or fallback)")
    plt.tight_layout()
    plt.savefig(path, dpi=240)
    plt.close()

def extract_1d_rows(df):
    rows=[]
    for var,pat in TAG_1D_PATTERNS.items():
        for t in df["tag"].astype(str):
            m = pat.match(t)
            if m:
                d = {"var":var,"tag":t,"mort":m.group("mort"),"sex":m.group("sex")}
                if var=="mix":
                    d["val"] = m.group("mix")
                else:
                    d["val"] = safe_float(m.group("val"))
                rows.append(d)
    if not rows:
        return pd.DataFrame(columns=["var","tag","mort","sex","val"])
    out = pd.DataFrame(rows)
    return out

def extract_2d_rows(df):
    rows=[]
    for var,pat in TAG_2D_PATTERNS.items():
        for t in df["tag"].astype(str):
            m = pat.match(t)
            if m:
                d = {"pair":var,"tag":t,"mort":m.group("mort"),"sex":m.group("sex")}
                if var=="ann_wmax":
                    d["a"] = safe_float(m.group("a"))
                    d["w"] = safe_float(m.group("w"))
                elif var=="wmax_hfx":
                    d["w"] = safe_float(m.group("w"))
                    d["h"] = safe_float(m.group("h"))
                rows.append(d)
    if not rows:
        return pd.DataFrame(columns=["pair","tag","mort","sex","a","w","h"])
    return pd.DataFrame(rows)

def main():
    args = parse_args()
    ensure_dir(args.outdir)
    base = os.path.dirname(args.outdir) if args.outdir.endswith("figs") else args.outdir
    writer = pd.ExcelWriter(os.path.join(base,"Paper_1D_2D_Tables.xlsx"), engine="xlsxwriter")
    qtxt   = open(os.path.join(base,"Paper_Quick_Summary.txt"),"w",encoding="utf-8")

    dfM = load_df(args.metrics_csv)
    # 필요한 지표만 캐스팅
    for col in ["EW","ES95","Ruin"]:
        if col in dfM.columns:
            dfM[col] = pd.to_numeric(dfM[col], errors="coerce")

    # summary에 CompositeScore_benefit 있으면 merge
    dfS = None
    if args.summary_csv and os.path.exists(args.summary_csv):
        dfS = load_df(args.summary_csv)[["tag","CompositeScore_benefit"]]
    comp_col = "CompositeScore_benefit"

    # 1D 처리 --------------------------------------------------------
    meta1d = extract_1d_rows(dfM)
    if len(meta1d):
        df1 = meta1d.merge(dfM, on="tag", how="left")
        if dfS is not None:
            df1 = df1.merge(dfS, on="tag", how="left")
        # 그룹(변수 × mort × sex)별 표·그림
        for (v,mort,sex), g in df1.groupby(["var","mort","sex"]):
            gg = g.copy()
            if comp_col not in gg.columns or gg[comp_col].isna().all():
                w1,w2 = [float(x) for x in args.weights.split(",")]
                gg = make_composite(gg, w1, w2)
                score_col = "Composite_fallback"
            else:
                score_col = comp_col

            # 표 저장
            sheet = f"1D_{v}_{mort}_{sex}"
            gg_sort = gg.sort_values(by=("val" if v!="mix" else score_col))
            cols = ["tag","var","val","EW","ES95","Ruin"]
            if score_col in gg_sort.columns: cols += [score_col]
            gg_sort[cols].to_excel(writer, sheet_name=sheet, index=False)

            # 라인 그림(EW/ES95) — mix는 막대형 대체
            fig_prefix = os.path.join(args.outdir,f"1D_{v}_{mort}_{sex}")
            if v!="mix":
                x = gg_sort["val"].astype(float).values
                if "EW" in gg_sort:
                    line_plot(x, gg_sort["EW"].values, f"1D {v} — EW ({mort}/{sex})", v, "EW", f"{fig_prefix}_EW.png")
                if "ES95" in gg_sort:
                    line_plot(x, gg_sort["ES95"].values, f"1D {v} — ES95 ({mort}/{sex})", v, "ES95", f"{fig_prefix}_ES95.png")
            else:
                # mix: 표만, 혹 원하면 막대형 추가 가능
                pass

            # 베스트 한 줄 요약
            best = gg_sort.sort_values(by=score_col, ascending=False).head(1)
            if len(best):
                b = best.iloc[0]
                qtxt.write(f"[1D] {v} / {mort} / {sex} → best tag={b['tag']} val={b.get('val','-')} score={b.get(score_col,np.nan)} EW={b.get('EW',np.nan)} ES95={b.get('ES95',np.nan)} Ruin={b.get('Ruin',np.nan)}\n")

    # 2D 처리 --------------------------------------------------------
    meta2d = extract_2d_rows(dfM)
    if len(meta2d):
        df2 = meta2d.merge(dfM, on="tag", how="left")
        if dfS is not None:
            df2 = df2.merge(dfS, on="tag", how="left")
        for (pair,mort,sex), g in df2.groupby(["pair","mort","sex"]):
            gg = g.copy()
            if comp_col not in gg.columns or gg[comp_col].isna().all():
                w1,w2 = [float(x) for x in args.weights.split(",")]
                gg = make_composite(gg, w1, w2)
                score_col = "Composite_fallback"
            else:
                score_col = comp_col

            sheet = f"2D_{pair}_{mort}_{sex}"
            gg_out = gg.copy()
            gg_out.to_excel(writer, sheet_name=sheet, index=False)

            # 히트맵 (x축·y축 정의)
            if pair=="ann_wmax":
                xvals = np.sort(gg["a"].dropna().unique())
                yvals = np.sort(gg["w"].dropna().unique())
                Z = np.full((len(yvals),len(xvals)), np.nan)
                for i,y in enumerate(yvals):
                    for j,x in enumerate(xvals):
                        sub = gg[(gg["a"]==x)&(gg["w"]==y)]
                        if len(sub):
                          Z[i,j] = float(sub.iloc[0][score_col]) if pd.notnull(sub.iloc[0][score_col]) else np.nan
                heatmap_plot(Z, xvals, yvals, f"2D ann vs w_max ({mort}/{sex})", "ann_alpha", "w_max", os.path.join(args.outdir,f"2D_ann_wmax_{mort}_{sex}.png"))
            elif pair=="wmax_hfx":
                xvals = np.sort(gg["w"].dropna().unique())
                yvals = np.sort(gg["h"].dropna().unique())
                Z = np.full((len(yvals),len(xvals)), np.nan)
                for i,y in enumerate(yvals):
                    for j,x in enumerate(xvals):
                        sub = gg[(gg["w"]==x)&(gg["h"]==y)]
                        if len(sub):
                          Z[i,j] = float(sub.iloc[0][score_col]) if pd.notnull(sub.iloc[0][score_col]) else np.nan
                heatmap_plot(Z, xvals, yvals, f"2D w_max vs h_fx ({mort}/{sex})", "w_max", "h_fx", os.path.join(args.outdir,f"2D_wmax_hfx_{mort}_{sex}.png"))

            # 베스트 요약
            best = gg.sort_values(by=score_col, ascending=False).head(1)
            if len(best):
                b = best.iloc[0]
                if pair=="ann_wmax":
                    qtxt.write(f"[2D] ann×w_max / {mort} / {sex} → best a={b['a']} w={b['w']} score={b.get(score_col,np.nan)} tag={b['tag']}\n")
                elif pair=="wmax_hfx":
                    qtxt.write(f"[2D] w_max×h_fx / {mort} / {sex} → best w={b['w']} h={b['h']} score={b.get(score_col,np.nan)} tag={b['tag']}\n")

    writer.close()
    qtxt.close()
    print("[OK] Saved Excel & figures to", base)

if __name__=="__main__":
    main()
