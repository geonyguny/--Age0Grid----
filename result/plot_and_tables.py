# -*- coding: utf-8 -*-
"""
plot_and_tables.py
- 1D: ann / wrisk / fee / age / hedge / vpw + (신규) mix 카테고리 막대
- 2D: ann×wrisk, wrisk×hedge, wrisk×c
- Optimal table/bars, Bias 비교
- 입력 경로 유연화(argparse) 및 파일 존재 시 자동 선택
"""

import os, re, textwrap, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ─────────────────────────────────────────────────────────
# CLI & defaults
# ─────────────────────────────────────────────────────────
def find_first_existing(cands):
    for p in cands:
        if p and os.path.exists(p):
            return p
    return None

def build_args():
    ap = argparse.ArgumentParser(description="Generate 1D/2D plots & tables and manifest")
    ap.add_argument("--snapshot", default=None, help="metrics snapshot CSV")
    ap.add_argument("--scored",   default=None, help="scored(clean) CSV")
    ap.add_argument("--bhcmp",    default=None, help="OPT_BH_compare CSV")
    ap.add_argument("--optbest",  default=None, help="DEV_OPT_best_by_sex_mort CSV")
    ap.add_argument("--out_figs", default=os.path.join(".", "outputs_figs"))
    ap.add_argument("--out_tables", default=os.path.join(".", "outputs_tables"))
    return ap.parse_args()

# ─────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────
def short_title(s: str, width: int = 78) -> str:
    s = s.replace("—", "-")
    if len(s) <= width: return s
    return "\n".join(textwrap.wrap(s, width=width)[:2])

def setup_axis_format(ax):
    sf = ScalarFormatter(useMathText=True)
    sf.set_powerlimits((-2, 3))
    ax.yaxis.set_major_formatter(sf)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2,3))

def savefig(fig, out_dir, fname):
    os.makedirs(out_dir, exist_ok=True)
    fpath = os.path.join(out_dir, fname)
    fig.tight_layout()
    fig.savefig(fpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return fpath

def save_table(df, out_dir, fname):
    os.makedirs(out_dir, exist_ok=True)
    fpath = os.path.join(out_dir, fname)
    df.to_csv(fpath, index=False)
    return fpath

def read_csv_safe(path):
    return pd.read_csv(path) if (path and os.path.exists(path)) else pd.DataFrame()

def coerce_metrics(df):
    if df.empty: return df
    cols = [c for c in ["EW","ES95","Ruin","C_ES95_avg","p50_c_last","CompositeScore"] if c in df.columns]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

SEX_REGEX = re.compile(r"(?:^|[_\-])([MF])(?:[_\-]|$)", re.IGNORECASE)

def ensure_sex_column(df, tag_col="tag"):
    """sex 컬럼이 없으면 tag/mort_table 등에서 추론하여 추가"""
    if df.empty:
        return df
    cols_lower = {c.lower(): c for c in df.columns}
    if "sex" in cols_lower:
        return df.rename(columns={cols_lower["sex"]: "sex"})

    if "mort_table" in cols_lower and "sex" not in df.columns:
        mt = df[cols_lower["mort_table"]].astype(str).str.upper()
        guess = np.where(mt.str.contains("F"), "F",
                 np.where(mt.str.contains("M"), "M", np.nan))
        if np.any(pd.notna(guess)):
            df["sex"] = guess
            return df

    if (tag_col is not None) and (tag_col in df.columns):
        tags = df[tag_col].astype(str)
        sex_guess = tags.str.extract(SEX_REGEX, expand=False).str.upper()
        if sex_guess.notna().any():
            df["sex"] = sex_guess
            return df

    df["sex"] = "ALL"
    return df

# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def main():
    args = build_args()

    # 입력 경로 후보 구성 (대소문자/경로 차이 대응)
    snap_path = args.snapshot or find_first_existing([
        "DEV_metrics_snapshot.csv",
        "dev_metrics_snapshot.csv",
        os.path.join("outputs", "DEV_metrics_snapshot.csv"),
        os.path.join(".", "outputs", "DEV_metrics_snapshot.csv"),
    ])
    scored_path = args.scored or find_first_existing([
        "DEV_scored_clean.csv",
        os.path.join("outputs", "DEV_scored_clean.csv"),
    ])
    bhcmp_path = args.bhcmp or find_first_existing([
        "OPT_BH_compare.csv",
        os.path.join("outputs", "OPT_BH_compare.csv"),
    ])
    optbest_path = args.optbest or find_first_existing([
        "DEV_OPT_best_by_sex_mort.csv",
        os.path.join("outputs", "DEV_OPT_best_by_sex_mort.csv"),
    ])

    OUT_FIGS = args.out_figs
    OUT_TABLES = args.out_tables

    snap   = coerce_metrics(read_csv_safe(snap_path))
    scored = coerce_metrics(read_csv_safe(scored_path))
    bhcmp  = coerce_metrics(read_csv_safe(bhcmp_path))
    optbest= coerce_metrics(read_csv_safe(optbest_path))

    manifest = []

    # ───────── (1) 1D: ann/wrisk/fee/age/hedge/vpw (x축 실수) ─────────
    if not snap.empty:
        snap.columns = [c.strip() for c in snap.columns]
        pat_1d = re.compile(r"^DEV1D_([a-zA-Z]+)_BASE_([MF])_([0-9.]+)$")
        rows = []
        for _, r in snap.iterrows():
            tag = str(r.get("tag",""))
            m = pat_1d.match(tag)
            if not m:
                continue
            var, sex, val = m.groups()
            try:
                xval = float(val)
            except:
                continue
            rows.append({
                "var": var.lower(), "sex": sex, "x": xval,
                "EW":  pd.to_numeric(r.get("EW", np.nan), errors="coerce"),
                "ES95":pd.to_numeric(r.get("ES95", np.nan), errors="coerce"),
                "Ruin":pd.to_numeric(r.get("Ruin", np.nan), errors="coerce"),
            })
        df1d = pd.DataFrame(rows)

        keep_vars = ["ann","wrisk","fee","age","hedge","vpw"]
        df1d = df1d[df1d["var"].isin(keep_vars)].sort_values(["var","sex","x"])

        metrics = [("EW","EW"),("ES95","ES95 (wealth)"),("Ruin","Ruin")]
        xlabels = {
            "ann":"Annuity share (alpha)",
            "wrisk":"Risk asset weight (w)",
            "fee":"Fee (annual)",
            "age":"Start age",
            "hedge":"FX hedge ratio (h)",
            "vpw":"VPW c"
        }

        for var in keep_vars:
            for sex in ["M","F"]:
                sub = df1d[(df1d["var"]==var) & (df1d["sex"]==sex)].copy()
                if sub.empty: continue
                tname = f"table_1D_{var}_{sex}.csv"
                save_table(sub.rename(columns={"x":var}), OUT_TABLES, tname)
                for mcol, mlabel in metrics:
                    fig, ax = plt.subplots(figsize=(8,4.2))
                    ax.plot(sub["x"], sub[mcol], marker="o")
                    ax.set_xlabel(xlabels.get(var,var)); ax.set_ylabel(mlabel)
                    setup_axis_format(ax); ax.grid(True, alpha=0.3)
                    title = short_title(f"[1D] {xlabels.get(var,var)} — "
                                        f"{'Male' if sex=='M' else 'Female'} / {mlabel}")
                    ax.set_title(title)
                    p = savefig(fig, OUT_FIGS, f"plot_1D_{var}_{sex}_{mcol}.png")
                    manifest.append({"title":title,"plot":p,"table":os.path.join(OUT_TABLES,tname)})

    # ───────── (1D-추가) mix: 카테고리 막대 (x축 문자열) ─────────
    # 예시 태그: DEV1D_mix_KR_BASE_M
    if not snap.empty:
        pat_mix = re.compile(r"^DEV1D_mix_(KR|US|GLD|EQUAL3)_BASE_([MF])$")
        rows_mix = []
        for _, r in snap.iterrows():
            tag = str(r.get("tag",""))
            m = pat_mix.match(tag)
            if not m: continue
            mix, sex = m.groups()
            rows_mix.append({
                "mix": mix, "sex": sex,
                "EW":  pd.to_numeric(r.get("EW", np.nan), errors="coerce"),
                "ES95":pd.to_numeric(r.get("ES95", np.nan), errors="coerce"),
                "Ruin":pd.to_numeric(r.get("Ruin", np.nan), errors="coerce"),
            })
        df_mix = pd.DataFrame(rows_mix)

        if not df_mix.empty:
            for sex in ["M","F"]:
                sub = df_mix[df_mix["sex"]==sex].copy()
                if sub.empty: continue
                tname = f"table_1D_mix_{sex}.csv"
                save_table(sub, OUT_TABLES, tname)
                for mcol, mlabel in [("EW","EW"),("ES95","ES95 (wealth)"),("Ruin","Ruin")]:
                    fig, ax = plt.subplots(figsize=(6.8,4.0))
                    sub2 = sub.sort_values("mix")
                    ax.bar(sub2["mix"], pd.to_numeric(sub2[mcol], errors="coerce"))
                    ax.set_xlabel("Asset mix preset"); ax.set_ylabel(mlabel)
                    setup_axis_format(ax); ax.grid(True, alpha=0.2)
                    title = short_title(f"[1D] Asset mix — {'Male' if sex=='M' else 'Female'} / {mlabel}")
                    ax.set_title(title)
                    p = savefig(fig, OUT_FIGS, f"plot_1D_mix_{sex}_{mcol}.png")
                    manifest.append({"title":title,"plot":p,"table":os.path.join(OUT_TABLES,tname)})

    # ───────── (2) 2D heatmaps ─────────
    if not snap.empty:
        pat_2d_aw = re.compile(r"^DEV2D_ann_wrisk_BASE_([MF])_a([0-9.]+)_w([0-9.]+)$")
        pat_2d_wh = re.compile(r"^DEV2D_wrisk_hedge_BASE_([MF])_w([0-9.]+)_h([0-9.]+)$")
        pat_2d_wc = re.compile(r"^DEV2D_wrisk_c_BASE_([MF])_w([0-9.]+)_c([0-9.]+)$")

        def collect_2d(pat, a, b):
            rec = []
            for _, r in snap.iterrows():
                tag = str(r.get("tag",""))
                m = pat.match(tag)
                if not m: continue
                sex = m.group(1)
                x = float(m.group(2)); y = float(m.group(3))
                rec.append({
                    "sex": sex, a: x, b: y,
                    "EW":  pd.to_numeric(r.get("EW", np.nan), errors="coerce"),
                    "ES95":pd.to_numeric(r.get("ES95", np.nan), errors="coerce"),
                    "Ruin":pd.to_numeric(r.get("Ruin", np.nan), errors="coerce"),
                })
            df = pd.DataFrame(rec)
            for c in [a,b]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df

        df_aw = collect_2d(pat_2d_aw, "alpha", "w")
        df_wh = collect_2d(pat_2d_wh, "w", "h")
        df_wc = collect_2d(pat_2d_wc, "w", "c")

        def pivot_plot(df, a, b, metric, mlabel, sex):
            sub = df[df["sex"]==sex].copy()
            if sub.empty: return
            sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
            pv = sub.pivot_table(index=a, columns=b, values=metric, aggfunc="mean")
            pv = pv.sort_index().sort_index(axis=1)

            tname = f"table_2D_{a}_{b}_{sex}_{metric}.csv"
            save_table(pv.reset_index(), OUT_TABLES, tname)

            fig, ax = plt.subplots(figsize=(6.4,5.2))
            im = ax.imshow(pv.values, origin="lower", aspect="auto")
            ax.set_xticks(np.arange(len(pv.columns))); ax.set_xticklabels([str(x) for x in pv.columns])
            ax.set_yticks(np.arange(len(pv.index)));  ax.set_yticklabels([str(x) for x in pv.index])
            ax.set_xlabel(b); ax.set_ylabel(a)
            title = short_title(f"[2D] {a} × {b} — "
                                f"{'Male' if sex=='M' else 'Female'} / {mlabel}")
            ax.set_title(title)
            fig.colorbar(im, ax=ax, label=mlabel)
            p = savefig(fig, OUT_FIGS, f"plot_2D_{a}_{b}_{sex}_{metric}.png")
            manifest.append({"title":title,"plot":p,"table":os.path.join(OUT_TABLES,tname)})

        for metric, mlabel in [("EW","EW"),("ES95","ES95 (wealth)"),("Ruin","Ruin")]:
            for sex in ["M","F"]:
                if not df_aw.empty: pivot_plot(df_aw, "alpha","w", metric, mlabel, sex)
                if not df_wh.empty: pivot_plot(df_wh, "w","h",    metric, mlabel, sex)
                if not df_wc.empty: pivot_plot(df_wc, "w","c",    metric, mlabel, sex)

    # ───────── (3) Optimal design (table + barh) ─────────
    opt_table = pd.DataFrame()
    if not optbest.empty:
        optbest = ensure_sex_column(optbest, tag_col="tag" if "tag" in optbest.columns else None)
        opt_table = optbest.copy()
    elif not scored.empty:
        scored = ensure_sex_column(scored, tag_col="tag" if "tag" in scored.columns else None)
        if "CompositeScore" in scored.columns:
            opt_table = scored.sort_values("CompositeScore", ascending=False).groupby("sex", dropna=False).head(5)

    if not opt_table.empty:
        tname = "table_OPT_best_by_sex.csv"
        save_table(opt_table, OUT_TABLES, tname)
        for sex in sorted(opt_table["sex"].astype(str).unique()):
            sub = opt_table[opt_table["sex"].astype(str)==sex]
            labels = sub.get("tag", pd.Series([f"row{i}" for i in range(len(sub))])).astype(str)
            y = pd.to_numeric(sub.get("CompositeScore", np.nan), errors="coerce")
            fig, ax = plt.subplots(figsize=(7.2,4.2))
            ax.barh(range(len(y)), y.values)
            ax.set_yticks(range(len(y))); ax.set_yticklabels([l[:40] for l in labels]); ax.invert_yaxis()
            ax.set_xlabel("CompositeScore"); setup_axis_format(ax)
            title = short_title(f"[OPT] Top candidates — "
                                f"{'Male' if sex=='M' else ('Female' if sex=='F' else 'All')}"
                                f" / CompositeScore")
            ax.set_title(title)
            p = savefig(fig, OUT_FIGS, f"plot_OPT_top_{sex}.png")
            manifest.append({"title":title,"plot":p,"table":os.path.join(OUT_TABLES,tname)})

    # ───────── (4) Behavioral bias ─────────
    if not bhcmp.empty:
        bhcmp = ensure_sex_column(bhcmp, tag_col="tag" if "tag" in bhcmp.columns else None)
        colmap = {}
        for c in bhcmp.columns:
            cl = c.lower()
            if ("delta" in cl or "Δ" in c) and "ew" in cl:   colmap[c] = "Delta_EW"
            if ("delta" in cl or "Δ" in c) and "es95" in cl: colmap[c] = "Delta_ES95"
            if ("delta" in cl or "Δ" in c) and "ruin" in cl: colmap[c] = "Delta_Ruin"
        if colmap: bhcmp = bhcmp.rename(columns=colmap)
        for c in ["Delta_EW","Delta_ES95","Delta_Ruin"]:
            if c in bhcmp.columns:
                bhcmp[c] = pd.to_numeric(bhcmp[c], errors="coerce")

        tname = "table_BIAS_compare.csv"
        save_table(bhcmp, OUT_TABLES, tname)
        for sx in sorted(bhcmp["sex"].astype(str).unique()):
            sub = bhcmp[bhcmp["sex"].astype(str)==sx]
            if sub.empty or "Delta_EW" not in sub.columns: continue
            vals   = pd.to_numeric(sub["Delta_EW"], errors="coerce").fillna(0.0).values
            labels = sub.get("tag", pd.Series([f"row{i}" for i in range(len(sub))])).astype(str).str.slice(0,24).tolist()
            fig, ax = plt.subplots(figsize=(8,4.2))
            ax.bar(range(len(vals)), vals)
            ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("ΔEW (ON - OFF)"); setup_axis_format(ax)
            title = short_title(f"[BIAS] Effect on EW — "
                                f"{'Male' if sx=='M' else ('Female' if sx=='F' else 'All')}")
            ax.set_title(title)
            p = savefig(fig, OUT_FIGS, f"plot_BIAS_deltaEW_{sx}.png")
            manifest.append({"title":title,"plot":p,"table":os.path.join(OUT_TABLES,tname)})

    # ───────── manifest ─────────
    mf = pd.DataFrame(manifest)
    mf.to_csv("MANIFEST_plots_tables.csv", index=False)
    print(f"✔ Done. Generated {len(mf)} assets.")
    print("Figures :", os.path.abspath(OUT_FIGS))
    print("Tables  :", os.path.abspath(OUT_TABLES))
    print("Manifest:", os.path.abspath("MANIFEST_plots_tables.csv"))

if __name__ == "__main__":
    main()
