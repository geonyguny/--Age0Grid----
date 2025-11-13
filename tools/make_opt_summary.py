import re, json, pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta

scored = r"G:\01_simul\outputs\DEV_scored_enriched.csv"
out_dir = Path(r"G:\01_simul\outputs")
out_csv = out_dir / "OPTIMAL_Design_Summary.csv"
out_json = out_dir / "OPTIMAL_Design_Summary.json"

df = pd.read_csv(scored)

# ---- (A) 필요한 변수열 표준화(없으면 생성) ----
param_cols = [
    "grid_type","sex","mort_id",
    "ann_alpha","wrisk","cstar_m",
    "fee_annual","age0","vpw",
    "hedge","ann_index","use_real_rf",
    "market_mode","bias_on"
]
for c in param_cols:
    if c not in df.columns:
        df[c] = pd.NA

# ---- (B) tag에서 변수 복원(누락 보완) ----
pat_1d_ann   = re.compile(r"DEV1D_ann_(?P<mort>[A-Z]+)_(?P<sex>[MF])_(?P<a>[0-9.]+)")
pat_2d_aw    = re.compile(r"DEV2D_ann_wrisk_(?P<mort>[A-Z]+)_(?P<sex>[MF])_a(?P<a>[0-9.]+)_w(?P<w>[0-9.]+)")
pat_2d_wc    = re.compile(r"DEV2D_wrisk_c_(?P<mort>[A-Z]+)_(?P<sex>[MF])_w(?P<w>[0-9.]+)_c(?P<c>[0-9.]+)")

def patch_from_tag(row):
    tag = str(row["tag"])
    m = pat_1d_ann.fullmatch(tag)
    if m:
        row["grid_type"] = row.get("grid_type", "1D") or "1D"
        row["mort_id"]   = row["mort_id"] if pd.notna(row["mort_id"]) else m["mort"]
        row["sex"]       = row["sex"] if pd.notna(row["sex"]) else m["sex"]
        if pd.isna(row["ann_alpha"]): row["ann_alpha"] = float(m["a"])
        return row
    m = pat_2d_aw.fullmatch(tag)
    if m:
        row["grid_type"] = row.get("grid_type", "2D") or "2D"
        row["mort_id"]   = row["mort_id"] if pd.notna(row["mort_id"]) else m["mort"]
        row["sex"]       = row["sex"] if pd.notna(row["sex"]) else m["sex"]
        if pd.isna(row["ann_alpha"]): row["ann_alpha"] = float(m["a"])
        if pd.isna(row["wrisk"]):     row["wrisk"] = float(m["w"])
        return row
    m = pat_2d_wc.fullmatch(tag)
    if m:
        row["grid_type"] = row.get("grid_type", "2D") or "2D"
        row["mort_id"]   = row["mort_id"] if pd.notna(row["mort_id"]) else m["mort"]
        row["sex"]       = row["sex"] if pd.notna(row["sex"]) else m["sex"]
        if pd.isna(row["wrisk"]):     row["wrisk"] = float(m["w"])
        if pd.isna(row["cstar_m"]):   row["cstar_m"] = float(m["c"])
        return row
    return row

df = df.apply(patch_from_tag, axis=1)

# ---- (C) 최적점 선별: 전역 1개 + by_sex_mort 1개씩 ----
score_col = "CompositeScore"
if score_col not in df.columns:
    raise SystemExit(f"'{score_col}' column missing in {scored}")

# 전역 최적
global_best = df.sort_values(score_col, ascending=False).head(1).copy()
global_best["selection_level"] = "global"

# by_sex_mort 최적
grp_cols = ["sex","mort_id"]
by_sm = (df.sort_values(score_col, ascending=False)
           .groupby(grp_cols, dropna=False).head(1).copy())
by_sm["selection_level"] = "by_sex_mort"

final = pd.concat([global_best, by_sm], ignore_index=True)

# ---- (D) 최소 표시 컬럼 정리(읽기 좋게) ----
show_cols = [
  "selection_level","tag","grid_type",
  "sex","mort_id",
  "ann_alpha","wrisk","cstar_m",
  "fee_annual","age0","vpw","hedge",
  "ann_index","use_real_rf","market_mode","bias_on",
  "CompositeScore","norm_EW","norm_ES95"
]
for c in show_cols:
    if c not in final.columns:
        final[c] = pd.NA
final = final[show_cols]

# CSV 저장
final.to_csv(out_csv, index=False, encoding="utf-8-sig")

# JSON 저장(사람이 보기 좋게)
items = []
for _, r in final.iterrows():
    coords = {k: (None if pd.isna(r[k]) else float(r[k]) if k in ["ann_alpha","wrisk","cstar_m","fee_annual","vpw"] else r[k])
              for k in ["ann_alpha","wrisk","cstar_m","fee_annual","age0","vpw","hedge"]}
    attrs  = {k: (None if pd.isna(r[k]) else r[k]) for k in ["sex","mort_id","ann_index","use_real_rf","market_mode","bias_on"]}
    metrics= {k: (None if pd.isna(r[k]) else float(r[k])) for k in ["CompositeScore","norm_EW","norm_ES95"]}
    items.append({
        "selection_level": r["selection_level"],
        "tag": r["tag"],
        "grid_type": r["grid_type"],
        "coords": coords,
        "attrs": attrs,
        "metrics": metrics
    })

kst = timezone(timedelta(hours=9))
out = {
    "generated_at": datetime.now(kst).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
    "source_scored_csv": str(Path(scored)),
    "count": len(items),
    "items": items
}
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"[OK] CSV: {out_csv}")
print(f"[OK] JSON: {out_json}")
