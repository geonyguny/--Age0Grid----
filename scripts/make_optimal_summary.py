import re, json, pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
out_dir = Path(r"G:\01_simul\outputs")
candidates = [out_dir/"DEV_scored_enriched.csv", out_dir/"DEV_scored.csv",
              out_dir/"DEV_scored_clean.csv", out_dir/"DEV_metrics_snapshot.csv"]
src = next((str(p) for p in candidates if p.exists()), None)
if src is None: raise SystemExit("[ERR] 소스 CSV 없음: " + ", ".join(map(str,candidates)))
df = pd.read_csv(src)
for c in ["grid_type","sex","mort_id","ann_alpha","wrisk","cstar_m","fee_annual","age0","vpw","hedge",
          "ann_index","use_real_rf","market_mode","bias_on"]:
    if c not in df.columns: df[c] = pd.NA
pat_2d_wc=re.compile(r"DEV2D_wrisk_c_(?P<sex>[MF])_w(?P<w>[0-9.]+)_c(?P<c>[0-9.]+)")
def patch(row):
    tag=str(row.get("tag",""))
    m=pat_2d_wc.fullmatch(tag)
    if m:
        if pd.isna(row["sex"]): row["sex"]=m["sex"]
        if pd.isna(row["wrisk"]): row["wrisk"]=float(m["w"])
        if pd.isna(row["cstar_m"]): row["cstar_m"]=float(m["c"])
    return row
if "tag" in df.columns: df=df.apply(patch,axis=1)
score_col="CompositeScore"
if score_col not in df.columns:
    if {"EW","ES95"}.issubset(df.columns):
        df[score_col]=0.6*df["EW"].astype(float)-0.4*df["ES95"].astype(float)
    else:
        raise SystemExit("[ERR] 점수 컬럼 부재")
if "norm_EW" not in df.columns: df["norm_EW"]=pd.NA
if "norm_ES95" not in df.columns: df["norm_ES95"]=pd.NA
global_best=df.sort_values(score_col,ascending=False).head(1).copy(); global_best["selection_level"]="global"
for g in ["sex","mort_id"]:
    if g not in df.columns: df[g]=pd.NA
by_sm=(df.sort_values(score_col,ascending=False).groupby(["sex","mort_id"],dropna=False).head(1).copy())
by_sm["selection_level"]="by_sex_mort"
final=pd.concat([global_best,by_sm],ignore_index=True)
show=["selection_level","tag","grid_type","sex","mort_id","ann_alpha","wrisk","cstar_m","fee_annual","age0","vpw","hedge",
      "ann_index","use_real_rf","market_mode","bias_on","CompositeScore","norm_EW","norm_ES95"]
for c in show:
    if c not in final.columns: final[c]=pd.NA
final=final[show]
out_csv=out_dir/"OPTIMAL_Design_Summary.csv"
out_json=out_dir/"OPTIMAL_Design_Summary.json"
final.to_csv(out_csv,index=False,encoding="utf-8-sig")
items=[]
for _,r in final.iterrows():
    coords={k:(None if pd.isna(r[k]) else float(r[k]) if k in ["ann_alpha","wrisk","cstar_m","fee_annual","vpw"] else r[k])
            for k in ["ann_alpha","wrisk","cstar_m","fee_annual","age0","vpw","hedge"]}
    attrs ={k:(None if pd.isna(r[k]) else r[k]) for k in ["sex","mort_id","ann_index","use_real_rf","market_mode","bias_on"]}
    metrics={k:(None if pd.isna(r[k]) else float(r[k])) for k in ["CompositeScore","norm_EW","norm_ES95"]}
    items.append({"selection_level":r["selection_level"],"tag":r["tag"],"grid_type":r["grid_type"],"coords":coords,"attrs":attrs,"metrics":metrics})
kst=timezone(timedelta(hours=9))
out={"generated_at":datetime.now(kst).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),"source_scored_csv":src,"count":len(items),"items":items}
with open(out_json,"w",encoding="utf-8") as f: json.dump(out,f,ensure_ascii=False,indent=2)
print(f"[OK] CSV: {out_csv}")
print(f"[OK] JSON: {out_json}")
