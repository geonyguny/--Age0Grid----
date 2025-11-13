import argparse, os, json
import pandas as pd
import matplotlib.pyplot as plt

# --------- args ---------
ap = argparse.ArgumentParser()
ap.add_argument("--snapshot", required=True)   # outputs/DEV_metrics_snapshot.csv
ap.add_argument("--scored",   required=True)   # outputs/DEV_scored_clean.csv
ap.add_argument("--outdir",   required=True)   # outputs
args = ap.parse_args()

os.makedirs(os.path.join(args.outdir, "figs", "1d"), exist_ok=True)

# --------- load ---------
snap = pd.read_csv(args.snapshot)
scored = pd.read_csv(args.scored)

# 기대 컬럼 예시: tag, sex, EW, ES95, fee_annual, w_max, ann_alpha, age0, cstar_m, ...
# tag 패턴: DEV1D_{var}_{mort}_{sex}_{value}

# --------- helper: parse var & value from tag ---------
def parse_1d(row):
    tag = str(row.get("tag",""))
    parts = tag.split("_")
    # DEV1D_ann_BASE_M_0.25  → var=ann, sex=M, value=0.25
    if len(parts) >= 5 and parts[0]=="DEV1D":
        var  = parts[1]
        sex  = parts[3]
        val  = parts[4]
        try:
            # 숫자 캐스팅
            if var in ("ann","wrisk","hedge","vpw","fee"): x = float(val)
            elif var=="age": x = int(val)
            elif var.startswith("mix"): x = parts[1].replace("mix","") or "mix"
            else: x = val
        except: x = val
        return pd.Series({"_var":var, "_sex":sex, "_x":x})
    return pd.Series({"_var":None,"_sex":None,"_x":None})

snap = pd.concat([snap, snap.apply(parse_1d, axis=1)], axis=1)
snap = snap.dropna(subset=["_var","_sex"])

# VPW는 내부 키가 cstar_m일 수 있으므로 별칭 처리
if "cstar_m" in snap.columns and "VPW" not in snap.columns:
    snap["VPW"] = snap["cstar_m"]

# --------- 1D plot per variable x sex ---------
VAR_ORDER = ["ann","wrisk","hedge","vpw","fee","age","mix"]
METRICS   = ["EW","ES95"]

def plot_1d(df, var, sex):
    d = df[(df["_var"]==var) & (df["_sex"]==sex)].copy()
    if d.empty: return
    # x축 정렬
    try:
        d = d.sort_values(by="_x")
    except:
        pass
    for m in METRICS:
        if m not in d.columns: continue
        plt.figure()
        plt.plot(d["_x"], d[m], marker="o")
        plt.xlabel(var)
        plt.ylabel(m)
        plt.title(f"{var.upper()} 1D — {sex}")
        out = os.path.join(args.outdir, "figs", "1d", f"1D_{var}_{sex}_{m}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()

for var in VAR_ORDER:
    for sex in ("M","F"):
        plot_1d(snap, var, sex)

# --------- 요약 테이블(엑셀) ---------
xlsx = os.path.join(args.outdir, "design_tables.xlsx")
with pd.ExcelWriter(xlsx, engine="xlsxwriter") as wx:
    for var in VAR_ORDER:
        for sex in ("M","F"):
            d = snap[(snap["_var"]==var) & (snap["_sex"]==sex)][["_x","EW","ES95","tag"]].copy()
            if d.empty: continue
            d = d.sort_values(by="_x")
            d.rename(columns={"_x":var}, inplace=True)
            d.to_excel(wx, sheet_name=f"{var}_{sex}", index=False)

# --------- 최적설계 선택(CompositeScore 기준) ---------
# scored에는 CompositeScore 또는 CompositeScore_benefit가 존재
score_col = "CompositeScore_benefit" if "CompositeScore_benefit" in scored.columns else "CompositeScore"
sc = scored.copy()
sc[score_col] = pd.to_numeric(sc[score_col], errors="coerce")
sc = sc.dropna(subset=[score_col])

# sex, mort_id 기준 top1 뽑기(없으면 sex만)
has_sex  = "sex" in sc.columns
has_mort = "mort_id" in sc.columns
grp_keys = []
if has_sex: grp_keys.append("sex")
if has_mort: grp_keys.append("mort_id")
if not grp_keys:
    sc["_dummy"]=1; grp_keys=["_dummy"]

tops = sc.sort_values(score_col, ascending=False).groupby(grp_keys).head(1).copy()

# 태그에서 파라미터 복원(가벼운 규칙 기반)
def decode_from_tag(tag, base):
    # 기본값 반영 후 tag 덮어쓰기
    out = dict(ann_alpha=base["ann_alpha"], w_max=base["w_max"], hedge=base["hedge"],
               vpw=base["vpw"], fee_annual=base["fee_annual"], age0=base["age0"], mix=base["mix"])
    parts = str(tag).split("_")
    # DEV1D_var_mort_sex_val or DEV2D_var1_var2_...
    try:
        if parts[0]=="DEV1D":
            var = parts[1]
            sex = parts[3]
            val = parts[4]
            if var=="ann":   out["ann_alpha"]  = float(val)
            if var=="wrisk": out["w_max"]      = float(val)
            if var=="hedge": out["hedge"]      = float(val)
            if var=="vpw":   out["vpw"]        = float(val)
            if var=="fee":   out["fee_annual"] = float(val)
            if var=="age":   out["age0"]       = int(val)
            if var.startswith("mix"): out["mix"] = var.replace("mix_","").upper()
        # DEV2D 태그가 들어왔을 경우는 그대로 base 유지(필요 시 확장)
    except: pass
    return out

# 기본값(당신의 실험 기본값과 동일하게 맞춤)
BASE = dict(ann_alpha=0.0, w_max=0.4, hedge=0.5, vpw=0.05, fee_annual=0.004, age0=55, mix="EQUAL3")

rows = []
for _, r in tops.iterrows():
    params = decode_from_tag(r["tag"], BASE)
    row = dict(sex=r.get("sex","M"), tag=r["tag"])
    row.update(params)
    rows.append(row)

opt_df = pd.DataFrame(rows)
opt_csv = os.path.join(args.outdir, "optimal_selections.csv")
opt_df.to_csv(opt_csv, index=False)

# 행동편향 비교용 리마인더
readme = f"""# Design report generated
- Snapshot: {os.path.abspath(args.snapshot)}
- Scored  : {os.path.abspath(args.scored)}

## Outputs
- 1D charts  : outputs/figs/1d/*.png
- Tables     : outputs/design_tables.xlsx
- Optimal    : outputs/optimal_selections.csv

## Next
- Re-run only the optimal rows with Bias=on (same seeds) to measure behavioral impact.
"""
with open(os.path.join(args.outdir, "DESIGN_README.txt"), "w", encoding="utf-8") as f:
    f.write(readme)

print("[OK] 1D charts/tables + optimal selections done.")
