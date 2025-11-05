import argparse, pandas as pd, os
p=argparse.ArgumentParser(); p.add_argument("--src"); p.add_argument("--figdir"); p.add_argument("--tables"); p.add_argument("--out"); a=p.parse_args()
df=pd.read_csv(a.src)
os.makedirs(os.path.dirname(a.out),exist_ok=True)
with pd.ExcelWriter(a.out) as xw:
    df.to_excel(xw, index=False, sheet_name="snapshot")
    # 간단 피벗
    if {"tag","EW","ES95"}.issubset(df.columns):
        pv=df.pivot_table(index="tag", values=["EW","ES95","Ruin","mean_WT"], aggfunc="median")
        pv.to_excel(xw, sheet_name="pivot_median")
print("[OK] final report ->", a.out)
