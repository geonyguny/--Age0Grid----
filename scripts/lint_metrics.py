import argparse, pandas as pd, sys
p=argparse.ArgumentParser(); p.add_argument("--src"); p.add_argument("--fail_on_mixed_scales", default="off")
a=p.parse_args()
df=pd.read_csv(a.src)
need={"tag","method","es_mode","EW","ES95","Ruin","mean_WT"}
miss=need - set(df.columns)
if miss: print("[FAIL] missing columns:", ",".join(sorted(miss))); sys.exit(1)
# es_mode 혼재 경고
em=df["es_mode"].unique().tolist()
if len(em)>1 and a.fail_on_mixed_scales=="on":
    print("[FAIL] mixed es_mode:", em); sys.exit(2)
print("[OK] lint")
