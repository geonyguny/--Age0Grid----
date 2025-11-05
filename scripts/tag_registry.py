import argparse, pandas as pd, json, os
p=argparse.ArgumentParser(); p.add_argument("cmd"); p.add_argument("--from"); p.add_argument("--out"); a=p.parse_args()
if a.cmd!="sync": raise SystemExit("usage: tag_registry.py sync --from <metrics.csv> --out <json>")
df=pd.read_csv(os.path.join(a.__dict__["from"],"metrics.csv"))
reg=df.groupby(["tag","method"])["seed"].unique().apply(list).reset_index()
out={"items":[dict(tag=r.tag, method=r.method, seeds=r.seed) for r in reg.itertuples()]}
os.makedirs(os.path.dirname(a.out),exist_ok=True)
with open(a.out,"w",encoding="utf-8") as f: json.dump(out,f,ensure_ascii=False,indent=2)
print("[OK] tag_registry ->", a.out, f"(n={len(out['items'])})")
