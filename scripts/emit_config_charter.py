import argparse, json, os, datetime as dt
cfg=dict(market_mode="bootstrap", bootstrap_block=24, q_floor=0.02, fee_annual=0.004,
         alpha=0.95, lambda_term=0.8, beta=0.996, horizon_years=35,
         mortality="on", sex="M", max_age=110, w_max=0.70)
p=argparse.ArgumentParser(); p.add_argument("--out"); a=p.parse_args()
os.makedirs(os.path.dirname(a.out),exist_ok=True)
cfg["_meta"]=dict(timestamp=dt.datetime.now().isoformat(timespec="seconds"))
with open(a.out,"w",encoding="utf-8") as f: json.dump(cfg,f,ensure_ascii=False,indent=2)
print("[OK] config_charter ->", a.out)
