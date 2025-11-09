# -*- coding: utf-8 -*-
import os, json, numpy as np
from src.metrics import es95_loss_from_wealth

ROOT = r".\outputs"
CANDIDATE_FILES = [
    "eval_WT_wealth.npy", "eval_WT_wealth.csv", "wealth_eval.npy", "wealth_eval.csv"
]

def load_wealth_vec(path):
    if path.endswith(".npy"):
        return np.load(path).reshape(-1)
    elif path.endswith(".csv"):
        import pandas as pd
        s = pd.read_csv(path, header=None).iloc[:,0]
        return s.to_numpy(dtype=float).reshape(-1)
    return None

for d in os.listdir(ROOT):
    dd = os.path.join(ROOT, d)
    if not os.path.isdir(dd): 
        continue
    mj = os.path.join(dd, "metrics.json")
    if not os.path.exists(mj): 
        continue

    wealth = None
    for fn in CANDIDATE_FILES:
        fp = os.path.join(dd, fn)
        if os.path.exists(fp):
            w = load_wealth_vec(fp)
            if w is not None and w.size>0:
                wealth = w
                break

    if wealth is None:
        print(f"[skip] wealth vector not found in {dd}")
        continue

    es, n = es95_loss_from_wealth(wealth)

    try:
        m = json.load(open(mj, "r", encoding="utf-8"))
    except:
        print(f"[bad json] {mj}")
        continue

    m["ES95"] = float(es)
    m["es95_n"] = int(n)
    m["es95_source"] = "recomputed_from_eval_vector"

    json.dump(m, open(mj, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[ok] {dd}: ES95={m['ES95']:.6f} (n={n})")
