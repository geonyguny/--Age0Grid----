# scripts/validate_sanity.py
import pandas as pd, numpy as np, argparse, pathlib as p

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=r".\outputs\_logs\metrics.csv")
ap.add_argument("--out", default=r".\outputs\sanity_report.txt")
args = ap.parse_args()

df = pd.read_csv(args.src)
rep = []

def chk_monotone(tag_like, var_col, metric, direction="+"):
  d = df[df["tag"].astype(str).str.startswith(tag_like)].copy()
  if var_col not in d.columns or metric not in d.columns or d.empty: return
  d[var_col] = pd.to_numeric(d[var_col], errors="coerce")
  d = d.dropna(subset=[var_col, metric]).sort_values(var_col)
  if len(d)<3: return
  corr = np.corrcoef(d[var_col], d[metric])[0,1]
  ok = (corr>0.2) if direction=="+" else (corr<-0.2)
  rep.append(f"[{tag_like}] {var_col} vs {metric} corr={corr:.2f}  -> {'OK' if ok else 'CHECK'}")

chk_monotone("OAT_", "hedge_sigma_k", "ES95", "+")   # sigma hedge ↑ → ES95 ↑(wealth 기준) 또는 ↓(loss 기준) 프로젝트 컨벤션에 맞게 조정
chk_monotone("OAT_", "hedge_sigma_k", "EW",   "-")

p.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
p.Path(args.out).write_text("\n".join(rep) if rep else "NO TESTS")
print(f"[OK] {args.out}")
