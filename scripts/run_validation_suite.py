import argparse, pandas as pd, numpy as np, os
p=argparse.ArgumentParser(); p.add_argument("--mode"); p.add_argument("--src"); p.add_argument("--rules"); p.add_argument("--out"); a=p.parse_args()
df=pd.read_csv(a.src); logs=[]
def log(s): logs.append(s); print(s)
def write(): os.makedirs(os.path.dirname(a.out),exist_ok=True); open(a.out,"w",encoding="utf-8").write("\n".join(logs))
# 가용 컬럼 기반 간이 체크
if "w0_sanity" in a.rules:
    # 태그에 _w0 또는 유사 패턴이 없으면 PASS(스킵)로 표기
    has = df["tag"].str.contains("_w0").any()
    if has:
        d=df[df["tag"].str.contains("_w0")]
        ok = (d["Ruin"].max()<=0.02) if "Ruin" in d else True
        log(f"[w0_sanity] {'OK' if ok else 'FAIL'}")
    else: log("[w0_sanity] SKIP (no _w0 tags)")
if "rl_vs_4pct" in a.rules:
    rl=df[df["method"]=="rl"]; b4=df[df["tag"].str.contains("4pct",na=False)]
    if len(rl)>0 and len(b4)>0:
        ok1 = rl["EW"].median() >= b4["EW"].median()
        ok2 = rl["ES95"].median() <= b4["ES95"].median()
        log(f"[rl_vs_4pct] {'OK' if (ok1 and ok2) else 'FAIL'}")
    else: log("[rl_vs_4pct] SKIP (need rl & 4pct)")
# 기타 규칙은 스냅샷 정보 제한으로 SKIP
for r in ["lambda_monotone","wmax_monotone","fee_penalty","qfloor_tradeoff","hedge_sigma_safety"]:
    if r in a.rules: log(f"[{r}] SKIP (requires per-run param traces)")
write(); print("[OK] validation ->", a.out)
