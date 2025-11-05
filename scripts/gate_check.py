import argparse, pandas as pd, numpy as np, re, sys, json, os
p=argparse.ArgumentParser(); p.add_argument("--src"); p.add_argument("--rules"); p.add_argument("--out"); a=p.parse_args()
df=pd.read_csv(a.src)
res=[]
def add(rule,ok,msg): res.append({"rule":rule,"ok":bool(ok),"msg":msg})
# OAT: tag like DEV_OAT_*_h{val} or *_la{val}
def parse_val(tag, key):
    m=re.search(rf"{key}([0-9]+(\.[0-9]+)?)", tag)
    return float(m.group(1)) if m else np.nan
if "OAT_shape" in a.rules:
    d=df[df["tag"].str.startswith("DEV_OAT_")].copy()
    if len(d)==0: add("OAT_shape",False,"no DEV_OAT_* rows")
    else:
        # 그룹별(변수축) EW/ES95 단조성 근사: Spearman rho>|0.6|
        ok_all=True; msgs=[]
        for key in ["h","la"]:
            g=d.dropna().copy(); g["x"]=g["tag"].apply(lambda t: parse_val(t,key))
            g=g[~g["x"].isna()]
            if len(g)>=3:
                rho_ew=g[["x","EW"]].corr(method="spearman").iloc[0,1]
                rho_es=g[["x","ES95"]].corr(method="spearman").iloc[0,1]
                ok= (abs(rho_ew)>=0.6) or (abs(rho_es)>=0.6)
                ok_all=ok_all and ok; msgs.append(f"{key}: rho(EW)={rho_ew:.2f}, rho(ES)={rho_es:.2f}")
        add("OAT_shape",ok_all,"; ".join(msgs) if msgs else "insufficient")
# 2D 프런티어: 동일 ES에서 EW↑(혹은 반대) 단일분지 근사 — 상관계수 부호 일관성
if "2D_frontier" in a.rules:
    d=df[df["tag"].str.startswith("DEV_2D_")].copy()
    if len(d)<4: add("2D_frontier",False,"few DEV_2D_* rows")
    else:
        rho=d[["EW","ES95"]].corr().iloc[0,1]
        add("2D_frontier", True if rho<0 else False, f"corr(EW,ES95)={rho:.2f} (expect negative)")
# reseed: 동일 tag-prefix에서 seed별 순위 일치율
if "reseed_shape" in a.rules:
    d=df[df["tag"].str.startswith(("DEV_OAT_","DEV_2D_"))].copy()
    if "seed" not in d.columns or d["seed"].nunique()<2:
        add("reseed_shape",False,"need >=2 seeds")
    else:
        ok=True; msg=[]
        for prefix in ["DEV_OAT_","DEV_2D_"]:
            g=d[d["tag"].str.startswith(prefix)]
            if len(g)==0: continue
            piv=g.pivot_table(index="tag", columns="seed", values="CompositeScore", aggfunc="last")
            if piv.shape[1]>=2:
                rank=piv.rank(ascending=False)
                # 상위3 순위 교집합 비율
                try:
                    s=list(rank.columns)
                    top=set(rank.sort_values(s[0]).index[:3])
                    top2=set(rank.sort_values(s[1]).index[:3])
                    inter=len(top&top2)/max(1,len(top))
                    ok= ok and (inter>=0.66); msg.append(f"{prefix} top3 overlap={inter:.2f}")
                except: pass
        add("reseed_shape",ok,"; ".join(msg) if msg else "insufficient")
os.makedirs(os.path.dirname(a.out),exist_ok=True)
pd.DataFrame(res).to_csv(a.out,index=False)
print("[OK] gate_check ->", a.out); 
