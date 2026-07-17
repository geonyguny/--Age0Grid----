# -*- coding: utf-8 -*-
"""다자산(한국·미국·나스닥·금·미국장기채) KRW 수익률 → 실측 상관 → 최대샤프 포트폴리오
→ 시뮬레이션용 market csv 생성. 데이터: Yahoo Finance 월간 수정종가."""
import urllib.request, json, datetime, csv, itertools
import numpy as np

def fetch(sym, rng="25y"):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1mo&range={rng}"
    req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    d = json.loads(urllib.request.urlopen(req, timeout=20).read())
    r = d["chart"]["result"][0]
    ts = r["timestamp"]
    ind = r["indicators"]
    closes = (ind.get("adjclose",[{}])[0].get("adjclose") or ind["quote"][0]["close"])
    out = {}
    for t, c in zip(ts, closes):
        if c is None: continue
        ym = datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m")
        out[ym] = float(c)   # 월 마지막 관측 우선(덮어씀)
    return out

symbols = {"KR":"^KS11","US":"^GSPC","NDX":"^NDX","GOLD":"GC=F","TLT":"TLT","FX":"KRW=X"}
px = {k: fetch(v) for k, v in symbols.items()}
print("다운로드:", {k: len(v) for k, v in px.items()})

# [정제] KRW=X 오류 틱 제거: 원/달러 정상범위(800~2000원) 밖 관측 폐기
bad_fx = [m for m, v in px["FX"].items() if not (800.0 <= v <= 2000.0)]
for m in bad_fx: del px["FX"][m]
print(f"FX 오류 틱 제거: {len(bad_fx)}건")

months = sorted(set.intersection(*[set(v) for v in px.values()]))
months = [m for m in months if m >= "2003-12" and m <= "2026-06"]
print(f"공통 표본: {months[0]} ~ {months[-1]} ({len(months)}개월)")

# KRW 기준 월수익률 (미국계 자산 = USD수익률 × 환율변동 반영)
def rets(key, krw=False):
    out=[]
    for a, b in zip(months[:-1], months[1:]):
        r = px[key][b]/px[key][a] - 1.0
        if krw:
            r = (1+r)*(px["FX"][b]/px["FX"][a]) - 1.0
        out.append(r)
    return np.array(out)

R = {
    "KR":   rets("KR"),
    "US":   rets("US",  krw=True),
    "NDX":  rets("NDX", krw=True),
    "GOLD": rets("GOLD",krw=True),
    "TLT":  rets("TLT", krw=True),
}
names = list(R)
M = np.column_stack([R[k] for k in names])
# [정제] 잔여 이상월 제거: 어느 자산이든 |월수익률|>50%면 해당 월 전체 제외(정렬 유지)
mask = (np.abs(M) <= 0.50).all(axis=1)
n_drop = int((~mask).sum())
M = M[mask]
kept_months = [months[1:][i] for i in range(len(mask)) if mask[i]]
print(f"이상월 제거: {n_drop}건 → 사용 표본 {len(M)}개월")
mu_m = M.mean(axis=0); sd_m = M.std(axis=0, ddof=1)
mu_a = (1+mu_m)**12 - 1; sd_a = sd_m*np.sqrt(12)
C = np.corrcoef(M.T)

print("\n=== 실측 연환산 모멘트 (KRW, 명목, 2004~2026) ===")
for i,k in enumerate(names):
    print(f"  {k:>5}: mu={mu_a[i]*100:6.2f}%  sigma={sd_a[i]*100:5.2f}%")
print("\n=== 실측 상관행렬 ===")
print("       " + " ".join(f"{k:>6}" for k in names))
for i,k in enumerate(names):
    print(f"  {k:>5} " + " ".join(f"{C[i,j]:6.2f}" for j in range(len(names))))

# ── 최대샤프 (롱온리, 실무 캡: 개별 ≤50%, GOLD ≤25%, TLT ≤40%) ──
rf_nom = 0.025
S = np.outer(sd_a, sd_a)*C
best=None
grid=np.arange(0,1.0001,0.05)
caps={"KR":0.5,"US":0.5,"NDX":0.5,"GOLD":0.25,"TLT":0.4}
for w in itertools.product(grid, repeat=4):
    if sum(w)>1.0001: continue
    full=np.array(list(w)+[1.0-sum(w)])
    if any(full[i]>caps[names[i]]+1e-9 for i in range(5)) or full[-1]<-1e-9: continue
    m=float(full@mu_a); s=float(np.sqrt(full@S@full))
    sh=(m-rf_nom)/s
    if best is None or sh>best[0]: best=(sh,m,s,full.copy())
sh,m,s,wopt = best
print(f"\n=== 최대샤프 포트폴리오 (롱온리, 실무 캡) ===")
print("  배분: " + ", ".join(f"{k} {wopt[i]*100:.0f}%" for i,k in enumerate(names) if wopt[i]>0.001))
print(f"  명목 mu={m*100:.2f}%  sigma={s*100:.2f}%  Sharpe={sh:.3f}")
cpi_avg = 0.020
print(f"  실질 근사(mu-CPI {cpi_avg*100:.0f}%): mu={100*(m-cpi_avg):.2f}%  sigma={s*100:.2f}%")

# ── 시뮬레이션용 csv 저장 (포트폴리오 수익률 = risky_nom) ──
port = M @ wopt
with open(r"G:\01_simul\data\dev\market_globalmix.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["date","risky_nom","tbill_nom","cpi"])
    cpi=100.0
    for i,d in enumerate(kept_months):
        cpi*= (1+cpi_avg/12)
        w.writerow([d, f"{port[i]:.6f}", f"{rf_nom/12:.6f}", f"{cpi:.4f}"])
print(f"\n저장: data/dev/market_globalmix.csv ({len(port)}개월, 실측 혼합 수익률)")
print(f"참고: KR단일 샤프={( (mu_a[0]-rf_nom)/sd_a[0]):.3f} → 최적혼합 {sh:.3f}")
