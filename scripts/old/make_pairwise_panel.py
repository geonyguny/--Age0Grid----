# scripts/make_pairwise_panel.py
import argparse, pandas as pd, numpy as np

METHODS = ["hjb","rl","rule"]
GROUP = ["seed","path_id","window","hedge_ratio","mix_kr","mix_us","mix_gold","es_mode"]
TARGET = "CompositeScore"   # 필요 시 변경

def boot_ci(x, reps=2000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(42)
    n=len(x)
    if n==0: return (np.nan,np.nan)
    bs = np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0,n,n)
        bs[i] = np.nanmean(x[idx])
    lo = np.quantile(bs, alpha/2)
    hi = np.quantile(bs, 1-alpha/2)
    return lo, hi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="_summary_scored.csv (or norm)")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--thr", type=float, default=1e-4, help="tie threshold")
    args = ap.parse_args()

    S = pd.read_csv(args.in_csv)
    need = set(GROUP + ["method", TARGET])
    assert need.issubset(S.columns), f"missing cols {need - set(S.columns)}"

    # 동일 경로에서 메서드별 점수 피벗
    P = S.groupby(GROUP + ["method"], dropna=False)[TARGET].mean().reset_index()
    Pv = P.pivot_table(index=GROUP, columns="method", values=TARGET, aggfunc="mean")
    Pv = Pv.dropna(subset=[m for m in METHODS if m in Pv.columns], how="any").copy()

    # 승/무/패 판정
    best = Pv.min(axis=1)  # 낮을수록 좋다(ES 가중 포함 CompositeScore 가 작은게 우수) ← 필요 시 부호 조정
    W = {}
    for m in METHODS:
        if m in Pv.columns:
            diff = Pv[m] - best
            W[m] = {
                "win": np.mean(diff<=args.thr),
                "tie": np.mean((diff>0) & (diff<=args.thr)),
            }
    # 부트스트랩 CI
    rows=[]
    n = len(Pv)
    for m in METHODS:
        if m not in W: continue
        wins = (Pv[m]-best<=args.thr).astype(float).values
        ties = (((Pv[m]-best)>0)&((Pv[m]-best)<=args.thr)).astype(float).values
        w_lo,w_hi = boot_ci(wins)
        t_lo,t_hi = boot_ci(ties)
        rows.append(dict(method=m,
                         n=n,
                         win_pct= W[m]["win"],
                         tie_pct= W[m]["tie"],
                         win_ci_lo=w_lo, win_ci_hi=w_hi,
                         tie_ci_lo=t_lo, tie_ci_hi=t_hi))
    out = pd.DataFrame(rows).sort_values("win_pct", ascending=False)
    out.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"[OK] pairwise saved -> {args.out_csv}")

if __name__=="__main__":
    main()
