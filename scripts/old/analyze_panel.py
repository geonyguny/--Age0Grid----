# scripts/analyze_panel.py
import argparse, pandas as pd, numpy as np
from collections import defaultdict

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="summary_scored_norm.csv (rescored 포함 가능)")
    ap.add_argument("--out_panel", default="./outputs/_panel_summary.csv")
    ap.add_argument("--out_ci", default="./outputs/_panel_win_ci.csv")
    ap.add_argument("--use_col", default="Composite_rescored", help="승부 판단에 사용할 점수 열(Composite_rescored | CompositeScore)")
    ap.add_argument("--ties_thr", type=float, default=1e-4, help="동률 임계값(절대차)")
    ap.add_argument("--n_boot", type=int, default=2000, help="부트스트랩 횟수")
    return ap.parse_args()

def group_key_no_seed(r):
    return (r["es_mode"], r["window"], r["hedge_ratio"], r["mix_kr"], r["mix_us"], r["mix_gold"], r["es_metric"])

def main():
    args = parse_args()
    df = pd.read_csv(args.in_csv)
    use = args.use_col if args.use_col in df.columns else "CompositeScore"
    df = df[~df[use].isna()].copy()

    # 패널 단위(시드 제외 동일 환경)별 승자/동률 집계
    wins = []
    ties = []
    for k, g in df.groupby(df.apply(group_key_no_seed, axis=1)):
        g2 = g.sort_values(by=use, ascending=True)
        if g2.empty: 
            continue
        best = float(g2.iloc[0][use])
        tied = g2.loc[(g2[use] - best).abs() <= args.ties_thr, "method"]
        wins.append(g2.iloc[0]["method"])
        ties.extend(list(tied.values))

    def count_series(items):
        s = pd.Series(items)
        return s.value_counts().rename_axis("method").reset_index(name="count")

    win_tbl = count_series(wins); win_tbl["type"]="win"
    tie_tbl = count_series(ties); tie_tbl["type"]="tie"
    panel_summary = pd.concat([win_tbl, tie_tbl], ignore_index=True)
    panel_summary.to_csv(args.out_panel, index=False, encoding="utf-8")

    # 부트스트랩 CI (win rate)
    methods = sorted(df["method"].dropna().unique())
    wins_arr = np.array(wins)
    n = len(wins_arr)
    ci_rows=[]
    if n>0:
        rng = np.random.default_rng(42)
        for m in methods:
            boot = []
            for _ in range(args.n_boot):
                sample = wins_arr[rng.integers(0, n, n)]
                boot.append((sample==m).mean())
            lo, hi = np.percentile(boot, [2.5,97.5])
            ci_rows.append({"method": m, "win_rate_boot_mean": float(np.mean(boot)), "ci_lo": float(lo), "ci_hi": float(hi), "n_panels": n})
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(args.out_ci, index=False, encoding="utf-8")
    print(f"[OK] panel -> {args.out_panel}, ci -> {args.out_ci}")

if __name__ == "__main__":
    main()
