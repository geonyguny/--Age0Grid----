# scripts/quality_checks.py
import argparse, pandas as pd, numpy as np
from itertools import combinations

# ─────────────────────────────────────────────────────────
# 키/설정
# ─────────────────────────────────────────────────────────
KEYS_SEED_STAB = ["es_mode","hedge_ratio","mix_kr","mix_us","mix_gold","es_metric"]
ID_KEYS_WIN_GEN = ["hedge_ratio","mix_kr","mix_us","mix_gold","es_metric","es_mode"]
ENV_WITH_SEED   = ["es_mode","window","hedge_ratio","mix_kr","mix_us","mix_gold","es_metric","seed"]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="_summary_scored_norm.csv (또는 CompositeScore 포함 CSV)")
    ap.add_argument("--out_rank_stability", default="./outputs/_rank_stability.csv")
    ap.add_argument("--out_window_generalization", default="./outputs/_window_generalization.csv")
    ap.add_argument("--out_pairwise_gap", default="./outputs/_pairwise_gap_stats.csv")
    ap.add_argument("--use_col", default="Composite_rescored",
                    help="승부/순위에 사용할 점수 열명 (기본: Composite_rescored, 폴백: CompositeScore)")
    ap.add_argument("--ties_thr", type=float, default=1e-4, help="동률로 간주할 절대차 임계값")
    return ap.parse_args()

# ─────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────
def method_norm(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.lower()

def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def pick_use_col(df: pd.DataFrame, prefer: str) -> str:
    if prefer in df.columns:
        return prefer
    if "CompositeScore" in df.columns:
        return "CompositeScore"
    raise ValueError("점수열이 없습니다: neither Composite_rescored nor CompositeScore")

def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    # NaN 제거 공통 인덱스
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]; b = b[mask]
    if a.size < 2:
        return np.nan
    av = a.mean(); bv = b.mean()
    da = a - av; db = b - bv
    denom = np.sqrt((da*da).sum() * (db*db).sum())
    if denom == 0:
        return np.nan
    return float((da*db).sum() / denom)

def spearman_rho_from_ranks(rank_a: np.ndarray, rank_b: np.ndarray) -> float:
    # Spearman = Pearson(rank(a), rank(b))
    return pearson_corr(rank_a, rank_b)

def kendall_tau_b(a: np.ndarray, b: np.ndarray) -> float:
    """
    Kendall's tau-b for two ranking arrays with possible ties.
    a, b: rank vectors (ties allowed, same length)
    O(n^2) since n = number of methods (작음).
    """
    n = len(a)
    if n < 2:
        return np.nan
    C = D = T1 = T2 = 0
    for i in range(n-1):
        for j in range(i+1, n):
            ai, aj = a[i], a[j]
            bi, bj = b[i], b[j]
            if not (np.isfinite(ai) and np.isfinite(aj) and np.isfinite(bi) and np.isfinite(bj)):
                continue
            if ai == aj and bi == bj:
                # double tie: doesn't affect C/D, counts to both tie terms
                T1 += 1; T2 += 1
            elif ai == aj:
                T1 += 1
            elif bi == bj:
                T2 += 1
            else:
                s1 = np.sign(ai - aj)
                s2 = np.sign(bi - bj)
                v = s1 * s2
                if v > 0:
                    C += 1
                elif v < 0:
                    D += 1
                # if v == 0: already handled by tie branches
    denom = np.sqrt((C + D + T1) * (C + D + T2))
    if denom == 0:
        return np.nan
    return float((C - D) / denom)

# ─────────────────────────────────────────────────────────
# 1) 시드 안정성
# ─────────────────────────────────────────────────────────
def rank_stability(df: pd.DataFrame, use_col: str) -> pd.DataFrame:
    dfs = df.copy()
    dfs["method"] = method_norm(dfs["method"])
    rows=[]
    for k, g in dfs.groupby(KEYS_SEED_STAB, dropna=False):
        # seed x method 테이블
        p = g.pivot_table(index="seed", columns="method", values=use_col, aggfunc="min")
        p = p.dropna(axis=0, how="any")
        if p.shape[0] < 2:
            continue
        # 낮을수록 우수 → rank
        ranks = p.rank(axis=1, method="average", ascending=True)
        taus=[]; rhos=[]
        for (i, j) in combinations(ranks.index, 2):
            a = ranks.loc[i].to_numpy()
            b = ranks.loc[j].to_numpy()
            kt = kendall_tau_b(a, b)
            sp = spearman_rho_from_ranks(a, b)
            if np.isfinite(kt): taus.append(kt)
            if np.isfinite(sp): rhos.append(sp)
        if not taus and not rhos:
            continue
        rows.append({
            **dict(zip(KEYS_SEED_STAB, k)),
            "kendall_tau_mean": float(np.mean(taus)) if len(taus) else np.nan,
            "spearman_rho_mean": float(np.mean(rhos)) if len(rhos) else np.nan,
            "n_seed_pairs": int(len(taus))
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────
# 2) 윈도 일반화
# ─────────────────────────────────────────────────────────
def window_generalization(df: pd.DataFrame, use_col: str, ties_thr: float) -> pd.DataFrame:
    dfs = df.copy()
    dfs["method"] = method_norm(dfs["method"])
    winners = []
    for k, g in dfs.groupby(ID_KEYS_WIN_GEN + ["window"], dropna=False):
        g2 = g.sort_values(by=use_col, ascending=True)
        if g2.empty:
            continue
        best = float(g2.iloc[0][use_col])
        tied = g2.loc[(g2[use_col] - best).abs() <= ties_thr, "method"].tolist()
        winners.append({**dict(zip(ID_KEYS_WIN_GEN + ["window"], k)),
                        "winner": g2.iloc[0]["method"],
                        "ties": int(len(tied))})
    win_df = pd.DataFrame(winners)
    if win_df.empty:
        return pd.DataFrame(columns=ID_KEYS_WIN_GEN + ["win_consistency_rate","pairs"])
    rows=[]
    for k, sub in win_df.groupby(ID_KEYS_WIN_GEN, dropna=False):
        sub = sub.reset_index(drop=True)
        if sub.shape[0] < 2:
            continue
        same=0; tot=0
        for i, j in combinations(range(len(sub)), 2):
            tot += 1
            if sub.loc[i, "winner"] == sub.loc[j, "winner"]:
                same += 1
        rows.append({**dict(zip(ID_KEYS_WIN_GEN, k)),
                     "win_consistency_rate": (same / tot) if tot else np.nan,
                     "pairs": int(tot)})
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────
# 3) Δ분포(환경 내 최우수 대비 격차)
# ─────────────────────────────────────────────────────────
def pairwise_gap_stats(df: pd.DataFrame, use_col: str) -> pd.DataFrame:
    dfs = df.copy()
    dfs["method"] = method_norm(dfs["method"])
    gap_rows=[]
    for k, g in dfs.groupby(ENV_WITH_SEED, dropna=False):
        g2 = g.sort_values(by=use_col, ascending=True)
        if g2.empty:
            continue
        best = float(g2.iloc[0][use_col])
        for _, r in g2.iterrows():
            gap_rows.append({**dict(zip(ENV_WITH_SEED, k)),
                             "method": r["method"],
                             "delta_to_best": float(r[use_col] - best)})
    gap = pd.DataFrame(gap_rows)
    if gap.empty:
        return pd.DataFrame(columns=["method","n","median","q25","q75","iqr"])
    by_m = gap.groupby("method")["delta_to_best"]
    stats = pd.DataFrame({
        "n": by_m.count(),
        "median": by_m.median(),
        "q25": by_m.quantile(0.25),
        "q75": by_m.quantile(0.75),
        "iqr":  by_m.quantile(0.75) - by_m.quantile(0.25)
    }).reset_index()
    return stats

# ─────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────
def main():
    args = parse_args()
    S = pd.read_csv(args.in_csv)

    # 누락 컬럼 보정
    for c in ["es_metric","es_mode","window","hedge_ratio","mix_kr","mix_us","mix_gold","seed","method"]:
        if c not in S.columns:
            S[c] = np.nan

    use_col = pick_use_col(S, args.use_col)
    S[use_col] = coerce_numeric(S[use_col])
    S = S[~S[use_col].isna()].copy()
    if S.empty:
        raise ValueError("사용할 점수 데이터가 없습니다.")

    stab = rank_stability(S, use_col)
    stab.to_csv(args.out_rank_stability, index=False, encoding="utf-8")

    gen  = window_generalization(S, use_col, args.ties_thr)
    gen.to_csv(args.out_window_generalization, index=False, encoding="utf-8")

    gap  = pairwise_gap_stats(S, use_col)
    gap.to_csv(args.out_pairwise_gap, index=False, encoding="utf-8")

    print(f"[OK] saved -> {args.out_rank_stability}, {args.out_window_generalization}, {args.out_pairwise_gap}")

if __name__ == "__main__":
    main()
