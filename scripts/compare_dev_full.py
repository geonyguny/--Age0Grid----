# scripts/compare_dev_full.py
import argparse
import pathlib as p
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# -----------------------------
# CLI
# -----------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--dev_src",  default=r".\outputs\_summary_scored.csv")
ap.add_argument("--full_src", default=r".\outputs\_logs\metrics.csv")
ap.add_argument("--out",      default=r".\outputs\dev_full_drift.xlsx")
ap.add_argument("--abs_tol",  type=float, default=0.05, help="ES95 절대 오차 허용치")
ap.add_argument("--rel_tol",  type=float, default=0.05, help="ES95 상대 오차 허용치(비율)")
args = ap.parse_args()

# -----------------------------
# 유틸
# -----------------------------
def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    """지정 컬럼을 전부 numeric으로 강제 변환 (실패 시 NaN)."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def normalize_alpha_mix(df: pd.DataFrame) -> pd.Series:
    """
    alpha_mix 문자열이 없으면 alpha_kr, alpha_us, alpha_au 조합으로 생성.
    모두 없으면 빈 문자열 반환.
    """
    if "alpha_mix" in df.columns and df["alpha_mix"].notna().any():
        return df["alpha_mix"].astype(str)
    ks = ["alpha_kr", "alpha_us", "alpha_au"]
    if all(k in df.columns for k in ks):
        return (df["alpha_kr"].astype(str) + "," +
                df["alpha_us"].astype(str) + "," +
                df["alpha_au"].astype(str))
    return pd.Series([""] * len(df), index=df.index)

def build_key(df: pd.DataFrame) -> pd.Series:
    """
    비교 key: method | alpha_mix | hedge_sigma_k(or hedge_ratio)
    필요 컬럼이 없어도 안전하게 처리.
    """
    method = df["method"].astype(str) if "method" in df.columns else ""
    alpha  = normalize_alpha_mix(df)
    hedge  = None
    if "hedge_sigma_k" in df.columns:
        hedge = df["hedge_sigma_k"].astype(str)
    elif "hedge_ratio" in df.columns:
        hedge = df["hedge_ratio"].astype(str)
    else:
        hedge = pd.Series([""] * len(df), index=df.index)
    return method + "|" + alpha + "|" + hedge

# -----------------------------
# 로드
# -----------------------------
dev  = pd.read_csv(args.dev_src)
full = pd.read_csv(args.full_src)

# key 생성
dev["key"]  = build_key(dev)
full["key"] = build_key(full)

# 숫자 컬럼 강제 numeric
num_cols_dev  = [c for c in ["EW", "ES95", "CompositeScore", "Ruin"] if c in dev.columns]
num_cols_full = [c for c in ["EW", "ES95", "Ruin"] if c in full.columns]
coerce_numeric(dev,  num_cols_dev)
coerce_numeric(full, num_cols_full)

# 집계
left = dev.groupby("key")[num_cols_dev].mean(numeric_only=True).add_prefix("dev_")
rght = full.groupby("key")[num_cols_full].mean(numeric_only=True).add_prefix("full_")
m = left.join(rght, how="inner")

# 드리프트 계산
if "dev_ES95" in m.columns and "full_ES95" in m.columns:
    m["d_ES95_abs"] = m["full_ES95"] - m["dev_ES95"]
    m["d_ES95_rel"] = m["d_ES95_abs"] / m["dev_ES95"]
else:
    m["d_ES95_abs"] = np.nan
    m["d_ES95_rel"] = np.nan

# 상관 통계(선택)
if "dev_CompositeScore" in m.columns and "full_ES95" in m.columns:
    rho = spearmanr(m["dev_CompositeScore"], m["full_ES95"], nan_policy="omit")[0]
else:
    rho = np.nan

# 임계치 플래그
m["flag_abs"] = m["d_ES95_abs"].abs() > args.abs_tol
m["flag_rel"] = m["d_ES95_rel"].abs() > args.rel_tol

summary_row = {
    "pairs":      int(len(m)),
    "abs_tol":    args.abs_tol,
    "rel_tol":    args.rel_tol,
    "n_abs_fail": int(m["flag_abs"].sum()),
    "n_rel_fail": int(m["flag_rel"].sum()),
    "spearman(dev CompScore vs full ES95)": rho,
}
overall_ok = (summary_row["n_abs_fail"] == 0 and summary_row["n_rel_fail"] == 0)
print(f"[DRIFT] pairs={summary_row['pairs']} abs_fail={summary_row['n_abs_fail']} "
      f"rel_fail={summary_row['n_rel_fail']} -> {'OK' if overall_ok else 'CHECK'}")

# -----------------------------
# 쓰기 (FutureWarning 방지: keyword-only)
# -----------------------------
out = p.Path(args.out)
with pd.ExcelWriter(out) as xw:
    m.reset_index().to_excel(excel_writer=xw, sheet_name="match", index=False)
    pd.DataFrame([summary_row]).to_excel(excel_writer=xw, sheet_name="stats", index=False)

print(f"[OK] {out}")
