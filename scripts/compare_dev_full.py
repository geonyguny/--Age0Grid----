# scripts/compare_dev_full.py
import argparse
import pathlib as p
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--dev_src",  default=r".\outputs\_summary_scored.csv")
ap.add_argument("--full_src", default=r".\outputs\_logs\metrics.csv")
ap.add_argument("--out",      default=r".\outputs\dev_full_drift.xlsx")
ap.add_argument("--abs_tol",  type=float, default=0.05)
ap.add_argument("--rel_tol",  type=float, default=0.05)
# 필터/정규화 옵션
ap.add_argument("--tag_startswith")
ap.add_argument("--method")
ap.add_argument("--es_mode")
ap.add_argument("--round_mix", type=int, default=2, help="alpha_mix 라운딩 자리")
ap.add_argument("--round_h",   type=int, default=2, help="hedge_sigma_k 라운딩 자리")
# 키에 data_profile까지 포함할지
ap.add_argument("--include_profile_in_key", choices=["on","off"], default="off")
args = ap.parse_args()

def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def normalize_alpha_mix(df: pd.DataFrame, nd: int) -> pd.Series:
    """alpha_mix 문자열 없으면 alpha_kr/us/au 조합 생성 후 소수 nd자리로 라운드."""
    if "alpha_mix" in df.columns and df["alpha_mix"].notna().any():
        s = df["alpha_mix"].astype(str)
    else:
        ks = ["alpha_kr", "alpha_us", "alpha_au"]
        if all(k in df.columns for k in ks):
            s = (df["alpha_kr"].astype(float).round(nd).astype(str) + "," +
                 df["alpha_us"].astype(float).round(nd).astype(str) + "," +
                 df["alpha_au"].astype(float).round(nd).astype(str))
        else:
            return pd.Series([""] * len(df), index=df.index)
    # 이미 문자열인 경우도 라운딩 정규화 시도
    def _round_csv(x: str) -> str:
        try:
            a,b,c = [float(t) for t in str(x).split(",")]
            return f"{round(a,nd)},{round(b,nd)},{round(c,nd)}"
        except Exception:
            return str(x)
    return s.map(_round_csv)

def hedge_flag(df: pd.DataFrame) -> pd.Series:
    """hedge 컬럼(on/off) 또는 True/False 추정."""
    if "hedge" in df.columns:
        return df["hedge"].astype(str).str.lower().map(lambda z: "on" if z=="on" else "off").fillna("off")
    # 대체: hedge_ratio/hedge_sigma_k 존재하면 on으로 볼지 여부는 프로젝트 규칙에 따름.
    # 보수적으로 명시 플래그 없으면 off로 처리
    return pd.Series(["off"] * len(df), index=df.index)

def build_key(df: pd.DataFrame, nd_mix: int, nd_h: int, include_profile: bool) -> pd.Series:
    method = df["method"].astype(str) if "method" in df.columns else ""
    alpha  = normalize_alpha_mix(df, nd_mix)
    # es_mode
    esm = df["es_mode"].astype(str) if "es_mode" in df.columns else ""
    # hedge on/off + 강도
    hedge_on = hedge_flag(df)
    if "hedge_sigma_k" in df.columns:
        h = df["hedge_sigma_k"].astype(float).round(nd_h).astype(str)
    elif "hedge_ratio" in df.columns:
        h = df["hedge_ratio"].astype(float).round(nd_h).astype(str)
    else:
        h = pd.Series([""] * len(df), index=df.index)
    # (선택) data_profile
    if include_profile and "data_profile" in df.columns:
        prof = df["data_profile"].astype(str)
        return method + "|" + alpha + "|" + hedge_on + "|" + h + "|" + esm + "|" + prof
    return method + "|" + alpha + "|" + hedge_on + "|" + h + "|" + esm

# 로드
dev  = pd.read_csv(args.dev_src)
full = pd.read_csv(args.full_src)

# 사전 필터링(선택)
for df in (dev, full):
    if args.tag_startswith and "tag" in df.columns:
        df = df[df["tag"].astype(str).str.startswith(args.tag_startswith)]
    if args.method and "method" in df.columns:
        df = df[df["method"].astype(str).str.lower() == args.method.lower()]
    if args.es_mode and "es_mode" in df.columns:
        df = df[df["es_mode"].astype(str).str.lower() == args.es_mode.lower()]
    # 재할당
    if df is dev: dev = df
    else: full = df

# 키 생성
dev["key"]  = build_key(dev,  args.round_mix, args.round_h, args.include_profile_in_key=="on")
full["key"] = build_key(full, args.round_mix, args.round_h, args.include_profile_in_key=="on")

# 숫자화
num_cols_dev  = [c for c in ["EW","ES95","CompositeScore","Ruin"] if c in dev.columns]
num_cols_full = [c for c in ["EW","ES95","Ruin"]                if c in full.columns]
coerce_numeric(dev,  num_cols_dev)
coerce_numeric(full, num_cols_full)

# 집계
left = dev.groupby("key")[num_cols_dev].mean(numeric_only=True).add_prefix("dev_")
rght = full.groupby("key")[num_cols_full].mean(numeric_only=True).add_prefix("full_")
m = left.join(rght, how="inner")

# 드리프트
if "dev_ES95" in m.columns and "full_ES95" in m.columns:
    m["d_ES95_abs"] = m["full_ES95"] - m["dev_ES95"]
    m["d_ES95_rel"] = m["d_ES95_abs"] / m["dev_ES95"].replace(0, np.nan)
else:
    m["d_ES95_abs"] = np.nan
    m["d_ES95_rel"] = np.nan

# 상관
if "dev_CompositeScore" in m.columns and "full_ES95" in m.columns:
    rho = spearmanr(m["dev_CompositeScore"], m["full_ES95"], nan_policy="omit")[0]
else:
    rho = np.nan

# 플래그
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

out = p.Path(args.out)
with pd.ExcelWriter(out) as xw:
    m.reset_index().to_excel(excel_writer=xw, sheet_name="match", index=False)
    pd.DataFrame([summary_row]).to_excel(excel_writer=xw, sheet_name="stats", index=False)
print(f"[OK] {out}")
