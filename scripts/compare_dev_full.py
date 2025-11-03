# scripts/compare_dev_full.py
from __future__ import annotations

import argparse
import pathlib as p
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--dev_src",  default=r".\outputs\_summary_scored.csv",
                help="개발 프로파일(요약/스코어) CSV 또는 스냅샷 CSV 경로 (예: DEV_metrics_snapshot.csv)")
ap.add_argument("--full_src", default=r".\outputs\_logs\metrics.csv",
                help="풀 로그 메트릭 CSV")
ap.add_argument("--out",      default=r".\outputs\dev_full_drift.xlsx",
                help="결과 엑셀 경로")

# 필터 옵션
ap.add_argument("--tag_startswith", default="",
                help="이 접두사로 시작하는 tag만 비교 (미지정 시 전체). 스냅샷을 쓸 땐 DEV_/OVN_ 접두사와 함께 쓰면 좋아요.")
ap.add_argument("--method", default="",
                help="예: hjb / rl")
ap.add_argument("--es_mode", default="",
                help="예: wealth / cons")

# 허용 오차
ap.add_argument("--abs_tol",  type=float, default=0.05, help="ES95 절대 오차 허용치")
ap.add_argument("--rel_tol",  type=float, default=0.05, help="ES95 상대 오차 허용치(비율)")

# 키 구성 옵션
ap.add_argument("--round_mix", type=int, default=2, help="alpha_mix 각 성분 라운딩 자리수(키 생성용)")
ap.add_argument("--round_h",   type=int, default=2, help="hedge_sigma_k/ratio 라운딩 자리수(키 생성용)")
ap.add_argument("--include_profile_in_key", choices=["on","off"], default="on",
                help="키에 data_profile 포함 여부 (dev vs full 구분 필요하면 on 권장)")

args = ap.parse_args()


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────
def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    """지정 컬럼을 전부 numeric으로 강제 변환 (실패 시 NaN)."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _series_empty_like(df: pd.DataFrame) -> pd.Series:
    """df 길이에 맞춘 빈 문자열 Series (길이 불일치 버그 방지)."""
    return pd.Series([""] * len(df), index=df.index, dtype="object")


def _norm_alpha_mix_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    alpha_mix 문자열이 없으면 alpha_kr, alpha_us, alpha_au 조합으로 생성.
    모두 없으면 빈 문자열.
    """
    out = df.copy()
    if "alpha_mix" in out.columns and out["alpha_mix"].notna().any():
        out["alpha_mix_norm"] = out["alpha_mix"].astype(str)
        return out

    ks = ["alpha_kr", "alpha_us", "alpha_au"]
    if all(k in out.columns for k in ks):
        out["alpha_mix_norm"] = (
            out["alpha_kr"].astype(str) + "," +
            out["alpha_us"].astype(str) + "," +
            out["alpha_au"].astype(str)
        )
    else:
        out["alpha_mix_norm"] = _series_empty_like(out)
    return out


def _round_alpha_mix_str(s: str, n: int) -> str:
    try:
        a, b, c = [float(x) for x in str(s).split(",")]
        return f"{round(a, n)},{round(b, n)},{round(c, n)}"
    except Exception:
        # 알 수 없는 형식은 그대로
        return str(s)


def _pick_hedge_col(df: pd.DataFrame) -> str:
    if "hedge_sigma_k" in df.columns:
        return "hedge_sigma_k"
    if "hedge_ratio" in df.columns:
        return "hedge_ratio"
    return ""


def _norm_hedge_col(df: pd.DataFrame, n: int) -> pd.Series:
    """hedge 강도 컬럼 선택 → 라운딩 → 문자열화. 없으면 빈 문자열 시리즈."""
    col = _pick_hedge_col(df)
    if not col:
        return _series_empty_like(df)
    v = pd.to_numeric(df[col], errors="coerce").round(n)
    return v.astype(str)


def _norm_profile(df: pd.DataFrame) -> pd.Series:
    return df["data_profile"].astype(str) if "data_profile" in df.columns else _series_empty_like(df)


def _norm_method(df: pd.DataFrame) -> pd.Series:
    return df["method"].astype(str) if "method" in df.columns else _series_empty_like(df)


def _norm_es_mode(df: pd.DataFrame) -> pd.Series:
    return df["es_mode"].astype(str) if "es_mode" in df.columns else _series_empty_like(df)


def build_key(df: pd.DataFrame,
              round_mix: int,
              round_h: int,
              include_profile: bool) -> pd.Series:
    """
    비교 key 포맷:
      [data_profile|] method | alpha_mix(라운드) | hedge_strength(라운드) | es_mode
    - data_profile 포함 여부는 include_profile 플래그로 제어
    - hedge on/off 플래그는 키에서 제외(강도 값으로 충분히 식별 가능하다고 가정)
    """
    tmp = _norm_alpha_mix_fields(df)
    method = _norm_method(tmp)
    alpha  = tmp["alpha_mix_norm"].astype(str).map(lambda s: _round_alpha_mix_str(s, round_mix))
    hedge  = _norm_hedge_col(tmp, round_h)
    esmd   = _norm_es_mode(tmp)
    base   = method + "|" + alpha + "|" + hedge + "|" + esmd
    if include_profile:
        profile = _norm_profile(tmp)
        return profile + "|" + base
    return base


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """tag/method/es_mode 필터 적용. 해당 컬럼이 없으면 안전하게 skip."""
    m = df.copy()

    if args.tag_startswith:
        if "tag" in m.columns:
            m = m[m["tag"].astype(str).str.startswith(args.tag_startswith)]
        else:
            # tag가 없으면 결과를 비우는 편이 논리적
            m = m.iloc[0:0]

    if args.method:
        if "method" in m.columns:
            m = m[m["method"].astype(str).str.lower() == args.method.lower()]
        else:
            m = m.iloc[0:0]

    if args.es_mode:
        if "es_mode" in m.columns:
            m = m[m["es_mode"].astype(str).str.lower() == args.es_mode.lower()]
        else:
            m = m.iloc[0:0]

    return m


# ─────────────────────────────────────────────────────────
# Load & preprocess
# ─────────────────────────────────────────────────────────
dev  = pd.read_csv(args.dev_src)
full = pd.read_csv(args.full_src)

dev  = _apply_filters(dev)
full = _apply_filters(full)

# key 생성
inc_prof = (args.include_profile_in_key == "on")
dev["key"]  = build_key(dev,  args.round_mix, args.round_h, inc_prof)
full["key"] = build_key(full, args.round_mix, args.round_h, inc_prof)

# 숫자화
num_cols_dev  = [c for c in ["EW", "ES95", "CompositeScore", "Ruin"] if c in dev.columns]
num_cols_full = [c for c in ["EW", "ES95", "Ruin"] if c in full.columns]
coerce_numeric(dev,  num_cols_dev)
coerce_numeric(full, num_cols_full)

# 집계
left = dev.groupby("key", dropna=False)[num_cols_dev].mean(numeric_only=True).add_prefix("dev_")
rght = full.groupby("key", dropna=False)[num_cols_full].mean(numeric_only=True).add_prefix("full_")
m = left.join(rght, how="inner")

# 드리프트 계산
if "dev_ES95" in m.columns and "full_ES95" in m.columns:
    m["d_ES95_abs"] = m["full_ES95"] - m["dev_ES95"]
    m["d_ES95_rel"] = m["d_ES95_abs"] / m["dev_ES95"].replace(0, np.nan)
else:
    m["d_ES95_abs"] = np.nan
    m["d_ES95_rel"] = np.nan

# 상관(선택)
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
print(
    f"[DRIFT] pairs={summary_row['pairs']} "
    f"abs_fail={summary_row['n_abs_fail']} "
    f"rel_fail={summary_row['n_rel_fail']} -> {'OK' if overall_ok else 'CHECK'}"
)

# ─────────────────────────────────────────────────────────
# Write
# ─────────────────────────────────────────────────────────
out = p.Path(args.out)
with pd.ExcelWriter(out) as xw:
    m.reset_index().to_excel(excel_writer=xw, sheet_name="match", index=False)
    pd.DataFrame([summary_row]).to_excel(excel_writer=xw, sheet_name="stats", index=False)

print(f"[OK] {out}")
