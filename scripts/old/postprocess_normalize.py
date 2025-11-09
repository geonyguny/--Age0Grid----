# scripts/postprocess_normalize.py
import argparse, os, re
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

DEFAULT_KEYS = ["window","hedge_ratio","mix_kr","mix_us","mix_gold","es_mode"]
NUM_COLS_HINT = [
    "wealth","EW","ES","ES95","Ruin","RuinPct","CompositeScore",
    "EU_per_year","AlivePathRate","HedgeHit"
]

# ─────────────────────────────────────────────────────────
# A) 기본 유틸
# ─────────────────────────────────────────────────────────
def ann_factor(freq: str) -> int:
    return {"M":12, "Q":4, "Y":1}.get(freq.upper(), 12)

def to_real_if_needed(df: pd.DataFrame, cpi_col: Optional[str], price_cols: List[str]) -> pd.DataFrame:
    if cpi_col and cpi_col in df.columns:
        base = df[cpi_col].iloc[0]
        for c in price_cols:
            if c in df.columns:
                df[c] = df[c] / (df[cpi_col] / base)
    return df

def sanitize_numeric(s: pd.Series) -> pd.Series:
    # 문자열 내 %, 콤마, 공백 제거 → 숫자화. "NaN"/"" → NaN
    if s.dtype.kind in "biufc":
        return s.astype(float)
    s = s.astype(str).str.strip()
    s = s.replace({"": np.nan, "None": np.nan, "NaN": np.nan, "nan": np.nan})
    s = s.str.replace("%", "", regex=False).str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

# ── const 처리 포함 정규화 (상수열 → 중립값 대체 옵션) ──
def zscore_safe(x: np.ndarray, const_val: Optional[float]) -> np.ndarray:
    valid = np.isfinite(x); n = valid.sum()
    if n < 2:
        return np.full_like(x, np.nan, dtype=float) if const_val is None else np.full_like(x, const_val, dtype=float)
    xv = x[valid]
    with np.errstate(all='ignore'):
        m = np.nanmean(xv); s = np.nanstd(xv, ddof=1)
    if not np.isfinite(s) or s == 0:
        return np.full_like(x, np.nan, dtype=float) if const_val is None else np.full_like(x, const_val, dtype=float)
    out = (x - m) / s
    out[~valid] = np.nan
    return out

def minmax_safe(x: np.ndarray, const_val: Optional[float]) -> np.ndarray:
    valid = np.isfinite(x); n = valid.sum()
    if n < 2:
        return np.full_like(x, np.nan, dtype=float) if const_val is None else np.full_like(x, const_val, dtype=float)
    xv = x[valid]
    with np.errstate(all='ignore'):
        a = np.nanmin(xv); b = np.nanmax(xv)
    if not (np.isfinite(a) and np.isfinite(b)) or b <= a:
        return np.full_like(x, np.nan, dtype=float) if const_val is None else np.full_like(x, const_val, dtype=float)
    out = (x - a) / (b - a)
    out[~valid] = np.nan
    return out

def parse_const_map(s: Optional[str]) -> Dict[str, float]:
    m: Dict[str, float] = {}
    if not s:
        return m
    for tok in s.split(","):
        tok = tok.strip()
        if not tok or ":" not in tok:
            continue
        k, v = tok.split(":", 1)
        try:
            m[k.strip()] = float(v.strip())
        except:
            pass
    return m

# ─────────────────────────────────────────────────────────
# B) tag → 키 보정 유틸
# ─────────────────────────────────────────────────────────
TAG_PAT_WINDOW = re.compile(r"_w(FULL|\d{4}-\d{2}to)")
TAG_PAT_HEDGE  = re.compile(r"_h([0-9]+(?:\.[0-9]+)?)")
TAG_PAT_MIX    = re.compile(r"_m([0-9\.]+)-([0-9\.]+)-([0-9\.]+)")

def backfill_keys_from_tag(df: pd.DataFrame) -> pd.DataFrame:
    if "tag" not in df.columns:
        return df

    def parse_row(row: pd.Series) -> pd.Series:
        tag = str(row.get("tag", "") or "")

        # window
        if pd.isna(row.get("window")) or row.get("window")=="":
            m = TAG_PAT_WINDOW.search(tag)
            if m: row["window"] = m.group(1)

        # hedge_ratio
        if pd.isna(row.get("hedge_ratio")) or row.get("hedge_ratio")=="":
            m = TAG_PAT_HEDGE.search(tag)
            if m:
                try: row["hedge_ratio"] = float(m.group(1))
                except: pass

        # mix
        need_mix = any(pd.isna(row.get(k)) or row.get(k)=="" for k in ["mix_kr","mix_us","mix_gold"])
        if need_mix:
            m = TAG_PAT_MIX.search(tag)
            if m:
                for k, val in zip(["mix_kr","mix_us","mix_gold"], m.groups()):
                    try: row[k] = float(val)
                    except: pass
        return row

    return df.apply(parse_row, axis=1)

def infer_es_mode_if_empty(df: pd.DataFrame) -> pd.DataFrame:
    if "es_mode" not in df.columns:
        df["es_mode"] = np.nan

    def _infer(m) -> float:
        if m is None or (isinstance(m, float) and np.isnan(m)): return np.nan
        ms = str(m).lower()
        # 0=wealth, 1=cons, 2=loss (summarize_outputs.ps1 규칙과 합치)
        if ms == "wealth": return 0.0
        if ms == "cons":   return 1.0
        if ms == "loss":   return 2.0
        # 숫자로 이미 들어온 경우
        try:
            return float(ms)
        except:
            return np.nan

    mask_empty = df["es_mode"].isna() | (df["es_mode"]=="")
    if "es_metric" in df.columns:
        df.loc[mask_empty, "es_mode"] = df.loc[mask_empty, "es_metric"].map(_infer)
    return df

def is_all_group_keys_empty(row: pd.Series, key_cols: List[str]) -> bool:
    for k in key_cols:
        v = row.get(k)
        if v is not None and not (isinstance(v, float) and np.isnan(v)) and str(v) != "":
            return False
    return True

# ─────────────────────────────────────────────────────────
# C) 메인
# ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--freq", default="M", help="M/Q/Y (연환산용)")
    ap.add_argument("--real", choices=["on","off"], default="off")
    ap.add_argument("--cpi_col", default=None)

    ap.add_argument("--es_col", default=None, help="ES 열명 강제 지정 (예: ES95)")
    ap.add_argument("--cols", default=None, help="정규화 대상 컬럼(쉼표 구분)")
    ap.add_argument("--group_keys", default=",".join(DEFAULT_KEYS),
                    help="그룹키(쉼표 구분). 기본: window,hedge_ratio,mix_kr,mix_us,mix_gold,es_mode")

    # ▶ 상수열 대체 옵션
    ap.add_argument("--const_as", type=float, default=None,
                    help="그룹 내 상수열(z/mm 불가)일 때 대체할 값. 예: 0.5. 미지정 시 NaN 유지")
    ap.add_argument("--const_map", default=None,
                    help='지표별 상수 대체값 매핑. 예: "Ruin:0.5,ES95:0.4" (const_as보다 우선)')

    # ▶ 자동 재가중 옵션 (Composite_rescored 생성)
    ap.add_argument("--rescore_cols", default=None,
                    help='재가중 대상(Composite) 컬럼 셋. 예: "ES95,EW,Ruin"')
    ap.add_argument("--weights", default=None,
                    help='재가중 가중치. 예: "0.6,0.3,0.1" (rescore_cols와 길이 일치)')
    ap.add_argument("--rescore_variant", choices=["mm","z"], default="mm",
                    help="재가중에 사용할 정규화 값 선택(mm or z). 기본 mm")

    # ▶ 6키 전부 공백인 행 드롭 스위치
    ap.add_argument("--drop_all_empty_keys", choices=["on","off"], default="on",
                    help="window/hedge/mix/us/gold/es_mode 모두 공백인 행을 드롭")

    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    # 연환산/보조 파생
    f = ann_factor(args.freq)
    for col in list(df.columns):
        low = col.lower()
        if low in ["ew","return","ret","mu","sharpe"]:
            df[col+"_ann"] = sanitize_numeric(df[col]) * f
        if "ruin" in low and not low.endswith("pct"):
            df[col+"_pct"] = sanitize_numeric(df[col]) * 100.0

    if args.real == "on":
        df = to_real_if_needed(df, args.cpi_col, price_cols=["wealth","EW","return"])

    # 정규화 대상 컬럼 결정
    if args.cols:
        cand = [c.strip() for c in args.cols.split(",") if c.strip() in df.columns]
    else:
        cand = [c for c in df.columns if any(h.lower() in c.lower() for h in NUM_COLS_HINT)]
        if args.es_col and args.es_col in df.columns:
            cand.append(args.es_col)
        cand = sorted(set(cand))

    # 숫자화(%, 콤마 제거 포함)
    for c in cand:
        if c in df.columns:
            df[c] = sanitize_numeric(df[c])

    # ▶ 키 백필 + es_mode 보정
    df = backfill_keys_from_tag(df)
    df = infer_es_mode_if_empty(df)

    # ▶ 6키가 전부 공백인 행 드롭(옵션)
    key_cols = ["window","hedge_ratio","mix_kr","mix_us","mix_gold","es_mode"]
    if args.drop_all_empty_keys == "on":
        mask_bad = df.apply(lambda r: is_all_group_keys_empty(r, key_cols), axis=1)
        bad_cnt = int(mask_bad.sum())
        if bad_cnt > 0:
            print(f"[WARN] drop rows with all-empty group keys: {bad_cnt}")
            df = df.loc[~mask_bad].copy()

    # 전부 NaN인 컬럼은 제외
    cand = [c for c in cand if c in df.columns and df[c].notna().any()]

    # 그룹키 교집합만 사용
    requested_keys = [s.strip() for s in args.group_keys.split(",") if s.strip()]
    keys: List[str] = [k for k in requested_keys if k in df.columns]

    CONST_MAP = parse_const_map(args.const_map)
    CONST_DEFAULT = args.const_as

    # 그룹 내 정규화 함수
    def add_norm(g: pd.DataFrame) -> pd.DataFrame:
        out = g.copy()
        for c in cand:
            x = out[c].to_numpy(dtype=float)
            const_val = CONST_MAP.get(c, CONST_DEFAULT)
            out[c+"_z"]  = zscore_safe(x, const_val)
            out[c+"_mm"] = minmax_safe(x, const_val)
        return out

    # ── deprecation/미래 호환: groupby.apply 경고 회피 + 키 보존 보장 ──
    if keys:
        groups_count = df.drop_duplicates(subset=keys).shape[0]
        parts: List[pd.DataFrame] = []
        # dropna=False 유지 → NaN 키도 독립 그룹
        for key_vals, g in df.groupby(keys, dropna=False, group_keys=False):
            out = add_norm(g)
            # 키 재부착(향후 include_groups 변경 대비)
            if not isinstance(key_vals, tuple):
                key_vals = (key_vals,)
            for k, v in zip(keys, key_vals):
                if k not in out.columns:
                    out[k] = v
            parts.append(out)
        df = pd.concat(parts, ignore_index=True)
    else:
        df = add_norm(df)
        groups_count = 1

    # ── (선택) Composite 재가중 ──
    def rescore_if_requested(ddf: pd.DataFrame) -> pd.DataFrame:
        if not args.rescore_cols or not args.weights:
            return ddf
        cols = [c.strip() for c in args.rescore_cols.split(",") if c.strip()]
        wtxt = [x for x in args.weights.split(",") if x.strip()]
        try:
            W = np.array([float(x) for x in wtxt], dtype=float)
        except:
            print("[WARN] rescore: weights parse 실패 → skip")
            return ddf
        if len(cols) != len(W):
            print("[WARN] rescore: 길이 불일치 → skip")
            return ddf

        norm_suffix = "_mm" if args.rescore_variant == "mm" else "_z"
        out = ddf.copy()
        scores = []
        for _, row in out.iterrows():
            use_vals = []
            use_w = []
            for i, col in enumerate(cols):
                ncol = f"{col}{norm_suffix}"
                v = row.get(ncol, np.nan)
                if pd.notna(v):
                    use_vals.append(float(v))
                    use_w.append(W[i])
            if len(use_vals) == 0:
                scores.append(np.nan)
                continue
            use_w = np.array(use_w, dtype=float)
            use_w = use_w / use_w.sum()
            scores.append(float(np.dot(use_w, np.array(use_vals, dtype=float))))
        out["Composite_rescored"] = scores
        return out

    df = rescore_if_requested(df)

    # 저장
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8")

    # 로그
    skipped_cols = [c for c in cand if not pd.Series(df.get(c+"_z")).notna().any()]
    print(f"[OK] saved -> {args.out_csv} (rows={len(df)}, groups={groups_count})")
    print(f"[INFO] normalized cols: {', '.join(cand) if cand else '(none)'}")
    if skipped_cols:
        print(f"[WARN] all-NaN after norm (likely low support or non-numeric source): {', '.join(skipped_cols)}")
    if "Composite_rescored" in df.columns:
        print(f"[INFO] Composite_rescored created (variant={args.rescore_variant}, missing/constant auto-handled)")

if __name__ == "__main__":
    main()
