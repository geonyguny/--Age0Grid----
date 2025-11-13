# scripts/score_snapshot.py
from __future__ import annotations
import argparse, os, sys, math, re
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np

EPS = 1e-12

DEF_METRICS = "EW,ES95"          # 점수 계산에 사용할 원천 지표
DEF_WEIGHTS = "0.6,0.4"          # 각 지표 가중치(합=1 권장)
# 낮을수록 좋은 지표(위험/손실 계열). 여기 있으면 정규화 후 '반대로' 매핑(=낮을수록 1에 가깝게)
LOWER_IS_BETTER_DEFAULT = ["ES95", "Ruin", "RuinPct", "VaR", "DD", "MDD"]

# 예: DEV2D_wrisk_hedge_BASE_M_w0.5_h0.20, DEV1D_ann_BASE_M_0.0
_PAT_ID = re.compile(r"(?i)[_\-](?P<mort>BASE|COHORT)(?:[_\-])(?P<sex>M|F)(?:[_\-]|$)")
# 예: ..._2D_us0.6_h0.25  (기존 좌표 보강)
_TAG_PAT_US_H = re.compile(r".*?_2D_us(?P<u>[0-9.]+)_h(?P<h>[0-9.]+)")

# ─────────────────────────────────────────────────────────
# 파서 & 유틸
# ─────────────────────────────────────────────────────────
def parse_list(s: str) -> List[str]:
    return [t.strip() for t in str(s).split(",") if t is not None and str(t).strip() != ""]

def parse_float_list(s: str) -> List[float]:
    return [float(t.strip()) for t in str(s).split(",") if t is not None and str(t).strip() != ""]

def safe_minmax(x: pd.Series) -> Tuple[float, float]:
    x = pd.to_numeric(x, errors="coerce")
    x = x[np.isfinite(x)]
    if x.empty:
        return (np.nan, np.nan)
    return float(x.min()), float(x.max())

def minmax_scale(x: pd.Series, higher_is_better: bool) -> pd.Series:
    vmin, vmax = safe_minmax(x)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < EPS:
        # 상수열이면 0.5로 부여(정보 없음)
        out = pd.Series(np.full(len(x), 0.5), index=x.index, dtype=float)
    else:
        scaled = (x - vmin) / max(vmax - vmin, EPS)
        out = scaled.astype(float)
    # 방향 반전(낮을수록 좋음 → 1에 가깝게)
    if not higher_is_better:
        out = 1.0 - out
    # NaN 보정
    return out.where(np.isfinite(out), np.nan)

def compute_composite(
    df: pd.DataFrame,
    metrics: List[str],
    weights: List[float],
    lower_is_better: List[str],
    prefix_norm: str = "norm_",
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    assert len(metrics) == len(weights), "metrics와 weights의 길이가 일치해야 합니다."
    df = df.copy()

    # 가중치 정규화(합=1)
    w = np.asarray(weights, dtype=float)
    if not np.isfinite(w).all() or w.sum() <= 0:
        raise SystemExit("[ERR] weights에 유효하지 않은 값이 있습니다.")
    w = w / w.sum()

    # 정규화 및 점수
    norms: Dict[str, pd.Series] = {}
    ranges: Dict[str, Tuple[float, float]] = {}
    for i, m in enumerate(metrics):
        if m not in df.columns:
            raise SystemExit(f"[ERR] '{m}' 컬럼이 없습니다.")
        higher_is_better = (m not in lower_is_better)
        vmin, vmax = safe_minmax(df[m])
        ranges[m] = (vmin, vmax)
        norms[m] = minmax_scale(df[m], higher_is_better)
        df[f"{prefix_norm}{m}"] = norms[m]

    # CompositeScore = Σ w_i * norm_i
    comp = np.zeros(len(df), dtype=float)
    for i, m in enumerate(metrics):
        comp += w[i] * df[f"{prefix_norm}{m}"].astype(float).fillna(0.5).to_numpy()
    df["CompositeScore"] = comp
    return df, ranges

# ─────────────────────────────────────────────────────────
# 스키마 표준화 훅
# ─────────────────────────────────────────────────────────
def _infer_xy_from_tag(df: pd.DataFrame) -> pd.DataFrame:
    """tag에서 us, h 파싱 → mix_us, hedge_sigma_k 채우기(없을 때만)."""
    if "tag" not in df.columns:
        return df
    def _one(tag: str):
        m = _TAG_PAT_US_H.match(str(tag))
        if not m:
            return {}
        try:
            return {
                "mix_us": float(m.group("u")),
                "hedge_sigma_k": float(m.group("h")),
            }
        except Exception:
            return {}
    extra = pd.DataFrame(list(df["tag"].map(_one)))
    for c in ("mix_us", "hedge_sigma_k"):
        if c not in df.columns and c in extra.columns:
            df[c] = extra[c]
    return df

def _infer_identity_from_tag(df: pd.DataFrame) -> pd.DataFrame:
    """tag에서 mort_id/sex 추론 → mort_id, sex, mort_table 보강(없을 때만)."""
    if "tag" not in df.columns:
        # 최소 열 생성
        for c in ("sex", "mort_id", "mort_table"):
            if c not in df.columns:
                df[c] = ""
        return df

    def _one(tag: str):
        m = _PAT_ID.search(str(tag))
        if not m:
            return {}
        mort = m.group("mort").upper()
        sex  = m.group("sex").upper()
        mort_table = "cohort_2020" if mort == "COHORT" else "base"
        return {"mort_id": mort, "sex": sex, "mort_table": mort_table}

    extra = pd.DataFrame(list(df["tag"].map(_one)))
    for c in ("sex", "mort_id", "mort_table"):
        if c not in df.columns and c in extra.columns:
            df[c] = extra[c]

    # 타입/정규화
    if "sex" in df.columns:
        df["sex"] = df["sex"].astype(str).str.upper()
        df.loc[~df["sex"].isin(["M","F"]), "sex"] = ""
    if "mort_id" in df.columns:
        df["mort_id"] = df["mort_id"].astype(str).str.upper()
        df.loc[~df["mort_id"].isin(["BASE","COHORT"]), "mort_id"] = ""
    if "mort_table" in df.columns:
        df["mort_table"] = df["mort_table"].astype(str).str.lower()
        # mort_id와 일치하도록 보정(정보 상충 시 tag기반 mort_id 우선)
        mask = df["mort_table"].isna() | (df["mort_table"].eq(""))
        if "mort_id" in df.columns:
            df.loc[mask & df["mort_id"].eq("BASE"), "mort_table"] = "base"
            df.loc[mask & df["mort_id"].eq("COHORT"), "mort_table"] = "cohort_2020"

    # 누락 열 보장
    for c in ("sex", "mort_id", "mort_table"):
        if c not in df.columns:
            df[c] = ""
    return df

def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """수치 캐스팅 + 필수 컬럼 보강 + tag로부터 좌표/신원 추론."""
    d = df.copy()
    d = _infer_xy_from_tag(d)
    d = _infer_identity_from_tag(d)

    # 기본 수치형 캐스팅
    for c in ("EW", "ES95", "Ruin", "RuinPct", "mix_us", "hedge_sigma_k", "CompositeScore"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # Ruin 정규화(있다면 0~1 확률로 정리)
    if "Ruin" in d.columns:
        over_one = d["Ruin"] > 1.0
        d.loc[over_one, "Ruin"] = d.loc[over_one, "Ruin"] / 100.0
    elif "RuinPct" in d.columns and "Ruin" not in d.columns:
        d["Ruin"] = pd.to_numeric(d["RuinPct"], errors="coerce") / 100.0

    # 필수 텍스트/메타 컬럼 기본값
    for c in ("method", "es_mode", "seed"):
        if c not in d.columns:
            d[c] = "" if c != "seed" else np.nan

    # 분석 편의를 위한 메타 열(없으면 생성만)
    for c in ("market_mode", "use_real_rf"):
        if c not in d.columns:
            d[c] = ""

    return d

# ─────────────────────────────────────────────────────────
# 필터
# ─────────────────────────────────────────────────────────
def filter_scope(df: pd.DataFrame, tag_startswith: str, method: str, es_mode: str) -> pd.Index:
    if "tag" not in df.columns:
        raise SystemExit("[ERR] snapshot에 'tag' 컬럼이 없습니다.")

    m = pd.Series([True] * len(df), index=df.index)

    if tag_startswith:
        m &= df["tag"].astype(str).str.startswith(tag_startswith)

    if method:
        if "method" in df.columns:
            m &= df["method"].astype(str).str.lower().eq(method.lower())
        else:
            # 컬럼이 없으면 전부 False
            m &= False

    if es_mode:
        if "es_mode" in df.columns:
            m &= df["es_mode"].astype(str).str.lower().eq(es_mode.lower())
        else:
            m &= False

    idx = df.index[m]
    if len(idx) == 0:
        raise SystemExit(f"[ERR] 필터 결과가 없습니다. tag_startswith={tag_startswith}, method={method}, es_mode={es_mode}")
    return idx

# ─────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Add CompositeScore to snapshot CSV")
    ap.add_argument("--src", required=True, help="입력 스냅샷 CSV (예: .\\outputs\\DEV_metrics_snapshot.csv)")
    ap.add_argument("--out", default="inplace",
                    help="출력 경로. 'inplace'면 원본 덮어쓰기, 비우면 *_scored.csv로 저장")
    ap.add_argument("--tag_startswith", default="", help="예: DEV_2D_, OVN_2D_, DEV_OAT_ …")
    ap.add_argument("--method", default="", help="필터: hjb/rl 등")
    ap.add_argument("--es_mode", default="", help="필터: wealth/cons 등")
    ap.add_argument("--metrics", default=DEF_METRICS, help=f"기본 '{DEF_METRICS}'")
    ap.add_argument("--weights", default=DEF_WEIGHTS, help=f"기본 '{DEF_WEIGHTS}' (합=1 권장)")
    ap.add_argument("--lower_is_better", default=",".join(LOWER_IS_BETTER_DEFAULT),
                    help="낮을수록 좋은 지표 리스트(콤마 구분)")
    ap.add_argument("--round", type=int, default=4, help="CompositeScore 소수 반올림 자리수")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    if not os.path.exists(src):
        raise SystemExit(f"[ERR] 파일이 없습니다: {src}")

    df = pd.read_csv(src)

    # 스키마/신원 필드 보강(먼저 실행해 열을 확보)
    df = _ensure_schema(df)

    # 계산 대상 인덱스만 분리(나머지는 그대로 유지)
    idx = filter_scope(df, args.tag_startswith, args.method, args.es_mode)

    metrics = parse_list(args.metrics)
    weights = parse_float_list(args.weights)
    lower = [c.strip() for c in parse_list(args.lower_is_better)]

    # 숫자화
    for c in metrics:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 부분 DataFrame에 대해 계산
    part = df.loc[idx, :].copy()
    scored, ranges = compute_composite(part, metrics, weights, lower, prefix_norm="norm_")

    # 원본에 병합(선택 영역만 갱신)
    for c in [f"norm_{m}" for m in metrics] + ["CompositeScore"]:
        df.loc[idx, c] = scored[c]

    if args.round is not None and "CompositeScore" in df.columns:
        df["CompositeScore"] = pd.to_numeric(df["CompositeScore"], errors="coerce").round(args.round)

    # 저장 경로
    if args.out == "inplace":
        out_path = src
    else:
        out_path = args.out.strip() or os.path.splitext(src)[0] + "_scored.csv"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 로그
    print(f"[OK] saved: {out_path}")
    print(f"[INFO] filtered rows: {len(idx)} / total: {len(df)}")
    for m in metrics:
        r = ranges.get(m, (np.nan, np.nan))
        print(f"[RANGE] {m}: min={r[0]:.6g}, max={r[1]:.6g}")
    cols_preview = ["tag"] + [f"norm_{m}" for m in metrics] + ["CompositeScore"]
    cols_preview = [c for c in cols_preview if c in df.columns]
    print("[HEAD]")
    print(df.loc[idx, cols_preview].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
