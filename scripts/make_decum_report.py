# scripts/make_decum_report.py  (DECUM 전용, 안전판)
import argparse
import os
import pathlib
from datetime import datetime

import pandas as pd


def load_metrics():
    """
    우선순위: dev_metrics_snapshot.csv -> _summary_scored.csv -> _logs/metrics.csv
    """
    for p in (
        r".\outputs\dev_metrics_snapshot.csv",
        r".\outputs\_summary_scored.csv",
        r".\outputs\_logs\metrics.csv",
    ):
        if os.path.exists(p):
            return pd.read_csv(p), p
    raise FileNotFoundError(
        "metrics source not found (dev_metrics_snapshot.csv / _summary_scored.csv / _logs/metrics.csv)"
    )


def to_num(s: pd.Series) -> pd.Series:
    # 숫자 강제 변환, 실패 시 NaN
    return pd.to_numeric(s, errors="coerce")


def ensure_method(df: pd.DataFrame) -> pd.DataFrame:
    """
    method 컬럼 보정: 없거나 결측이면 tag 휴리스틱으로 추론
    """
    if "method" not in df.columns:
        df["method"] = pd.NA

    def infer_from_tag(tag: str):
        if not isinstance(tag, str):
            return pd.NA
        t = tag.lower()
        if "_rl_" in t or "mini" in t or t.startswith("fullchk_p3_rl"):
            return "rl"
        if t.startswith("rob_") or "hjb" in t:
            return "hjb"
        if "rule" in t or "4pct" in t or "vpw" in t or "cpb" in t:
            return "rule"
        return pd.NA

    if "tag" in df.columns:
        miss = df["method"].isna() | (df["method"] == "")
        df.loc[miss, "method"] = df.loc[miss, "tag"].map(infer_from_tag)

    # 안전장치: 전부 결측이면 임의 라벨 부여 방지 → 그대로 둔다.
    return df


def ensure_numeric_and_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    주요 수치형 컬럼 숫자화 + CompositeScore 없거나 전부 결측이면 z-score 기반 임시 점수 생성
    (방향성: EW↑, ES95↑, Ruin↓)
    """
    for c in ("EW", "ES95", "Ruin", "CompositeScore"):
        if c in df.columns:
            df[c] = to_num(df[c])

    need_score = ("CompositeScore" not in df.columns) or df["CompositeScore"].isna().all()

    if need_score:
        # 누락 컬럼은 NaN 채움
        for c in ("EW", "ES95", "Ruin"):
            if c not in df.columns:
                df[c] = pd.NA

        def z(x: pd.Series) -> pd.Series:
            x = to_num(x)
            m = x.mean(skipna=True)
            s = x.std(skipna=True)
            if pd.isna(m) or pd.isna(s) or s == 0:
                return pd.Series(0.0, index=x.index)
            return (x - m) / s

        z_ew = z(df["EW"]) if "EW" in df.columns else 0.0
        z_es = z(df["ES95"]) if "ES95" in df.columns else 0.0
        z_ru = -z(df["Ruin"]) if "Ruin" in df.columns else 0.0  # 낮을수록 좋음 → 부호 반전

        # 가중합(논문 기본 가중치)
        w_ew, w_es, w_ru = 0.4, 0.4, 0.2
        df["CompositeScore"] = w_ew * z_ew + w_es * z_es + w_ru * z_ru

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="compare")
    ap.add_argument("--outdir", default=r".\outputs")
    # (옵션) winners 상위 N 스냅샷 크기
    ap.add_argument("--frontier_topn", type=int, default=50)
    args = ap.parse_args()

    df, src = load_metrics()
    df = ensure_method(df.copy())
    df = ensure_numeric_and_score(df)

    # dedup: (tag, seed, method) 존재 시 최근 기록 유지
    keys = [c for c in ("tag", "seed", "method") if c in df.columns]
    winners = df.copy()
    if keys:
        winners = winners.drop_duplicates(subset=keys, keep="last")

    # by method 집계
    if "method" in winners.columns:
        agg_spec = {}
        if "EW" in winners.columns:
            agg_spec["EW_mean"] = ("EW", "mean")
        if "ES95" in winners.columns:
            agg_spec["ES95_mean"] = ("ES95", "mean")
        if "Ruin" in winners.columns:
            agg_spec["Ruin_mean"] = ("Ruin", "mean")
        if "CompositeScore" in winners.columns:
            agg_spec["CompScore_mean"] = ("CompositeScore", "mean")
        by_method = (
            winners.groupby("method", dropna=False).agg(**agg_spec).reset_index()
            if agg_spec
            else pd.DataFrame({"method": winners["method"].unique()})
        )
    else:
        by_method = pd.DataFrame()

    # 간이 프런티어 스냅샷
    frontier = pd.DataFrame()
    need_cols = {"EW", "ES95"}
    if need_cols.issubset(winners.columns):
        cols = [c for c in ("tag", "method", "seed", "EW", "ES95", "CompositeScore") if c in winners.columns]
        frontier = winners[cols].copy()
        if "CompositeScore" not in frontier.columns:
            frontier["CompositeScore"] = pd.NA
        frontier = frontier.sort_values(["ES95", "EW"], ascending=[False, False]).head(args.frontier_topn)

    # 출력
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"Paper_Decum_Report_{args.tag}.xlsx"

    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        winners.to_excel(excel_writer=xw, sheet_name="winners_raw", index=False)
        by_method.to_excel(excel_writer=xw, sheet_name="summary_by_method", index=False)
        frontier.to_excel(excel_writer=xw, sheet_name="frontier_snapshot", index=False)

        meta = pd.DataFrame(
            {
                "field": ["generated_at", "metrics_source", "rows", "distinct_tags", "have_score"],
                "value": [
                    datetime.now().isoformat(timespec="seconds"),
                    src,
                    len(df),
                    (df["tag"].nunique() if "tag" in df.columns else None),
                    (not winners["CompositeScore"].isna().all() if "CompositeScore" in winners.columns else False),
                ],
            }
        )
        meta.to_excel(excel_writer=xw, sheet_name="params", index=False)

    print(f"[OK] Report written: {out}")


if __name__ == "__main__":
    main()
