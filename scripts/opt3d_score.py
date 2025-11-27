from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List, Tuple, Optional


def _parse_list(s: str) -> List[str]:
    return [t.strip() for t in str(s).split(",") if t is not None and str(t).strip() != ""]


def _parse_float_list(s: str) -> List[float]:
    out: List[float] = []
    for t in _parse_list(s):
        try:
            out.append(float(t))
        except Exception:
            raise ValueError(f"잘못된 숫자: {t!r}")
    return out


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _load_csv(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
        fields = list(reader.fieldnames or [])
    return rows, fields


def _normalize_metrics(
    rows: List[Dict[str, Any]],
    metrics: List[str],
    lower_is_better: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Tuple[float, float]]]:
    ranges: Dict[str, Tuple[float, float]] = {}
    # 1) 각 metric별 min / max 계산
    for m in metrics:
        vals: List[float] = []
        for r in rows:
            v = _safe_float(r.get(m))
            if v is not None:
                vals.append(v)
        if not vals:
            ranges[m] = (math.nan, math.nan)
            continue
        v_min, v_max = min(vals), max(vals)
        ranges[m] = (v_min, v_max)

    # 2) 각 row에 정규화 값 추가
    for r in rows:
        for m in metrics:
            col = f"norm_{m}"
            raw = _safe_float(r.get(m))
            v_min, v_max = ranges.get(m, (math.nan, math.nan))
            if raw is None or not math.isfinite(v_min) or not math.isfinite(v_max):
                r[col] = ""
                continue
            if v_max == v_min:
                # 모든 값이 동일한 경우 중간값(0.5)로
                norm = 0.5
            else:
                norm = (raw - v_min) / (v_max - v_min)
            # 낮을수록 좋은 지표는 방향 반전
            if m in lower_is_better:
                norm = 1.0 - norm
            r[col] = norm
    return rows, ranges


def _compute_composite(
    rows: List[Dict[str, Any]],
    metrics: List[str],
    weights: List[float],
) -> None:
    if len(metrics) != len(weights):
        raise ValueError("metrics와 weights 길이가 다릅니다.")
    ws_sum = sum(weights)
    if ws_sum <= 0:
        raise ValueError("weights 합이 0보다 커야 합니다.")
    ws = [w / ws_sum for w in weights]

    for r in rows:
        score = 0.0
        has_any = False
        for m, w in zip(metrics, ws):
            col = f"norm_{m}"
            v = r.get(col, "")
            if isinstance(v, str):
                try:
                    v = float(v)
                except Exception:
                    continue
            try:
                fv = float(v)
            except Exception:
                continue
            if not math.isfinite(fv):
                continue
            score += w * fv
            has_any = True
        r["CompositeScore"] = score if has_any else ""
    # rank는 나중에 정렬 후 부여
    return None


def _save_csv(path: str, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    # 기존 컬럼 + 새로 생긴 컬럼 병합
    extra_cols: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields and k not in extra_cols:
                extra_cols.append(k)
    all_fields = fields + extra_cols

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="OPT3D 요약 CSV(예: OPT3D_DEV_summary.csv)에 대해 정규화/CompositeScore를 계산하는 유틸"
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        required=True,
        help="opt3d_theory에서 생성된 summary CSV 경로 (예: outputs/OPT3D_DEV/OPT3D_DEV_summary.csv)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="EW,ES95,cons_coverage_mean",
        help="점수 계산에 사용할 지표 목록 (쉼표 구분)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="0.4,0.3,0.3",
        help="각 지표 가중치 (쉼표 구분, metrics와 길이 동일)",
    )
    parser.add_argument(
        "--lower-is-better",
        type=str,
        default="ES95,RuinPct,la_sf_mean",
        help="낮을수록 좋은 지표 목록 (쉼표 구분)",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="점수/정규화 결과를 저장할 CSV 경로 (기본: *_scored.csv)",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="최적 조합 및 기초 통계를 JSON으로 저장 (기본: *_best.json)",
    )

    args = parser.parse_args()

    metrics = _parse_list(args.metrics)
    weights = _parse_float_list(args.weights)
    lower_is_better = _parse_list(args.lower_is_better)

    rows, fields = _load_csv(args.summary_csv)
    if not rows:
        raise SystemExit("summary CSV에 데이터가 없습니다.")

    # 1) 정규화
    rows, ranges = _normalize_metrics(rows, metrics, lower_is_better)

    # 2) CompositeScore 계산
    _compute_composite(rows, metrics, weights)

    # 3) 점수 기준 정렬 및 rank 부여
    def _score_key(r: Dict[str, Any]) -> float:
        v = r.get("CompositeScore", "")
        try:
            return float(v)
        except Exception:
            return float("-inf")

    rows_sorted = sorted(rows, key=_score_key, reverse=True)
    for i, r in enumerate(rows_sorted, start=1):
        r["rank"] = i

    base_dir = os.path.dirname(os.path.abspath(args.summary_csv))
    base_name = os.path.splitext(os.path.basename(args.summary_csv))[0]

    out_csv = args.out_csv or os.path.join(base_dir, base_name + "_scored.csv")
    out_json = args.out_json or os.path.join(base_dir, base_name + "_best.json")

    _save_csv(out_csv, rows_sorted, fields)

    best = rows_sorted[0] if rows_sorted else {}
    # ranges는 (min,max) 튜플이므로 dict로 변환
    ranges_dict = {
        m: {
            "min": float(v[0]) if (v and math.isfinite(v[0])) else None,
            "max": float(v[1]) if (v and math.isfinite(v[1])) else None,
        }
        for m, v in ranges.items()
    }
    summary = {
        "summary_csv": os.path.abspath(args.summary_csv),
        "metrics": metrics,
        "weights": weights,
        "lower_is_better": lower_is_better,
        "ranges": ranges_dict,
        "best": best,
    }
    _save_json(out_json, summary)

    print(f"[OK] scored CSV  : {out_csv}")
    print(f"[OK] summary JSON: {out_json}")


if __name__ == "__main__":
    main()
