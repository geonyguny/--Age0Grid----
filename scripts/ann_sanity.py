# scripts/ann_sanity.py
from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

from project.runner.run import run_once, run_rl  # run.py의 공용 러너 사용


# -------------------------
# 간단한 유틸
# -------------------------
def _parse_ann_grid(raw: str) -> List[float]:
    out: List[float] = []
    for p in str(raw).split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(float(p))
        except ValueError:
            raise SystemExit(f"[ERR] ann-grid 항목을 float로 해석할 수 없습니다: '{p}'")
    if not out:
        raise SystemExit("[ERR] ann-grid가 비어 있습니다.")
    return out


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: Any) -> None:
    _safe_mkdir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    _safe_mkdir(path.parent)
    fields = list(rows[0].keys())
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    tmp.replace(path)


def _resolve_market_csv(args: argparse.Namespace) -> str:
    """
    data_profile 기준으로 기본 market CSV를 추론.
    - dev  -> project/data/market/kr_us_gold_bootstrap_mini_extended.csv
    - full -> project/data/market/kr_us_gold_bootstrap_full.csv
    """
    if getattr(args, "market_csv", None):
        return str(args.market_csv)

    profile = str(getattr(args, "data_profile", "") or "").lower()
    if profile == "dev":
        cand = Path("project/data/market/kr_us_gold_bootstrap_mini_extended.csv")
    elif profile == "full":
        cand = Path("project/data/market/kr_us_gold_bootstrap_full.csv")
    else:
        raise SystemExit(
            "market_csv를 찾지 못했습니다. --market-csv 또는 "
            "--data-profile dev|full 중 하나를 지정해 주세요."
        )

    if not cand.exists():
        raise SystemExit(
            "market_csv 파일을 찾지 못했습니다.\n"
            f"  expected: {cand}\n"
            "경로를 확인하거나 --market-csv 로 직접 지정해 주세요."
        )
    return str(cand)


def _build_child_args(base: argparse.Namespace, ann_alpha: float, tag: str) -> SimpleNamespace:
    """
    base argparse.Namespace를 복사하여, ann_alpha/ann_on/tag 등을 세팅한
    SimpleNamespace를 만든다. run_rl / run_once 에 넘길 args 역할을 한다.
    """
    d = vars(base).copy()

    # tag / ann 관련 설정
    d["tag"] = tag
    d["ann_alpha"] = float(ann_alpha)
    d["ann_on"] = "on" if ann_alpha > 0.0 else "off"

    # 수수료 파라미터 정리: run_rl / run_once 내부에서 phi_adval 우선
    fee = float(d.get("fee_annual", 0.0) or 0.0)
    d["fee_annual"] = fee
    d["phi_adval"] = fee

    # RL 관련 필수 파라미터 (run_rl에서 직접 참조)
    d.setdefault("rl_epochs", base.rl_epochs)
    d.setdefault("rl_steps_per_epoch", base.rl_steps_per_epoch)
    d.setdefault("rl_n_paths_eval", base.rl_n_paths_eval)

    # 기타 안전한 기본값들
    d.setdefault("floor_on", "off")
    d.setdefault("f_min_real", 0.0)
    d.setdefault("F_target", 0.0)
    d.setdefault("q_floor", 0.0)
    d.setdefault("rl_q_cap", 0.0)
    d.setdefault("bias_on", "off")

    # market_csv 보강 (data_profile 기반 자동 추론)
    d["market_csv"] = _resolve_market_csv(base)

    return SimpleNamespace(**d)


def _run_one_ann(ann_alpha: float, base_args: argparse.Namespace):
    """
    ann_alpha 하나에 대해 run_rl 또는 run_once 실행 후,
    (ann_alpha, metrics, out(dict)) 를 반환.
    """
    tag = f"{base_args.tag_prefix}_ann_{ann_alpha:.3f}".replace(".", "p")
    child_args = _build_child_args(base_args, ann_alpha, tag)

    print(f"[INFO] ann_alpha={ann_alpha} (ann_on={child_args.ann_on}) 실행 시작...")

    if str(child_args.method).lower() == "rl":
        out = run_rl(child_args)
    else:
        out = run_once(child_args)

    metrics: Dict[str, Any] = out.get("metrics", {}) if isinstance(out, dict) else {}
    return ann_alpha, metrics, out


def main():
    parser = argparse.ArgumentParser(description="Annuity sanity check runner (ann_sanity)")
    parser.add_argument(
        "--outputs",
        type=str,
        required=True,
        help="실험 결과를 저장할 루트 디렉터리 (예: ./outputs/ANN_SANITY)",
    )
    parser.add_argument(
        "--tag-prefix",
        type=str,
        default="ANN_SANITY",
        help="개별 run에 붙일 태그 prefix",
    )
    parser.add_argument(
        "--ann-grid",
        type=str,
        required=True,
        help="콤마로 구분된 ann_alpha 목록 (예: 0.0,0.3,0.5)",
    )

    # 공통 시뮬레이션 설정
    parser.add_argument("--method", type=str, default="rl", help="방법론: rl / rule / hjb 등")
    parser.add_argument("--asset", type=str, default="KR")
    parser.add_argument("--market-mode", type=str, default="bootstrap")
    parser.add_argument("--market-csv", type=str, default=None)
    parser.add_argument("--data-profile", type=str, default="dev")
    parser.add_argument("--bootstrap-block", type=int, default=24)
    parser.add_argument("--use-real-rf", type=str, default="on")
    parser.add_argument("--horizon-years", type=int, default=30)
    parser.add_argument("--w-max", type=float, default=1.0)
    parser.add_argument("--fee-annual", type=float, default=0.004)
    parser.add_argument("--age0", type=int, default=55)
    parser.add_argument("--sex", type=str, default="M")

    # RL 전용 하이퍼파라미터 (여기서 기본값을 반드시 지정)
    parser.add_argument(
        "--rl-epochs",
        type=int,
        default=4,
        help="RL 학습 epoch 수 (기본 4)",
    )
    parser.add_argument(
        "--rl-steps-per-epoch",
        type=int,
        default=512,
        help="epoch당 roll-out step 수 (기본 512)",
    )
    parser.add_argument(
        "--rl-n-paths-eval",
        type=int,
        default=300,
        help="평가용 시뮬레이션 path 수",
    )

    # 기타 선택 옵션
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", type=str, default="on")

    args = parser.parse_args()

    # ann_grid 파싱
    ann_grid = _parse_ann_grid(args.ann_grid)

    # outputs 절대경로 정리
    out_root = Path(args.outputs).resolve()
    _safe_mkdir(out_root)
    print(f"[INFO] ann_sanity 시작: outputs={out_root}")
    print(f"[INFO] ann_grid={ann_grid}, method={args.method}")

    # market_csv 자동 해석 결과를 한 번만 출력 (base args 차원에서)
    resolved_csv = _resolve_market_csv(args)
    print(f"[INFO] market_csv={resolved_csv}")

    summary_rows: List[Dict[str, Any]] = []

    for a in ann_grid:
        ann_alpha, metrics, out = _run_one_ann(a, args)

        # EU_mean / EU_std 추출
        meta = out.get("meta", {}) if isinstance(out, dict) else {}
        eu_mean = metrics.get("EU_mean")
        if eu_mean is None:
            eu_mean = metrics.get("eval_return_mean", None)
        if eu_mean is None:
            eu_mean = meta.get("eval_return_mean", None)

        eu_std = metrics.get("EU_std")
        if eu_std is None:
            eu_std = meta.get("eval_return_std", None)

        row = {
            "ann_alpha": ann_alpha,
            "ann_on": metrics.get("ann_on"),
            "EW": metrics.get("EW"),
            "ES95": metrics.get("ES95"),
            "RuinPct": metrics.get("RuinPct"),
            "y_ann": metrics.get("y_ann"),
            "ann_a_factor": metrics.get("ann_a_factor"),
            "P": metrics.get("P"),
            "W_after_ann": metrics.get("W_after_ann"),
            "la_sf_mean": metrics.get("la_sf_mean"),
            "cons_coverage_mean": metrics.get("cons_coverage_mean"),
            "WT_p5": metrics.get("WT_p5"),
            "WT_p50": metrics.get("WT_p50"),
            "WT_p95": metrics.get("WT_p95"),
            "log10_WT_mean": metrics.get("log10_WT_mean"),
            # 효용 기반 요약
            "EU_mean": eu_mean,
            "EU_std": eu_std,
            # 태그
            "tag": (out.get("args", {}) or {}).get("tag", None)
            if isinstance(out, dict)
            else None,
        }
        summary_rows.append(row)

    # 요약 저장
    summary_csv = out_root / "ann_sanity_summary.csv"
    summary_json = out_root / "ann_sanity_summary.json"
    _write_csv(summary_csv, summary_rows)
    _write_json(summary_json, summary_rows)

    print(f"[OK] ann_sanity 완료: {summary_csv}")
    print(f"[OK] ann_sanity 완료: {summary_json}")


if __name__ == "__main__":
    main()
