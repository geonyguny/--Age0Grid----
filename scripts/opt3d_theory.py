# scripts/opt3d_theory.py
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

from project.runner.run import run_once, run_rl


def _parse_float_grid(text: str) -> List[float]:
    if not text:
        return []
    out: List[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if part == "":
            continue
        out.append(float(part))
    return out


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_market_csv(args: argparse.Namespace) -> str:
    """
    data_profile 기준으로 기본 market CSV를 추론.
    - dev  -> kr_us_gold_bootstrap_mini_extended.csv
    - full -> kr_us_gold_bootstrap_full.csv
    """
    from pathlib import Path

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


def _build_base_args(args: argparse.Namespace, outputs_root: Path) -> SimpleNamespace:
    """
    run_rl / run_once 에 넘길 공통 args를 한 번 구성.
    개별 조합별로 ann_alpha / w_max / rl_q_cap / tag 만 덮어쓴다.
    """
    market_csv = _resolve_market_csv(args)

    base = dict(
        # 공통 I/O
        outputs=str(outputs_root),
        quiet="on",
        autosave="off",
        tag=None,  # 각 조합에서 채운다.

        # 시장 / 데이터
        asset=str(args.asset),
        market_mode=str(args.market_mode),
        market_csv=market_csv,
        data_profile=str(getattr(args, "data_profile", "") or ""),
        bootstrap_block=int(args.bootstrap_block),
        use_real_rf=str(args.use_real_rf),

        # ALM / 환경 기본 설정
        method=str(args.method),
        horizon_years=int(args.horizon_years),
        w_max=float(args.w_max),
        fee_annual=float(args.fee_annual),
        phi_adval=float(getattr(args, "phi_adval", 0.0) or 0.0),
        age0=int(args.age0),
        sex=str(args.sex),
        F_target=float(getattr(args, "F_target", 0.0) or 0.0),
        alpha=float(getattr(args, "alpha", 0.95) or 0.95),
        q_floor=float(getattr(args, "q_floor", 0.0) or 0.0),

        # annuity 관련 기본값 (실험 축에서 ann_alpha만 변경)
        ann_alpha=float(getattr(args, "ann_alpha_base", 0.0) or 0.0),
        ann_on="auto",  # run.py에서 ann_alpha>0이면 on 으로 해석

        # RL 관련 기본 설정
        rl_epochs=int(args.rl_epochs),
        rl_steps_per_epoch=int(args.rl_steps_per_epoch),
        rl_n_paths_eval=int(args.rl_n_paths_eval),
        n_paths=None,   # 필요 시 조합별로 덮어씌울 수 있음
        rl_q_cap=float(getattr(args, "rl_q_cap_base", 0.0) or 0.0),

        # 편향 옵션(기본 off)
        bias_on="off",
        bias_loss_aversion=None,
        bias_prob_gamma=None,
        bias_myopia=None,
        bias_w_floor=None,
        bias_w_cap_shock=None,
    )

    return SimpleNamespace(**base)


def _extract_metrics_row(
    out: Dict[str, Any],
    ann_alpha: float,
    w_max: float,
    q_cap: float,
    tag: str,
) -> Dict[str, Any]:
    """
    run_once / run_rl 결과(out)에서 3D 설계 축 + 주요 메트릭을 한 줄(row)로 추출.
    (후속 요약/스코어링에서 그대로 사용)
    """
    m = out.get("metrics", {}) or {}
    row: Dict[str, Any] = {}

    # 설계 축
    row["tag"] = tag
    row["ann_alpha"] = float(ann_alpha)
    row["w_max"] = float(w_max)
    row["rl_q_cap"] = float(q_cap)

    # 기본 메트릭
    row["EW"] = m.get("EW")
    row["ES95"] = m.get("ES95")
    row["RuinPct"] = m.get("RuinPct")
    row["mean_WT"] = m.get("mean_WT")

    # 소비/성공률 관련
    row["la_sf_mean"] = m.get("la_sf_mean")
    row["la_sf_rate"] = m.get("la_sf_rate")
    row["cons_coverage_mean"] = m.get("cons_coverage_mean")
    row["mean_cstar_amt"] = m.get("mean_cstar_amt")
    row["mean_consumption_amt"] = m.get("mean_consumption_amt")

    # wealth 분포 요약
    row["WT_p5"] = m.get("WT_p5")
    row["WT_p50"] = m.get("WT_p50")
    row["WT_p95"] = m.get("WT_p95")
    row["log10_WT_mean"] = m.get("log10_WT_mean")

    # annuity/환경 메타
    row["ann_on"] = m.get("ann_on")
    row["y_ann"] = m.get("y_ann")
    row["ann_a_factor"] = m.get("ann_a_factor")
    row["P"] = m.get("P")
    row["W_after_ann"] = m.get("W_after_ann")
    row["cstar_mode"] = m.get("cstar_mode")
    row["cstar_m"] = m.get("cstar_m")

    # 기대효용(할인된 효용 합) 관련 메트릭 (run_rl + run.py 수정분)
    row["EU_mean"] = m.get("EU_mean")
    row["EU_std"] = m.get("EU_std")

    # 기타 환경 설정
    row["method"] = out.get("method")
    row["asset"] = out.get("asset")
    row["alpha"] = out.get("alpha")
    row["F_target"] = out.get("F_target")
    row["es_mode"] = out.get("es_mode")
    row["n_paths"] = out.get("n_paths")

    timing = out.get("timing", {}) or {}
    row["time_total_s"] = timing.get("total_s")
    row["time_total_hms"] = timing.get("total_hms")

    return row


def _write_summary(
    rows: List[Dict[str, Any]],
    out_dir: Path,
    filename_prefix: str,
) -> None:
    """
    단순 요약:
    - <prefix>_summary.csv
    - <prefix>_summary.json
    """
    if not rows:
        print("[WARN] 요약할 행이 없습니다.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{filename_prefix}_summary.csv"
    json_path = out_dir / f"{filename_prefix}_summary.json"

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"[OK] opt3d summary CSV:  {csv_path}")
    print(f"[OK] opt3d summary JSON: {json_path}")


# --------------------------
# 스코어링 & best 설계 추출
# --------------------------
def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _score_row(row: Dict[str, Any]) -> float:
    """
    간단한 스코어 함수:
      - 기본: EU_mean(없으면 EW)을 최대화
      - 패널티: RuinPct, la_sf_mean 을 차감
    필요하면 가중치는 나중에 조정할 수 있도록 단순한 구조로 유지.
    """
    eu = row.get("EU_mean")
    ew = row.get("EW")
    ruin = row.get("RuinPct")
    la_sf = row.get("la_sf_mean")

    base = _as_float(eu, None) if eu is not None else None
    if base is None:
        base = _as_float(ew, 0.0)

    ruin_f = _as_float(ruin, 0.0)
    la_sf_f = _as_float(la_sf, 0.0)

    # 가중치(임시값): Ruin 10배, la_sf_mean 5배 패널티
    score = base - 10.0 * ruin_f - 5.0 * la_sf_f
    return score


def _write_scored_summary(
    rows: List[Dict[str, Any]],
    out_dir: Path,
    filename_prefix: str,
) -> None:
    """
    스코어 계산 후 정렬된 요약:
      - <prefix>_summary_scored.csv / .json
      - <prefix>_best.json (1등 설계 한 건)
    """
    if not rows:
        print("[WARN] 스코어링할 행이 없습니다.")
        return

    scored_rows: List[Dict[str, Any]] = []
    for r in rows:
        r_scored = dict(r)
        r_scored["score"] = _score_row(r)
        scored_rows.append(r_scored)

    # 점수 기준 내림차순 정렬
    scored_rows_sorted = sorted(
        scored_rows,
        key=lambda rr: _as_float(rr.get("score"), float("-inf")),
        reverse=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{filename_prefix}_summary_scored.csv"
    json_path = out_dir / f"{filename_prefix}_summary_scored.json"
    best_path = out_dir / f"{filename_prefix}_best.json"

    # CSV
    fieldnames = list(scored_rows_sorted[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in scored_rows_sorted:
            w.writerow(r)

    # JSON (전체)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(scored_rows_sorted, f, ensure_ascii=False, indent=2)

    # best 한 건
    best_row = scored_rows_sorted[0]
    with best_path.open("w", encoding="utf-8") as f:
        json.dump(best_row, f, ensure_ascii=False, indent=2)

    print(f"[OK] opt3d scored summary CSV:  {csv_path}")
    print(f"[OK] opt3d scored summary JSON: {json_path}")
    print(f"[OK] opt3d best design JSON:    {best_path}")


# --------------------------
# main
# --------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="세 축(q, w, ann_alpha) 이론적 최적설계용 3D 실험 스크립트"
    )

    # 출력/태그
    p.add_argument("--outputs", type=str, required=True)
    p.add_argument("--tag-prefix", type=str, default="OPT3D_BASE")

    # 세 축 그리드
    p.add_argument(
        "--ann-grid",
        type=str,
        required=True,
        help="예: 0.0,0.3,0.5,0.7,1.0",
    )
    p.add_argument(
        "--wmax-grid",
        type=str,
        required=True,
        help="예: 0.6,0.8,1.0",
    )
    p.add_argument(
        "--qcap-grid",
        type=str,
        required=True,
        help="예: 0.03,0.05,0.07 (연간 소비율 상한 수준)",
    )

    # 공통 시뮬레이션 설정
    p.add_argument("--method", type=str, default="rl",
                   help="rl / rule / hjb (현재는 rl 중심)")
    p.add_argument("--asset", type=str, default="KR")
    p.add_argument("--market-mode", type=str, default="bootstrap")
    p.add_argument("--data-profile", type=str, default="dev")
    p.add_argument("--market-csv", type=str, default=None)
    p.add_argument("--bootstrap-block", type=int, default=24)
    p.add_argument("--use-real-rf", type=str, default="on")

    p.add_argument("--horizon-years", type=int, default=30)
    p.add_argument("--w-max", type=float, default=1.0)
    p.add_argument("--fee-annual", type=float, default=0.004)
    p.add_argument("--phi-adval", type=float, default=0.0)
    p.add_argument("--age0", type=int, default=55)
    p.add_argument("--sex", type=str, default="M")
    p.add_argument("--alpha", type=float, default=0.95)
    p.add_argument("--F-target", type=float, default=0.0)
    p.add_argument("--q-floor", type=float, default=0.0)

    # annuity / RL 관련 기본값 (축이 아닌 부분)
    p.add_argument("--ann-alpha-base", type=float, default=0.0)
    p.add_argument("--rl-q-cap-base", type=float, default=0.0)

    # RL 학습/평가 관련
    p.add_argument("--rl-epochs", type=int, default=4)
    p.add_argument("--rl-steps-per-epoch", type=int, default=512)
    p.add_argument("--rl-n-paths-eval", type=int, default=500)

    args = p.parse_args()

    ann_grid = _parse_float_grid(args.ann_grid)
    wmax_grid = _parse_float_grid(args.wmax_grid)
    qcap_grid = _parse_float_grid(args.qcap_grid)

    if not ann_grid or not wmax_grid or not qcap_grid:
        raise SystemExit(
            "ann-grid, wmax-grid, qcap-grid 는 모두 최소 1개 이상의 값이 필요합니다."
        )

    outputs_root = _ensure_dir(args.outputs)
    base_args = _build_base_args(args, outputs_root)

    rows: List[Dict[str, Any]] = []

    print("[INFO] opt3d_theory 시작")
    print(f"  outputs    = {outputs_root}")
    print(f"  ann_grid   = {ann_grid}")
    print(f"  wmax_grid  = {wmax_grid}")
    print(f"  qcap_grid  = {qcap_grid}")
    print(f"  method     = {args.method}")

    for ann_alpha in ann_grid:
        for w_max in wmax_grid:
            for q_cap in qcap_grid:
                tag = (
                    f"{args.tag_prefix}"
                    f"_ann{ann_alpha:.2f}_w{w_max:.2f}_qcap{q_cap:.3f}"
                )
                print(
                    f"[RUN] ann_alpha={ann_alpha:.3f}, "
                    f"w_max={w_max:.3f}, rl_q_cap={q_cap:.3f} "
                    f"tag={tag}"
                )

                # 개별 run용 args 복사 후 축 값만 세팅
                run_args_dict = base_args.__dict__.copy()
                run_args_dict["ann_alpha"] = float(ann_alpha)
                run_args_dict["w_max"] = float(w_max)
                run_args_dict["rl_q_cap"] = float(q_cap)
                run_args_dict["tag"] = tag

                run_args = SimpleNamespace(**run_args_dict)

                if str(args.method).lower() == "rl":
                    out = run_rl(run_args)
                else:
                    out = run_once(run_args)

                row = _extract_metrics_row(
                    out,
                    ann_alpha=ann_alpha,
                    w_max=w_max,
                    q_cap=q_cap,
                    tag=tag,
                )
                rows.append(row)

    # 기본 요약
    _write_summary(rows, outputs_root, filename_prefix=args.tag_prefix)

    # 스코어링 요약 + best 설계
    _write_scored_summary(rows, outputs_root, filename_prefix=args.tag_prefix)

    print("[DONE] opt3d_theory 완료")


if __name__ == "__main__":
    main()
