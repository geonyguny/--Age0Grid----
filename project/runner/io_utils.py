# project/runner/io_utils.py
from __future__ import annotations

import os
import csv
import datetime
from typing import Any, Dict

# ---------------------------------
# FS helpers
# ---------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_iso() -> str:
    # 파일명/CSV 타임스탬프에 쓰기 좋은 형식
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------
# CSV schema (fallback for autosave)
# ---------------------------------
SCHEMA_VERSION = "v2"

# 고정 헤더(필요 최소 집합; 미존재 필드는 공란으로 기록)
CSV_FIELDS = [
    "ts", "schema", "asset", "method", "es_mode", "tag",
    "alpha", "lambda", "F_target",
    "EW", "ES95", "Ruin", "mean_WT",
    "best_epoch", "train_time_s", "eval_time_s",
    "y_ann", "a_factor", "P",
    "fee_annual", "w_max", "horizon_years",
    # market/meta
    "market_mode", "bootstrap_block", "market_csv", "data_window", "use_real_rf", "data_profile",
    "bands", "outputs",
    "seeds", "n_paths",
    # RL 파라미터
    "rl_epochs", "rl_steps_per_epoch", "rl_n_paths_eval",
    "entropy_coef", "value_coef", "gae_lambda", "lr", "max_grad_norm",
    "rl_q_cap", "teacher_eps0", "teacher_decay",
    "survive_bonus", "u_scale", "lw_scale",
    # Hedge / mortality / annuity
    "hedge", "hedge_mode", "hedge_sigma_k", "hedge_cost", "hedge_tx",
    "mortality", "mort_table", "age0", "sex",
    "ann_on", "ann_alpha", "ann_L", "ann_d", "ann_index",
    "bequest_kappa", "bequest_gamma",
    # 산출물
    "ckpt_path",
    # (옵션) 사망 통계
    "death_count", "death_rate",
]


def _s(v: Any) -> Any:
    """CSV 안전 변환: None→"" / list,tuple→공백-구분 문자열."""
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return " ".join(str(x) for x in v)
    return v


# ---------------------------------
# Args slimming (stdout/json 간결화)
# ---------------------------------
def slim_args(args) -> dict:
    """
    출력(JSON)에 포함할 최소 args 세트만 추려서 반환.
    (존재하지 않는 키는 자동으로 None)
    """
    keys = [
        # 핵심
        "asset", "method", "baseline", "w_max", "fee_annual", "horizon_years",
        "alpha", "lambda_term", "F_target", "p_annual", "g_real_annual",
        "w_fixed", "floor_on", "f_min_real", "es_mode", "outputs",
        # HJB
        "hjb_W_grid", "hjb_Nshock", "hjb_eta_n",
        # 헤지/시장
        "hedge", "hedge_mode", "hedge_cost", "hedge_sigma_k", "hedge_tx",
        "market_mode", "market_csv", "bootstrap_block", "use_real_rf",
        # 사망/연금
        "mortality", "mort_table", "age0", "sex", "bequest_kappa", "bequest_gamma",
        # Stage-wise CVaR
        "cvar_stage", "alpha_stage", "lambda_stage", "cstar_mode", "cstar_m",
        # RL
        "rl_q_cap", "teacher_eps0", "teacher_decay", "lw_scale", "survive_bonus",
        "crra_gamma", "u_scale", "xai_on",
        "seeds", "n_paths",
        "rl_epochs", "rl_steps_per_epoch", "rl_n_paths_eval", "gae_lambda",
        "entropy_coef", "value_coef", "lr", "max_grad_norm",
        # Misc
        "q_floor", "beta", "quiet",
        # Annuity overlay
        "ann_on", "ann_alpha", "ann_L", "ann_d", "ann_index",
        # 데이터/프로파일/태그
        "bands", "data_window", "data_profile", "tag",
        # (보강) 믹스/FX 헤지 관련
        "alpha_mix", "alpha_kr", "alpha_us", "alpha_au",
        "h_FX", "fx_hedge_cost",
        # (옵션) EU 리포팅
        "report_utility", "delta_annual",
    ]
    return {k: getattr(args, k, None) for k in keys}


# ---------------------------------
# CSV appender (fallback; run.py는 기본적으로 utils.logging_io 사용)
# ---------------------------------
def append_metrics_csv(path: str, payload: Dict[str, Any]) -> None:
    """
    간단한 CSV 누적 기록기. run.py의 기본 기록기는
    project/utils/logging_io.py:write_metrics_csv 을 사용하고,
    이 함수는 autosave 폴백에서만 사용된다.
    """
    args = payload.get("args") or {}
    metrics = payload.get("metrics") or {}

    row = {
        "ts": now_iso(),
        "schema": SCHEMA_VERSION,
        "asset": payload.get("asset"),
        "method": payload.get("method"),
        "es_mode": payload.get("es_mode"),
        "tag": (args.get("tag") if isinstance(args, dict) else None) or "",

        "alpha": payload.get("alpha"),
        "lambda": payload.get("lambda_term"),
        "F_target": payload.get("F_target"),

        "EW": metrics.get("EW"),
        "ES95": metrics.get("ES95"),
        "Ruin": metrics.get("Ruin"),
        "mean_WT": metrics.get("mean_WT"),
        "best_epoch": metrics.get("best_epoch"),
        "train_time_s": metrics.get("train_time_s"),
        "eval_time_s": metrics.get("eval_time_s"),

        "y_ann": metrics.get("y_ann"),
        "a_factor": metrics.get("a_factor"),
        "P": metrics.get("P"),

        "fee_annual": payload.get("fee_annual"),
        "w_max": payload.get("w_max"),
        "horizon_years": payload.get("horizon_years"),

        # market/meta
        "market_mode": metrics.get("market_mode") or (args.get("market_mode") if isinstance(args, dict) else None),
        "bootstrap_block": metrics.get("bootstrap_block") or (args.get("bootstrap_block") if isinstance(args, dict) else None),
        "market_csv": metrics.get("market_csv") or (args.get("market_csv") if isinstance(args, dict) else None),
        "data_window": metrics.get("data_window") or (args.get("data_window") if isinstance(args, dict) else None),
        "use_real_rf": metrics.get("use_real_rf") or (args.get("use_real_rf") if isinstance(args, dict) else None),
        "data_profile": metrics.get("data_profile") or (args.get("data_profile") if isinstance(args, dict) else None),

        "bands": (args.get("bands") if isinstance(args, dict) else None),
        "outputs": (args.get("outputs") if isinstance(args, dict) else None),

        "seeds": (args.get("seeds") if isinstance(args, dict) else None),
        "n_paths": payload.get("n_paths"),

        "rl_epochs": (args.get("rl_epochs") if isinstance(args, dict) else None),
        "rl_steps_per_epoch": (args.get("rl_steps_per_epoch") if isinstance(args, dict) else None),
        "rl_n_paths_eval": (args.get("rl_n_paths_eval") if isinstance(args, dict) else None),
        "entropy_coef": (args.get("entropy_coef") if isinstance(args, dict) else None),
        "value_coef": (args.get("value_coef") if isinstance(args, dict) else None),
        "gae_lambda": (args.get("gae_lambda") if isinstance(args, dict) else None),
        "lr": (args.get("lr") if isinstance(args, dict) else None),
        "max_grad_norm": (args.get("max_grad_norm") if isinstance(args, dict) else None),

        "rl_q_cap": (args.get("rl_q_cap") if isinstance(args, dict) else None),
        "teacher_eps0": (args.get("teacher_eps0") if isinstance(args, dict) else None),
        "teacher_decay": (args.get("teacher_decay") if isinstance(args, dict) else None),
        "survive_bonus": (args.get("survive_bonus") if isinstance(args, dict) else None),
        "u_scale": (args.get("u_scale") if isinstance(args, dict) else None),
        "lw_scale": (args.get("lw_scale") if isinstance(args, dict) else None),

        "hedge": (args.get("hedge") if isinstance(args, dict) else None),
        "hedge_mode": (args.get("hedge_mode") if isinstance(args, dict) else None),
        "hedge_sigma_k": (args.get("hedge_sigma_k") if isinstance(args, dict) else None),
        "hedge_cost": (args.get("hedge_cost") if isinstance(args, dict) else None),
        "hedge_tx": (args.get("hedge_tx") if isinstance(args, dict) else None),

        # mortality/annuity
        "mortality": (args.get("mortality") if isinstance(args, dict) else None),
        "mort_table": (args.get("mort_table") if isinstance(args, dict) else None),
        "age0": (args.get("age0") if isinstance(args, dict) else None),
        "sex": (args.get("sex") if isinstance(args, dict) else None),
        "ann_on": (args.get("ann_on") if isinstance(args, dict) else None),
        "ann_alpha": (args.get("ann_alpha") if isinstance(args, dict) else None),
        "ann_L": (args.get("ann_L") if isinstance(args, dict) else None),
        "ann_d": (args.get("ann_d") if isinstance(args, dict) else None),
        "ann_index": (args.get("ann_index") if isinstance(args, dict) else None),
        "bequest_kappa": (args.get("bequest_kappa") if isinstance(args, dict) else None),
        "bequest_gamma": (args.get("bequest_gamma") if isinstance(args, dict) else None),

        "ckpt_path": payload.get("ckpt_path"),

        # (옵션) 사망 통계(있으면 기록)
        "death_count": metrics.get("death_count"),
        "death_rate": metrics.get("death_rate"),
    }

    # 직렬화 안전 변환
    for k in list(row.keys()):
        row[k] = _s(row[k])

    ensure_dir(os.path.dirname(path))
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


# ---------------------------------
# Autosave
# ---------------------------------
def do_autosave(metrics: dict, cfg, args, out_payload: dict) -> None:
    """
    1) 가능하면 project.eval.save_metrics_autocsv 사용
    2) 실패 시, 로컬 CSV(이 모듈의 append_metrics_csv)로 폴백
    """
    try:
        try:
            from ..eval import save_metrics_autocsv  # optional
            csv_path = save_metrics_autocsv(metrics, cfg, outputs=cfg.outputs)
            print(f"[autosave] metrics -> {csv_path}")
        except Exception:
            csv_path = os.path.join(cfg.outputs, "_logs", "metrics.csv")
            append_metrics_csv(csv_path, out_payload)
            print(f"[autosave] metrics -> {csv_path}")
    except Exception as e:
        print(f"[autosave] skipped: {e}")
