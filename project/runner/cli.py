# project/runner/cli.py
from __future__ import annotations

import argparse
import json
import os
import re
import time
import sys
from typing import Any, Dict, List

from ..config import (
    CVAR_TARGET_DEFAULT,
    CVAR_TOL_DEFAULT,
    LAMBDA_MIN_DEFAULT,
    LAMBDA_MAX_DEFAULT,
)
from .run import run_once, run_rl
from .calibrate import calibrate_lambda

# ETA / CVaR / stdout packing
from .eta_utils import (
    fmt_hms, parse_hms_to_seconds,
    eta_db_path, eta_load_db, predict_eta_from_history, eta_record,
)
from .cvar_utils import fixup_metrics_with_cvar
from .pack_utils import prune_for_stdout, maybe_evaluate_with_es_mode

# (옵션) 구성 해시/버전 정보
try:
    from ..utils.config_hash import config_hash as _cfg_hash_fn  # type: ignore
except Exception:
    _cfg_hash_fn = None
try:
    from ..utils.version_info import get_version_info as _ver_info_fn  # type: ignore
except Exception:
    _ver_info_fn = None

# YYYY-MM:YYYY-MM, YYYY-MM:, :YYYY-MM
_WINDOW_RE = re.compile(r"^(?:\d{4}-\d{2})?:(?:\d{4}-\d{2})?$")


# -------------------------
# helpers
# -------------------------
def _csv_floats(s: str | None) -> List[float] | None:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    out: List[float] = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    return out or None


def _normalize_outputs_path(p: str | None) -> str:
    base = p or "./outputs"
    abs_p = os.path.abspath(base)
    os.makedirs(abs_p, exist_ok=True)
    return abs_p


def _normalize_seeds(seeds_arg: List[int]) -> List[int]:
    if not isinstance(seeds_arg, list) or len(seeds_arg) == 0:
        return [0, 1, 2, 3, 4]
    if len(seeds_arg) == 1:
        n = int(seeds_arg[0])
        return list(range(max(n, 1)))
    return sorted({int(x) for x in seeds_arg})


def _print_metrics_summary(metrics: dict) -> None:
    lines = [
        f"EW            : {metrics.get('EW')}",
        f"ES95          : {metrics.get('ES95')}",
        f"EL            : {metrics.get('EL')}",
        f"Ruin          : {metrics.get('Ruin')}",
        f"mean_WT       : {metrics.get('mean_WT')}",
        f"es95_source   : {metrics.get('es95_source')}",
        # EU 리포팅
        f"EU            : {metrics.get('EU')}",
        f"EU_per_year   : {metrics.get('EU_per_year')}",
        f"delta_annual  : {metrics.get('delta_annual')}",
        # 손실 기준선 감사
        f"F_target_used : {metrics.get('F_target_used')}",
    ]
    print("\n".join([s for s in lines if s is not None]))


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    # Core
    p.add_argument("--asset", type=str, default="KR")
    p.add_argument("--method", type=str, default="hjb", choices=["hjb", "rl", "rule"])
    p.add_argument("--baseline", type=str, default=None)
    p.add_argument("--w_max", type=float, default=0.70)
    p.add_argument("--fee_annual", type=float, default=0.004)
    p.add_argument("--phi_adval", type=float, default=None, help="(선택) fee_annual 대신 선취 ad-valorem 수수료")
    p.add_argument("--horizon_years", type=int, default=35)
    p.add_argument("--alpha", type=float, default=0.95)
    p.add_argument("--lambda_term", type=float, default=0.0)
    p.add_argument("--F_target", type=float, default=0.0)
    p.add_argument("--p_annual", type=float, default=0.04)
    p.add_argument("--g_real_annual", type=float, default=0.02)
    p.add_argument("--w_fixed", type=float, default=0.60)
    p.add_argument("--floor_on", action="store_true")
    p.add_argument("--f_min_real", type=float, default=0.0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n_paths", type=int, default=100)
    p.add_argument("--es_mode", type=str, default="wealth", choices=["wealth", "loss"])
    p.add_argument("--outputs", type=str, default="./outputs")

    # HJB
    p.add_argument("--hjb_W_grid", type=int, default=None)
    p.add_argument("--hjb_Nshock", type=int, default=None)
    p.add_argument("--hjb_eta_n", type=int, default=None)
    p.add_argument("--hjb_w_grid", type=str, default=None,
                   help="Comma-separated w grid for HJB (e.g. '0.10,0.20,0.30,0.40,0.50').")
    p.add_argument("--w_min_dev", type=float, default=None,
                   help="(dev) Minimum risky weight; drop w < w_min_dev from HJB action set.")

    # Hedge
    p.add_argument("--hedge", choices=["on", "off"], default="off")
    p.add_argument("--hedge_mode", choices=["mu", "sigma", "downside"], default="sigma")
    p.add_argument("--hedge_cost", type=float, default=0.005)
    p.add_argument("--hedge_sigma_k", type=float, default=0.20)
    p.add_argument("--hedge_tx", type=float, default=0.0)

    # Market
    p.add_argument("--market_mode", choices=["iid", "bootstrap"], default="iid")
    p.add_argument("--market_csv", type=str, default=None)
    p.add_argument("--bootstrap_block", type=int, default=24)
    p.add_argument("--use_real_rf", choices=["on", "off"], default="on")

    # Mortality
    p.add_argument("--mortality", choices=["on", "off"], default="off")
    p.add_argument("--mort_table", type=str, default=None)
    p.add_argument("--age0", type=int, default=65)
    p.add_argument("--sex", choices=["M", "F"], default="M")
    p.add_argument("--bequest_kappa", type=float, default=0.0)
    p.add_argument("--bequest_gamma", type=float, default=1.0)

    # CVaR Calibration
    p.add_argument("--calib", choices=["on", "off"], default="off")
    p.add_argument("--calib_param", choices=["lambda", "F"], default="lambda")
    p.add_argument("--cvar_target", type=float, default=CVAR_TARGET_DEFAULT)
    p.add_argument("--cvar_tol", type=float, default=CVAR_TOL_DEFAULT)
    p.add_argument("--lambda_min", type=float, default=LAMBDA_MIN_DEFAULT)
    p.add_argument("--lambda_max", type=float, default=LAMBDA_MAX_DEFAULT)
    p.add_argument("--calib_fast", choices=["on", "off"], default="on")
    p.add_argument("--calib_max_iter", type=int, default=8)
    p.add_argument("--F_min", type=float, default=None)
    p.add_argument("--F_max", type=float, default=None)

    # autosave
    p.add_argument("--autosave", choices=["on", "off"], default="off")

    # RL
    p.add_argument("--rl_epochs", type=int, default=60)
    p.add_argument("--rl_steps_per_epoch", type=int, default=2048)
    p.add_argument("--rl_n_paths_eval", type=int, default=300)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--rl_q_cap", type=float, default=0.0)
    p.add_argument("--teacher_eps0", type=float, default=0.0)
    p.add_argument("--teacher_decay", type=float, default=1.0)
    p.add_argument("--lw_scale", type=float, default=0.0)
    p.add_argument("--survive_bonus", type=float, default=0.0)
    p.add_argument("--crra_gamma", type=float, default=3.0)
    p.add_argument("--u_scale", type=float, default=0.0)

    # Lite overrides
    p.add_argument("--q_floor", type=float, default=None)
    p.add_argument("--beta", type=float, default=None)
    # Utility reporting
    p.add_argument("--report_utility", choices=["on", "off"], default="off")
    p.add_argument("--delta_annual", type=float, default=None)

    # Stage-wise CVaR
    p.add_argument("--cvar_stage", choices=["on", "off"], default="off")
    p.add_argument("--alpha_stage", type=float, default=0.95)
    p.add_argument("--lambda_stage", type=float, default=0.0)
    p.add_argument("--cstar_mode", choices=["fixed", "annuity", "vpw"], default="annuity")
    p.add_argument("--cstar_m", type=float, default=0.04 / 12)

    # XAI
    p.add_argument("--xai_on", choices=["on", "off"], default="on")

    # QUIET
    p.add_argument("--quiet", choices=["on", "off"], default="on")

    # ANN Overlay
    p.add_argument("--ann_on", choices=["on", "off"], default="off")
    p.add_argument("--ann_alpha", type=float, default=0.0)
    p.add_argument("--ann_L", type=float, default=0.0)
    p.add_argument("--ann_d", type=int, default=0)
    p.add_argument("--ann_index", choices=["real", "nominal"], default="real")

    # Data/profile/tag
    p.add_argument("--bands", choices=["on", "off"], default="on")
    p.add_argument("--data_window", type=str, default=None)
    p.add_argument("--data_profile", choices=["dev", "full"], default=None)
    p.add_argument("--tag", type=str, default=None)

    # Allocation & FX hedge
    p.add_argument("--alpha_mix", type=str, default=None)
    p.add_argument("--alpha_kr", type=float, default=None)
    p.add_argument("--alpha_us", type=float, default=None)
    p.add_argument("--alpha_au", type=float, default=None)
    p.add_argument("--h_FX", type=float, default=None)
    p.add_argument("--fx_hedge_cost", type=float, default=None)

    # stdout control
    p.add_argument("--print_mode", choices=["full", "metrics", "summary"], default="full")
    p.add_argument("--metrics_keys", type=str,
                   default="EW,ES95,EL,Ruin,mean_WT,es_mode,EU,EU_per_year,delta_annual,F_target_used")
    p.add_argument("--no_paths", action="store_true")
    p.add_argument("--validate", choices=["on","off"], default="off")
    p.add_argument("--return_actor", choices=["on", "off"], default="off")

    # ETA
    p.add_argument("--eta_mode", choices=["off", "history"], default="history")
    p.add_argument("--eta_budget_hms", type=str, default=None)
    p.add_argument("--eta_budget_s", type=float, default=None)
    p.add_argument("--eta_hard_stop", choices=["on", "off"], default="on")
    p.add_argument("--eta_db", type=str, default=None)

    return p


def _apply_data_profile_defaults(args) -> None:
    if getattr(args, "data_profile", None) and not getattr(args, "market_csv", None):
        base = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "market"))
        if args.data_profile == "dev":
            args.market_csv = os.path.abspath(os.path.join(base, "kr_us_gold_bootstrap_mini.csv"))
        elif args.data_profile == "full":
            args.market_csv = os.path.abspath(os.path.join(base, "kr_us_gold_bootstrap_full.csv"))


def _validate_args(args) -> None:
    if args.data_window is not None:
        s = str(args.data_window).strip()
        if s and not _WINDOW_RE.match(s):
            raise SystemExit(f"--data_window 형식 오류: '{args.data_window}'. 예: 2005-01:2020-12 또는 '2005-01:' / ':2020-12'")
    if args.method in ("hjb", "rule", "rl") and args.market_mode == "bootstrap":
        if not args.market_csv and not args.data_profile:
            raise SystemExit("market_mode=bootstrap 사용 시 --market_csv 또는 --data_profile(dev|full) 필요.")
    if args.method == "rule" and not args.baseline:
        raise SystemExit("method=rule 사용 시 --baseline (4pct|cpb|vpw|kgr) 필수.")


def _inject_meta(out: Dict[str, Any], args) -> None:
    if not isinstance(out, dict):
        return
    meta = out.setdefault("meta", {})
    meta.setdefault("tag", getattr(args, "tag", None))
    meta.setdefault("method", getattr(args, "method", None))
    meta.setdefault("asset", getattr(args, "asset", None))
    meta.setdefault("outputs_abs", getattr(args, "outputs", None))
    if _cfg_hash_fn:
        try:
            meta["config_hash"] = _cfg_hash_fn(vars(args))
        except Exception:
            pass
    if _ver_info_fn:
        try:
            vi = _ver_info_fn()
            if isinstance(vi, dict):
                meta.update({k: vi.get(k) for k in ("git_commit", "py_ver", "np_ver")})
        except Exception:
            pass


def main():
    t0 = time.perf_counter()
    p = _build_arg_parser()
    args = p.parse_args()

    # normalize
    args.outputs = _normalize_outputs_path(getattr(args, "outputs", None))
    try:
        args.seeds = _normalize_seeds(list(getattr(args, "seeds", [])))
    except Exception:
        args.seeds = [0, 1, 2, 3, 4]

    parsed_w_grid = _csv_floats(getattr(args, "hjb_w_grid", None))
    if parsed_w_grid is not None:
        args.hjb_w_grid = parsed_w_grid

    _apply_data_profile_defaults(args)
    _validate_args(args)

    # ETA budget check (best-effort)
    try:
        if str(getattr(args, "eta_mode", "history")).lower() == "history":
            db = eta_load_db(eta_db_path(args))
            eta_s, src = predict_eta_from_history(args, db)
            if eta_s is not None:
                print(f"[ETA] ~{fmt_hms(eta_s)} (source={src}) … starting", file=sys.stderr, flush=True)
                budget = getattr(args, "eta_budget_s", None)
                if budget is None:
                    budget = parse_hms_to_seconds(getattr(args, "eta_budget_hms", None))
                if budget is not None and eta_s > float(budget):
                    if str(getattr(args, "eta_hard_stop", "on")).lower() == "on":
                        print(f"[ETA] exceeds budget {fmt_hms(budget)} → abort.", file=sys.stderr, flush=True)
                        sys.exit(3)
                    else:
                        print(f"[ETA] exceeds budget {fmt_hms(budget)} → continue (soft-warn).",
                              file=sys.stderr, flush=True)
    except Exception as _e:
        print(f"[ETA] predictor skipped ({type(_e).__name__})", file=sys.stderr, flush=True)

    want_paths = (str(getattr(args, "print_mode", "full")).lower() == "full") and (not getattr(args, "no_paths", False))

    # routing
    if str(getattr(args, "calib", "off")).lower() == "on":
        out = calibrate_lambda(args)
        if isinstance(out, dict) and "es_mode" not in out:
            out["es_mode"] = str(getattr(args, "es_mode", "wealth")).lower()
    else:
        if args.method == "rl":
            res = run_rl(args)
            out = maybe_evaluate_with_es_mode(res, es_mode=getattr(args, "es_mode", "wealth"), want_paths=want_paths)
        else:
            res = run_once(args)
            out = maybe_evaluate_with_es_mode(res, es_mode=getattr(args, "es_mode", "wealth"), want_paths=want_paths)

    # meta
    if isinstance(out, dict):
        out.setdefault("tag", getattr(args, "tag", None))
        out.setdefault("method", getattr(args, "method", None))
        out.setdefault("asset", getattr(args, "asset", None))
        out.setdefault("outputs_abs", getattr(args, "outputs", None))
        _inject_meta(out, args)

    # ES95(CVaR) fixup (best-effort)
    try:
        if isinstance(out, dict):
            out = fixup_metrics_with_cvar(args, out)
    except Exception as _e:
        try:
            if isinstance(out, dict):
                tgt = out["metrics"] if "metrics" in out and isinstance(out["metrics"], dict) else out
                tgt["es95_note"] = f"post-fixup failed: {type(_e).__name__}"
        except Exception:
            pass

    # extras: eval_WT_n
    try:
        if isinstance(out, dict) and isinstance(out.get("extra"), dict):
            ew = out["extra"].get("eval_WT")
            if isinstance(ew, (list, tuple)):
                out["extra"]["eval_WT_n"] = len(ew)
    except Exception:
        pass

    # total time
    elapsed = time.perf_counter() - t0
    if isinstance(out, dict):
        out["time_total_s"] = round(elapsed, 3)
        out["time_total_hms"] = fmt_hms(elapsed)

    # Validation (simple)
    try:
        if str(getattr(args, "validate","off")).lower()=="on" and isinstance(out, dict):
            from .validate.checks import basic_checks
            tgt = out["metrics"] if "metrics" in out and isinstance(out["metrics"], dict) else out
            vres = basic_checks(tgt, hjb_ref=None)
            out.setdefault("validation", vres)
    except Exception:
        pass

    # ETA history
    try:
        eta_record(args, elapsed)
    except Exception:
        pass

    # summary mode short-circuit
    if str(getattr(args, "print_mode", "full")).lower() == "summary":
        if isinstance(out, dict):
            metrics = out["metrics"] if "metrics" in out and isinstance(out["metrics"], dict) else out
            _print_metrics_summary(metrics)
        else:
            print("[warn] summary mode but no metrics dict available.")
        return

    # stdout (JSON)
    to_print = prune_for_stdout(args, out) if isinstance(out, dict) else out
    print(json.dumps(to_print, ensure_ascii=False))


if __name__ == "__main__":
    main()
