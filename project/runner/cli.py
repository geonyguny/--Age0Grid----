# project/runner/cli.py
from __future__ import annotations

import argparse, json, os, re, time, sys, io, contextlib
from typing import Any, Dict, List

from ..config import (
    CVAR_TARGET_DEFAULT, CVAR_TOL_DEFAULT, LAMBDA_MIN_DEFAULT, LAMBDA_MAX_DEFAULT,
)
from .run import run_once, run_rl
from .calibrate import calibrate_lambda

from .eta_utils import (
    fmt_hms, parse_hms_to_seconds, eta_db_path, eta_load_db, predict_eta_from_history, eta_record,  # noqa: F401
)
from .cvar_utils import fixup_metrics_with_cvar
from .pack_utils import prune_for_stdout, maybe_evaluate_with_es_mode

# ── tests 호환용: _cvar_fallback 심볼 제공 (Discrete ES with interpolation)
import numpy as _np

def _cvar_fallback(x, a_or_F, maybe_alpha=None):
    """
    두 형태 모두 지원:
      1) _cvar_fallback(L, alpha): L=손실 표본, alpha=신뢰수준
      2) _cvar_fallback(W, F, alpha): W=말기부(wealth) 표본, F=목표, alpha=신뢰수준 → L = max(F-W,0)
    구현은 이산 표본에서의 ES 보간식(Acerbi–Tasche 스타일)에 맞춤.
    """
    def _es_discrete(L: _np.ndarray, alpha: float) -> float:
        L = _np.asarray(L, dtype=float)
        L = L[_np.isfinite(L)]
        n = int(L.size)
        if n == 0:
            return 0.0
        a = float(alpha)
        if a <= 0.0:
            return float(_np.mean(L))
        if a >= 1.0:
            return float(_np.max(L))
        # 오름차순 정렬(끝으로 갈수록 손실이 큼)
        x = _np.sort(L)
        t = a * n                      # fractional index
        k = int(_np.floor(t))          # 0-based
        if k >= n:
            k = n - 1
        beta = t - k                   # [0,1)
        # tail 평균: x[k]를 (1-beta) 가중으로 포함 + x[k+1..n-1] 전부 포함
        # 분모: n*(1-alpha)
        tail_sum = x[k+1:].sum()
        tail_sum += (1.0 - beta) * x[k]
        denom = n * (1.0 - a)
        return float(tail_sum / denom)

    x = _np.asarray(x, dtype=float)
    if maybe_alpha is None:
        # 손실 표본 L
        alpha = float(a_or_F)
        return _es_discrete(x, alpha)
    else:
        # 부(wealth) 표본 → 손실로 변환
        F = float(a_or_F)
        alpha = float(maybe_alpha)
        L = _np.maximum(F - x, 0.0)
        return _es_discrete(L, alpha)

# Behavioral meta (옵션)
try:
    from ..policy.behavioral import parse_behavioral_from_args, describe as _bh_describe  # type: ignore
except Exception:
    parse_behavioral_from_args = None  # type: ignore
    _bh_describe = None  # type: ignore

# (옵션) 구성 해시/버전
try:
    from ..utils.config_hash import config_hash as _cfg_hash_fn  # type: ignore
except Exception:
    _cfg_hash_fn = None
try:
    from ..utils.version_info import get_version_info as _ver_info_fn  # type: ignore
except Exception:
    _ver_info_fn = None

_WINDOW_RE = re.compile(r"^(?:\d{4}-\d{2})?:(?:\d{4}-\d{2})?$")

def _csv_floats(s: str | None) -> List[float] | None:
    if not s:
        return None
    out: List[float] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    return out or None

def _normalize_outputs_path(p: str | None) -> str:
    base = p or "./outputs"
    abs_p = os.path.abspath(base)
    os.makedirs(abs_p, exist_ok=True)
    os.makedirs(os.path.join(abs_p, "_logs"), exist_ok=True)
    return abs_p

def _normalize_seeds(seeds_arg: List[int]) -> List[int]:
    if not isinstance(seeds_arg, list) or not seeds_arg:
        return [0, 1, 2, 3, 4]
    if len(seeds_arg) == 1:
        return [int(seeds_arg[0])]
    return sorted({int(x) for x in seeds_arg})

def _safe_print_json(obj: Any) -> None:
    try:
        s = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except TypeError:
        s = json.dumps(obj, default=str, ensure_ascii=False, sort_keys=True)
    print(s)

def _parse_block(s: str | int | None) -> int:
    """예: '6m', '12'(개월), '90d'(일→월 근사: ~30일=1개월), '2y'(연×12)."""
    if s is None:
        return 24
    if isinstance(s, int):
        return int(s)
    txt = str(s).strip().lower()
    m = re.fullmatch(r'(\d+)\s*([dmy]?)', txt)
    if not m:
        raise argparse.ArgumentTypeError(f"invalid block spec: {s}")
    n = int(m.group(1))
    unit = m.group(2) or 'm'
    if unit == 'm':
        return max(1, n)
    if unit == 'y':
        return max(1, n * 12)
    if unit == 'd':
        return max(1, round(n / 30))
    raise argparse.ArgumentTypeError(f"invalid unit: {s}")

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Retirement experiment runner (JSON-only stdout).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # High-level
    p.add_argument("--mode", default="auto", choices=["auto","once","rl","calib"])
    # Core
    p.add_argument("--asset", default="KR")
    p.add_argument("--method", default="hjb", choices=["hjb","rl","rule"])
    p.add_argument("--baseline")
    p.add_argument("--w_max", type=float, default=0.70)
    p.add_argument("--fee_annual", type=float, default=0.004)
    p.add_argument("--phi_adval", type=float, help="optional: ad-valorem fee (if set, replaces fee_annual)")
    p.add_argument("--horizon_years", type=int, default=35)
    p.add_argument("--alpha", type=float, default=0.95)
    p.add_argument("--lambda_term", type=float, default=0.0)
    p.add_argument("--F_target", type=float, default=0.0)
    p.add_argument("--p_annual", type=float, default=0.04)
    p.add_argument("--g_real_annual", type=float, default=0.02)
    p.add_argument("--w_fixed", type=float, default=0.60)
    p.add_argument("--floor_on", action="store_true")
    p.add_argument("--f_min_real", type=float, default=0.0)
    # Seeds/paths
    p.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4])
    p.add_argument("--seed", type=int, help="single seed (overrides --seeds)")
    p.add_argument("--n_paths", type=int, default=100)
    # ES mode
    p.add_argument("--es_mode", default="wealth", choices=["wealth","loss"],
                   help="ES95 convention; 'loss' uses L=max(F−W,0) reporting")
    p.add_argument("--outputs", default="./outputs")
    # HJB
    p.add_argument("--hjb_W_grid", type=int)
    p.add_argument("--hjb_Nshock", type=int)
    p.add_argument("--hjb_eta_n", type=int)
    p.add_argument("--hjb_w_grid", help="comma-separated risky weight grid for HJB, e.g. '0.10,0.20,0.30'")
    p.add_argument("--w_min_dev", type=float, help="dev: drop HJB actions with w< w_min_dev")
    # Legacy hedge
    p.add_argument("--hedge", choices=["on","off"], default="off")
    p.add_argument("--hedge_mode", choices=["mu","sigma","downside"], default="sigma")
    p.add_argument("--hedge_cost", type=float, default=0.005)
    p.add_argument("--hedge_sigma_k", type=float, default=0.20)
    p.add_argument("--hedge_tx", type=float, default=0.0)
    # Market
    p.add_argument("--market_mode", choices=["iid","bootstrap"], default="iid")
    p.add_argument("--market_csv")
    p.add_argument("--bootstrap_block", default="24",
                   help="block length: '6m', '12'(months), '90d'(days≈months), '2y'(years×12)")
    p.add_argument("--use_real_rf", choices=["on","off"], default="on")
    # Mortality
    p.add_argument("--mortality", choices=["on","off"], default="off")
    p.add_argument("--mort_table")
    p.add_argument("--age0", type=int, default=65)
    p.add_argument("--sex", choices=["M","F"], default="M")
    p.add_argument("--bequest_kappa", type=float, default=0.0)
    p.add_argument("--bequest_gamma", type=float, default=1.0)
    # CVaR Calibration
    p.add_argument("--calib", choices=["on","off"], default="off")
    p.add_argument("--calib_param", choices=["lambda","F"], default="lambda")
    p.add_argument("--cvar_target", type=float, default=CVAR_TARGET_DEFAULT)
    p.add_argument("--cvar_tol", type=float, default=CVAR_TOL_DEFAULT)
    p.add_argument("--lambda_min", type=float, default=LAMBDA_MIN_DEFAULT)
    p.add_argument("--lambda_max", type=float, default=LAMBDA_MAX_DEFAULT)
    p.add_argument("--calib_fast", choices=["on","off"], default="on")
    p.add_argument("--calib_max_iter", type=int, default=8)
    p.add_argument("--F_min", type=float)
    p.add_argument("--F_max", type=float)
    # autosave
    p.add_argument("--autosave", choices=["on","off"], default="off")
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
    # Lite / Utility
    p.add_argument("--q_floor", type=float)
    p.add_argument("--beta", type=float, help="utility present-bias; recommend 0<beta<=1")
    p.add_argument("--report_utility", choices=["on","off"], default="off")
    p.add_argument("--delta_annual", type=float)
    # Stage-wise CVaR
    p.add_argument("--cvar_stage", choices=["on","off"], default="off")
    p.add_argument("--alpha_stage", type=float, default=0.95)
    p.add_argument("--lambda_stage", type=float, default=0.0)
    p.add_argument("--cstar_mode", choices=["fixed","annuity","vpw"], default="annuity")
    p.add_argument("--cstar_m", type=float, default=0.04/12)
    # XAI
    p.add_argument("--xai_on", choices=["on","off"], default="on")
    # Verbose/Quiet
    p.add_argument("--quiet", choices=["on","off"], default="on")
    p.add_argument("--verbose", choices=["on","off"], default="off",
                   help="print a bit more logs (bias params etc.)")
    # ANN overlay
    p.add_argument("--ann_on", choices=["on","off"], default="off")
    p.add_argument("--ann_alpha", type=float, default=0.0)
    p.add_argument("--ann_L", type=float, default=0.0)
    p.add_argument("--ann_d", type=int, default=0)
    p.add_argument("--ann_index", choices=["real","nominal"], default="real")
    # Data/profile/tag
    p.add_argument("--bands", choices=["on","off"], default="on")
    p.add_argument("--data_window")
    p.add_argument("--data_profile", choices=["dev","full"])
    p.add_argument("--tag")
    # Allocation & FX hedge
    p.add_argument("--alpha_mix")
    p.add_argument("--alpha_kr", type=float)
    p.add_argument("--alpha_us", type=float)
    p.add_argument("--alpha_au", type=float)
    p.add_argument("--h_FX", type=float)
    p.add_argument("--h_fx", type=float, help="alias for --h_FX")
    p.add_argument("--fx_hedge_cost", type=float)
    # Action-layer bias
    p.add_argument("--bh_on", choices=["on","off"], default="off")
    p.add_argument("--la_k", type=float, default=1.0)
    p.add_argument("--habit_phi", type=float, default=0.0)
    p.add_argument("--bias_on", choices=["on","off"], default="off")
    p.add_argument("--bias_loss_aversion", type=float, default=0.0)
    p.add_argument("--bias_prob_gamma", type=float, default=1.0)
    p.add_argument("--bias_myopia", type=float, default=0.0)
    p.add_argument("--bias_w_floor", type=float, default=0.0)
    p.add_argument("--bias_w_cap_shock", type=float, default=0.0)
    # stdout control
    p.add_argument("--print_mode", choices=["full","metrics","summary"], default="full")
    p.add_argument("--metrics_keys", default="EW,ES95,EL,Ruin,mean_WT,es_mode,es95_source,EU,EU_per_year,delta_annual,F_target_used")
    p.add_argument("--no_paths", action="store_true")
    p.add_argument("--validate", choices=["on","off"], default="off")
    p.add_argument("--return_actor", choices=["on","off"], default="off")
    # ETA
    p.add_argument("--eta_mode", choices=["off","history"], default="history")
    p.add_argument("--eta_budget_hms")
    p.add_argument("--eta_budget_s", type=float)
    p.add_argument("--eta_hard_stop", choices=["on","off"], default="on")
    p.add_argument("--eta_db")
    # Eval-time randomness
    p.add_argument("--eval_seed_jitter", choices=["on","off"], default="off")
    return p

def _apply_data_profile_defaults(args) -> None:
    if getattr(args,"data_profile",None) and not getattr(args,"market_csv",None):
        base = os.path.normpath(os.path.join(os.path.dirname(__file__),"..","data","market"))
        if args.data_profile=="dev":
            args.market_csv = os.path.abspath(os.path.join(base,"kr_us_gold_bootstrap_mini.csv"))
        elif args.data_profile=="full":
            args.market_csv = os.path.abspath(os.path.join(base,"kr_us_gold_bootstrap_full.csv"))

def _validate_args(args) -> None:
    if args.data_window is not None:
        s = str(args.data_window).strip()
        if s and not _WINDOW_RE.match(s):
            raise SystemExit(f"--data_window 형식 오류: '{args.data_window}' (예: 2005-01:2020-12)")
    if args.market_mode=="bootstrap" and not (args.market_csv or args.data_profile):
        raise SystemExit("market_mode=bootstrap 시 --market_csv 또는 --data_profile(dev|full) 필요.")
    if args.method=="rule" and not args.baseline:
        raise SystemExit("method=rule 시 --baseline (4pct|cpb|vpw|kgr) 필수.")

    def _warn(msg:str): print(f"[warn] {msg}", file=sys.stderr, flush=True)
    if args.beta is not None:
        try:
            b = float(args.beta)
            if not (0.0 < b <= 1.0): _warn(f"--beta 권장범위 벗어남: {b}")
        except Exception: _warn("--beta 파싱 실패")
    try:
        if float(args.la_k) < 0.0: _warn(f"--la_k 음수: {args.la_k}")
    except Exception: _warn("--la_k 파싱 실패")
    if args.habit_phi is not None:
        try:
            hp = float(args.habit_phi)
            if not (0.0 <= hp <= 1.0): _warn(f"--habit_phi 권장범위(0~1) 벗어남: {hp}")
        except Exception: _warn("--habit_phi 파싱 실패")
    for name in ["bias_loss_aversion","bias_myopia","bias_w_cap_shock"]:
        v = getattr(args,name,0.0)
        try:
            if float(v) < 0.0: _warn(f"--{name} 음수 비권장: {v}")
        except Exception: _warn(f"--{name} 파싱 실패")
    try:
        if float(args.bias_prob_gamma) <= 0.0: _warn(f"--bias_prob_gamma > 0 필요: {args.bias_prob_gamma}")
    except Exception: _warn("--bias_prob_gamma 파싱 실패")
    if args.bias_w_floor is not None:
        try:
            wf = float(args.bias_w_floor)
            if not (0.0 <= wf <= 1.0): _warn(f"--bias_w_floor 0~1 범위 벗어남: {wf}")
        except Exception: _warn("--bias_w_floor 파싱 실패")

def _inject_meta(out: Dict[str,Any], args) -> None:
    if not isinstance(out, dict): return
    meta = out.setdefault("meta", {})
    meta.setdefault("tag", getattr(args,"tag",None))
    meta.setdefault("method", getattr(args,"method",None))
    meta.setdefault("asset", getattr(args,"asset",None))
    meta.setdefault("outputs_abs", getattr(args,"outputs",None))
    try:
        if parse_behavioral_from_args and _bh_describe:
            _spec = parse_behavioral_from_args(args)
            meta["behavioral"] = _bh_describe(_spec)
        else:
            meta.setdefault("behavioral", {
                "bh_on": getattr(args,"bh_on","off"),
                "la_k": getattr(args,"la_k",1.0),
                "beta": getattr(args,"beta",None),
                "habit_phi": getattr(args,"habit_phi",0.0),
            })
        meta.setdefault("behavioral_bias", {
            "bias_on": getattr(args,"bias_on","off"),
            "loss_aversion": getattr(args,"bias_loss_aversion",0.0),
            "prob_gamma": getattr(args,"bias_prob_gamma",1.0),
            "myopia": getattr(args,"bias_myopia",0.0),
            "w_floor": getattr(args,"bias_w_floor",0.0),
            "w_cap_shock": getattr(args,"bias_w_cap_shock",0.0),
        })
        meta.setdefault("verbose", getattr(args,"verbose","off"))
    except Exception:
        pass
    if _cfg_hash_fn:
        try: meta["config_hash"] = _cfg_hash_fn(vars(args))
        except Exception: pass
    if _ver_info_fn:
        try:
            vi = _ver_info_fn()
            if isinstance(vi, dict):
                meta.update({k: vi.get(k) for k in ("git_commit","py_ver","np_ver")})
        except Exception: pass

def _route_mode(args)->str:
    m = str(getattr(args,"mode","auto")).lower()
    if m in ("once","rl","calib"): return m
    if str(getattr(args,"calib","off")).lower()=="on": return "calib"
    if str(getattr(args,"method","hjb")).lower()=="rl": return "rl"
    return "once"

def _run_core(args)->Dict[str,Any]|Any:
    want_paths = (str(getattr(args,"print_mode","full")).lower()=="full") and (not getattr(args,"no_paths",False))
    route = _route_mode(args)
    if route=="calib":
        out = calibrate_lambda(args)
        if isinstance(out,dict) and "es_mode" not in out:
            out["es_mode"] = str(getattr(args,"es_mode","wealth")).lower()
    elif route=="rl":
        out = run_rl(args)
    else:
        res = run_once(args)
        out = maybe_evaluate_with_es_mode(res, es_mode=getattr(args,"es_mode","wealth"), want_paths=want_paths)
    if isinstance(out,dict):
        out.setdefault("tag", getattr(args,"tag",None))
        out.setdefault("method", getattr(args,"method",None))
        out.setdefault("asset", getattr(args,"asset",None))
        out.setdefault("outputs_abs", getattr(args,"outputs",None))
        _inject_meta(out,args)
    try:
        if isinstance(out,dict):
            out = fixup_metrics_with_cvar(args,out)
    except Exception as _e:
        try:
            if isinstance(out,dict):
                tgt = out["metrics"] if "metrics" in out and isinstance(out["metrics"],dict) else out
                tgt["es95_note"] = f"post-fixup failed: {type(_e).__name__}"
        except Exception: pass
    try:
        if isinstance(out,dict) and isinstance(out.get("extra"),dict):
            ew = out["extra"].get("eval_WT")
            if isinstance(ew,(list,tuple)):
                out["extra"]["eval_WT_n"] = len(ew)
    except Exception: pass
    return out

def main():
    t0 = time.perf_counter()
    p = _build_arg_parser()
    args = p.parse_args()
    # normalize outputs
    args.outputs = _normalize_outputs_path(getattr(args,"outputs",None))
    # seeds
    try:
        if getattr(args,"seed",None) is not None:
            args.seeds = [int(args.seed)]
        else:
            args.seeds = _normalize_seeds(list(getattr(args,"seeds",[])))
    except Exception:
        args.seeds = [0]
    parsed_w_grid = _csv_floats(getattr(args,"hjb_w_grid",None))
    if parsed_w_grid is not None: args.hjb_w_grid = parsed_w_grid
    if getattr(args,"h_FX",None) is None and getattr(args,"h_fx",None) is not None:
        args.h_FX = args.h_fx
    try:
        args.bootstrap_block = _parse_block(getattr(args,"bootstrap_block",None))
    except argparse.ArgumentTypeError as e:
        sys.exit(f"--bootstrap_block 오류: {e}")
    _apply_data_profile_defaults(args)
    _validate_args(args)

    # ETA preview (stderr)
    try:
        if str(getattr(args,"eta_mode","history")).lower()=="history":
            db = eta_load_db(eta_db_path(args))
            eta_s, src = predict_eta_from_history(args, db)
            if eta_s is not None:
                print(f"[ETA] ~{fmt_hms(eta_s)} (source={src}) starting", file=sys.stderr, flush=True)
                budget = getattr(args,"eta_budget_s",None) or parse_hms_to_seconds(getattr(args,"eta_budget_hms",None))
                if budget is not None and eta_s > float(budget):
                    if str(getattr(args,"eta_hard_stop","on")).lower()=="on":
                        print(f"[ETA] exceeds budget {fmt_hms(budget)} -> abort.", file=sys.stderr, flush=True)
                        sys.exit(3)
                    else:
                        print(f"[ETA] exceeds budget {fmt_hms(budget)} -> continue.", file=sys.stderr, flush=True)
    except Exception as _e:
        print(f"[ETA] predictor skipped ({type(_e).__name__})", file=sys.stderr, flush=True)

    # JSON-only stdout
    captured_stdout = io.StringIO()
    with contextlib.redirect_stdout(captured_stdout):
        out = _run_core(args)

    elapsed = time.perf_counter() - t0
    if isinstance(out,dict):
        out["time_total_s"] = round(elapsed, 3)
        out["time_total_hms"] = fmt_hms(elapsed)

    # ── 공통 메타 패커: summary/metrics에서 사용
    def _meta_pack(_out_dict: dict) -> dict:
        _m = _out_dict.get("metrics", {})
        if not isinstance(_m, dict):
            _m = _out_dict
        return {
            "tag": getattr(args, "tag", None),
            "asset": getattr(args, "asset", None),
            "method": getattr(args, "method", None),
            "age0": getattr(args, "age0", None),
            "sex": getattr(args, "sex", None),
            "n_paths": getattr(args, "n_paths", None),
            "metrics": _m,
        }

    pmode = str(getattr(args,"print_mode","full")).lower()

    if pmode == "metrics" and isinstance(out,dict):
        m = dict(out.get("metrics", {}))
        keys = [s.strip() for s in str(getattr(args,"metrics_keys","")).split(",") if s.strip()]
        if keys:
            m = {k: m.get(k, None) for k in keys}
        # 평평한 dict + 메타 동봉
        packed = {
            "tag": getattr(args, "tag", None),
            "asset": getattr(args, "asset", None),
            "method": getattr(args, "method", None),
            "n_paths": getattr(args, "n_paths", None),
            **m,
        }
        _safe_print_json(packed)
        return

    if pmode == "summary" and isinstance(out,dict):
        # 테스트 기대 스키마로 JSON 출력
        _safe_print_json(_meta_pack(out))
        return

    # full (기본)
    to_print = prune_for_stdout(args, out) if isinstance(out,dict) else out
    _safe_print_json(to_print)

if __name__ == "__main__":
    main()
