# project/runner/cli.py
from __future__ import annotations

import argparse
import json
import os
import re
import time
import sys
import io
import contextlib
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
    eta_db_path, eta_load_db, predict_eta_from_history, eta_record,  # noqa: F401
)
from .cvar_utils import fixup_metrics_with_cvar
from .pack_utils import prune_for_stdout, maybe_evaluate_with_es_mode

# Behavioral spec echo (메타 기록용)
try:
    from ..policy.behavioral import parse_behavioral_from_args, describe as _bh_describe  # type: ignore
except Exception:
    parse_behavioral_from_args = None  # type: ignore
    _bh_describe = None  # type: ignore

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
    os.makedirs(os.path.join(abs_p, "_logs"), exist_ok=True)
    return abs_p


def _normalize_seeds(seeds_arg: List[int]) -> List[int]:
    if not isinstance(seeds_arg, list) or len(seeds_arg) == 0:
        return [0, 1, 2, 3, 4]
    if len(seeds_arg) == 1:
        return [int(seeds_arg[0])]
    return sorted({int(x) for x in seeds_arg})


def _print_metrics_summary(metrics: dict) -> None:
    lines = [
        f"EW            : {metrics.get('EW')}",
        f"ES95          : {metrics.get('ES95')}",
        f"EL            : {metrics.get('EL')}",
        f"Ruin          : {metrics.get('Ruin')}",
        f"mean_WT       : {metrics.get('mean_WT')}",
        f"es95_source   : {metrics.get('es95_source')}",
        f"EU            : {metrics.get('EU')}",
        f"EU_per_year   : {metrics.get('EU_per_year')}",
        f"delta_annual  : {metrics.get('delta_annual')}",
        f"F_target_used : {metrics.get('F_target_used')}",
    ]
    print("\n".join([s for s in lines if s is not None]))


def _safe_print_json(obj: Any) -> None:
    """
    최종 stdout에 **오직 JSON 하나**만 출력.
    - ensure_ascii=False: 한글 보존
    - sort_keys=True: metrics 모드 같은 비교 시 안정적
    """
    try:
        s = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except TypeError:
        s = json.dumps(_json_fallback(obj), ensure_ascii=False, sort_keys=True)
    print(s)


def _json_fallback(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _json_fallback(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_fallback(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def _parse_block(s: str | int | None) -> int:
    """
    '6m', '12', '90d', '2y' 등을 월 단위 정수로 변환.
    - m: months (기본)
    - y: years  (×12)
    - d: days   (~30일=1개월로 근사)
    """
    if s is None:
        return 24  # 기존 기본값 유지
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
        # 30일 ≈ 1개월 근사
        return max(1, round(n / 30))
    raise argparse.ArgumentTypeError(f"invalid unit in block spec: {s}")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    # --- High-level execution mode ---
    p.add_argument("--mode", type=str, default="auto", choices=["auto", "once", "rl", "calib"])

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
    # Seeds
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--seed", type=int, default=None, help="단일 시드(지정 시 seeds=[seed]로 사용)")
    p.add_argument("--n_paths", type=int, default=100)

    # ⚠️ ES 정의 안내
    p.add_argument("--es_mode", type=str, default="wealth", choices=["wealth", "loss"],
                   help="ES95는 통일(손실 = −W_T). 'loss' 선택 시 EL만 추가 보고.")

    p.add_argument("--outputs", type=str, default="./outputs")

    # HJB
    p.add_argument("--hjb_W_grid", type=int, default=None)
    p.add_argument("--hjb_Nshock", type=int, default=None)
    p.add_argument("--hjb_eta_n", type=int, default=None)
    p.add_argument("--hjb_w_grid", type=str, default=None,
                   help="Comma-separated w grid for HJB (e.g. '0.10,0.20,0.30,0.40,0.50').")
    p.add_argument("--w_min_dev", type=float, default=None,
                   help="(dev) Minimum risky weight; drop w < w_min_dev from HJB action set.")

    # Hedge (legacy logical hedge — 별개)
    p.add_argument("--hedge", choices=["on", "off"], default="off")
    p.add_argument("--hedge_mode", choices=["mu", "sigma", "downside"], default="sigma")
    p.add_argument("--hedge_cost", type=float, default=0.005)
    p.add_argument("--hedge_sigma_k", type=float, default=0.20)
    p.add_argument("--hedge_tx", type=float, default=0.0)

    # Market
    p.add_argument("--market_mode", choices=["iid", "bootstrap"], default="iid")
    p.add_argument("--market_csv", type=str, default=None)
    # ▼ 변경: 문자열 입력 허용('6m','12','90d','2y' 등)
    p.add_argument("--bootstrap_block", type=str, default="24",
                   help="블록 길이. 예: '6m', '12'(개월), '90d'(일→월 근사), '2y'(년→월). 내부적으로 월 단위 정수로 변환.")
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

    # autosave (run_once 내부/평가단에서 처리)
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

    # Lite overrides / Utility reporting
    p.add_argument("--q_floor", type=float, default=None)
    p.add_argument("--beta", type=float, default=None,
                   help="(utility present-bias) 0<beta<=1 권장. 효용층 현재편향 강도.")
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

    # QUIET / VERBOSE
    p.add_argument("--quiet", choices=["on", "off"], default="on")
    p.add_argument("--verbose", choices=["on", "off"], default="off",
                   help="콘솔 로그를 조금 더 상세히(편향 적용 로그 등).")

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
    p.add_argument("--h_fx", type=float, default=None, help="--h_FX alias")
    p.add_argument("--fx_hedge_cost", type=float, default=None)

    # Behavioral (utility-layer)
    p.add_argument("--bh_on", choices=["on", "off"], default="off",
                   help="효용 기반 편향 토글(손실가중/현재편향/습관효용 등).")
    p.add_argument("--la_k", type=float, default=1.0,
                   help="손실가중 λ (권장 >=1.0).")
    p.add_argument("--habit_phi", type=float, default=0.0,
                   help="습관효용 가중 φ (보통 0~1).")

    # Behavioral Bias (action-layer)
    p.add_argument("--bias_on", choices=["on", "off"], default="off",
                   help="행동 출력 편향(행동/결정 레이어 후처리) 토글.")
    p.add_argument("--bias_loss_aversion", type=float, default=0.0,
                   help="최근 손실 신호에서 w 축소 강도 (>=0).")
    p.add_argument("--bias_prob_gamma", type=float, default=1.0,
                   help="확률 왜곡 γ (>0, 1이면 중립; <1이면 꼬리 과대평가).")
    p.add_argument("--bias_myopia", type=float, default=0.0,
                   help="근시 편향 강도: q를 약간 상향 (>=0).")
    p.add_argument("--bias_w_floor", type=float, default=0.0,
                   help="행동층 risky 최소비중 하한 (0~1).")
    p.add_argument("--bias_w_cap_shock", type=float, default=0.0,
                   help="변동성 신호 기반 추가 w 캡 강도 (>=0).")

    # stdout control
    p.add_argument("--print_mode", choices=["full", "metrics", "summary"], default="full")
    p.add_argument("--metrics_keys", type=str,
                   default="EW,ES95,EL,Ruin,mean_WT,es_mode,es95_source,EU,EU_per_year,delta_annual,F_target_used")
    p.add_argument("--no_paths", action="store_true")
    p.add_argument("--validate", choices=["on","off"], default="off")
    p.add_argument("--return_actor", choices=["on", "off"], default="off")

    # ETA
    p.add_argument("--eta_mode", choices=["off", "history"], default="history")
    p.add_argument("--eta_budget_hms", type=str, default=None)
    p.add_argument("--eta_budget_s", type=float, default=None)
    p.add_argument("--eta_hard_stop", choices=["on", "off"], default="on")
    p.add_argument("--eta_db", type=str, default=None)

    # Eval-time randomness
    p.add_argument("--eval_seed_jitter", choices=["on", "off"], default="off",
                   help="on이면 평가 에피소드 시드에 시각 기반 소량 지터를 더해 매 실행 시 약간 다른 결과가 나오게 함.")

    return p


def _apply_data_profile_defaults(args) -> None:
    if getattr(args, "data_profile", None) and not getattr(args, "market_csv", None):
        base = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "market"))
        if args.data_profile == "dev":
            args.market_csv = os.path.abspath(os.path.join(base, "kr_us_gold_bootstrap_mini.csv"))
        elif args.data_profile == "full":
            args.market_csv = os.path.abspath(os.path.join(base, "kr_us_gold_bootstrap_full.csv"))


def _validate_args(args) -> None:
    # data_window 형식
    if args.data_window is not None:
        s = str(args.data_window).strip()
        if s and not _WINDOW_RE.match(s):
            raise SystemExit(f"--data_window 형식 오류: '{args.data_window}'. 예: 2005-01:2020-12 또는 '2005-01:' / ':2020-12'")
    # bootstrap 입력
    if args.market_mode == "bootstrap":
        if not args.market_csv and not args.data_profile:
            raise SystemExit("market_mode=bootstrap 사용 시 --market_csv 또는 --data_profile(dev|full) 필요.")
    # rule baseline 필수
    if args.method == "rule" and not args.baseline:
        raise SystemExit("method=rule 사용 시 --baseline (4pct|cpb|vpw|kgr) 필수.")

    # --- Behavioral sanity checks (경고 위주) ---
    def _warn(msg: str):
        print(f"[warn] {msg}", file=sys.stderr, flush=True)

    # beta: 권장 0<β<=1
    if args.beta is not None:
        try:
            b = float(args.beta)
            if not (0.0 < b <= 1.0):
                _warn(f"--beta 권장범위 벗어남 (got {b}). 보통 0<beta<=1.")
        except Exception:
            _warn("--beta 파싱 실패")

    # la_k: 권장 >=1.0
    try:
        if float(args.la_k) < 0.0:
            _warn(f"--la_k 음수({args.la_k})는 비권장. 보통 λ>=1.0.")
    except Exception:
        _warn("--la_k 파싱 실패")

    # habit_phi: 권장 0~1
    if args.habit_phi is not None:
        try:
            hp = float(args.habit_phi)
            if not (0.0 <= hp <= 1.0):
                _warn(f"--habit_phi 권장범위(0~1) 벗어남 (got {hp}).")
        except Exception:
            _warn("--habit_phi 파싱 실패")

    # --- Behavioral Bias (action-layer) sanity checks ---
    for name in ["bias_loss_aversion", "bias_myopia", "bias_w_cap_shock"]:
        v = getattr(args, name, 0.0)
        try:
            if float(v) < 0.0:
                _warn(f"--{name} 음수는 비권장 (got {v}). 0 이상 사용을 권장.")
        except Exception:
            _warn(f"--{name} 파싱 실패")

    try:
        if float(args.bias_prob_gamma) <= 0.0:
            _warn(f"--bias_prob_gamma > 0 이어야 함 (got {args.bias_prob_gamma}).")
    except Exception:
        _warn("--bias_prob_gamma 파싱 실패")

    if args.bias_w_floor is not None:
        try:
            wf = float(args.bias_w_floor)
            if not (0.0 <= wf <= 1.0):
                _warn(f"--bias_w_floor 권장범위(0~1) 벗어남 (got {wf}).")
        except Exception:
            _warn("--bias_w_floor 파싱 실패")


def _inject_meta(out: Dict[str, Any], args) -> None:
    if not isinstance(out, dict):
        return
    meta = out.setdefault("meta", {})
    meta.setdefault("tag", getattr(args, "tag", None))
    meta.setdefault("method", getattr(args, "method", None))
    meta.setdefault("asset", getattr(args, "asset", None))
    meta.setdefault("outputs_abs", getattr(args, "outputs", None))

    # 행동/효용 편향 스펙 메타 기록 (안전하게)
    try:
        if parse_behavioral_from_args and _bh_describe:
            _spec = parse_behavioral_from_args(args)
            meta["behavioral"] = _bh_describe(_spec)
        else:
            meta.setdefault("behavioral", {
                "bh_on": getattr(args, "bh_on", "off"),
                "la_k": getattr(args, "la_k", 1.0),
                "beta": getattr(args, "beta", None),
                "habit_phi": getattr(args, "habit_phi", 0.0),
            })
        meta.setdefault("behavioral_bias", {
            "bias_on": getattr(args, "bias_on", "off"),
            "loss_aversion": getattr(args, "bias_loss_aversion", 0.0),
            "prob_gamma": getattr(args, "bias_prob_gamma", 1.0),
            "myopia": getattr(args, "bias_myopia", 0.0),
            "w_floor": getattr(args, "bias_w_floor", 0.0),
            "w_cap_shock": getattr(args, "bias_w_cap_shock", 0.0),
        })
        meta.setdefault("verbose", getattr(args, "verbose", "off"))
    except Exception:
        pass

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


def _route_mode(args) -> str:
    """
    mode 해석:
      - explicit: once/rl/calib이면 그대로
      - auto: method와 calib을 참고
        * calib=on → calib
        * method=rl → rl
        * 그 외 → once
    """
    mode = str(getattr(args, "mode", "auto")).lower()
    if mode in ("once", "rl", "calib"):
        return mode
    if str(getattr(args, "calib", "off")).lower() == "on":
        return "calib"
    if str(getattr(args, "method", "hjb")).lower() == "rl":
        return "rl"
    return "once"


def _run_core(args) -> Dict[str, Any] | Any:
    """실제 실행 경로(학습/평가/보정)를 분리 — stdout은 여기서 절대 출력하지 않음."""
    want_paths = (str(getattr(args, "print_mode", "full")).lower() == "full") and (not getattr(args, "no_paths", False))

    route = _route_mode(args)

    if route == "calib":
        out = calibrate_lambda(args)
        if isinstance(out, dict) and "es_mode" not in out:
            out["es_mode"] = str(getattr(args, "es_mode", "wealth")).lower()
    elif route == "rl":
        out = run_rl(args)
    else:  # "once"
        res = run_once(args)
        out = maybe_evaluate_with_es_mode(res, es_mode=getattr(args, "es_mode", "wealth"), want_paths=want_paths)

    # meta
    if isinstance(out, dict):
        out.setdefault("tag", getattr(args, "tag", None))
        out.setdefault("method", getattr(args, "method", None))
        out.setdefault("asset", getattr(args, "asset", None))
        out.setdefault("outputs_abs", getattr(args, "outputs", None))
        _inject_meta(out, args)

    # ES95(CVaR) fixup
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

    return out


def main():
    t0 = time.perf_counter()
    p = _build_arg_parser()
    args = p.parse_args()

    # normalize outputs
    args.outputs = _normalize_outputs_path(getattr(args, "outputs", None))

    # seeds normalize
    try:
        if getattr(args, "seed", None) is not None:
            args.seeds = [int(args.seed)]
        else:
            args.seeds = _normalize_seeds(list(getattr(args, "seeds", [])))
    except Exception:
        args.seeds = [0]

    parsed_w_grid = _csv_floats(getattr(args, "hjb_w_grid", None))
    if parsed_w_grid is not None:
        args.hjb_w_grid = parsed_w_grid

    # alias: h_fx → h_FX (미지정 시에만)
    if getattr(args, "h_FX", None) is None and getattr(args, "h_fx", None) is not None:
        args.h_FX = args.h_fx

    # ▼ bootstrap_block 파싱 (문자열 → 월 단위 정수)
    try:
        args.bootstrap_block = _parse_block(getattr(args, "bootstrap_block", None))
    except argparse.ArgumentTypeError as e:
        sys.exit(f"--bootstrap_block 오류: {e}")

    _apply_data_profile_defaults(args)
    _validate_args(args)

    # ETA budget check (stderr만 사용)
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

    # -------- 핵심: **모든 실행 구간의 stdout 캡처** (순수 JSON 보장) --------
    captured_stdout = io.StringIO()
    with contextlib.redirect_stdout(captured_stdout):
        out = _run_core(args)

    # total time
    elapsed = time.perf_counter() - t0
    if isinstance(out, dict):
        out["time_total_s"] = round(elapsed, 3)
        out["time_total_hms"] = fmt_hms(elapsed)

    # summary 모드
    if str(getattr(args, "print_mode", "full")).lower() == "summary":
        if isinstance(out, dict):
            metrics = out["metrics"] if "metrics" in out and isinstance(out["metrics"], dict) else out
            _print_metrics_summary(metrics)
        else:
            print("[warn] summary mode but no metrics dict available.")
        return

    # stdout (JSON only)
    if str(getattr(args, "print_mode", "full")).lower() == "metrics" and isinstance(out, dict):
        m = out.get("metrics", {})
        keys = [s.strip() for s in str(getattr(args, "metrics_keys","")).split(",") if s.strip()]
        if keys:
            m = {k: m.get(k, None) for k in keys}
        _safe_print_json(m)
    else:
        to_print = prune_for_stdout(args, out) if isinstance(out, dict) else out
        _safe_print_json(to_print)

    # 참고: captured_stdout.getvalue()에는 내부 모듈이 찍은 stdout이 담김.


if __name__ == "__main__":
    main()
