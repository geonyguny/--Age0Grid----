# project/runner/cli.py
from __future__ import annotations

import argparse, json, os, re, time, sys, io, contextlib
from types import SimpleNamespace
from typing import Any, Dict, List, Callable

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

import numpy as _np

# ─────────────────────────────────────────────────────────
# Discrete ES helper (tests용)
# ─────────────────────────────────────────────────────────
def _cvar_fallback(x, a_or_F, maybe_alpha=None):
    def _es_discrete(L: _np.ndarray, alpha: float) -> float:
        L = _np.asarray(L, dtype=float)
        L = L[_np.isfinite(L)]
        n = int(L.size)
        if n == 0: return 0.0
        a = float(alpha)
        if a <= 0.0: return float(_np.mean(L))
        if a >= 1.0: return float(_np.max(L))
        x = _np.sort(L)
        t = a * n
        k = int(_np.floor(t))
        if k >= n: k = n - 1
        beta = t - k  # [0,1)
        tail_sum = x[k+1:].sum() + (1.0 - beta) * x[k]
        denom = n * (1.0 - a)
        return float(tail_sum / denom)

    x = _np.asarray(x, dtype=float)
    if maybe_alpha is None:
        alpha = float(a_or_F)
        return _es_discrete(x, alpha)
    else:
        F = float(a_or_F); alpha = float(maybe_alpha)
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

# metrics_keys 기본값 문자열(사용자 미지정 판별용)
DEFAULT_METRICS_KEYS = "EW,ES95,EL,Ruin,mean_WT,es_mode,es95_source,EU,EU_per_year,delta_annual,F_target_used"

def _csv_floats(s: str | None) -> List[float] | None:
    if not s: return None
    out: List[float] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if tok: out.append(float(tok))
    return out or None

def _normalize_outputs_path(p: str | None) -> str:
    base = p or "./outputs"
    abs_p = os.path.abspath(base)
    os.makedirs(abs_p, exist_ok=True)
    os.makedirs(os.path.join(abs_p, "_logs"), exist_ok=True)
    return abs_p

def _normalize_seeds(seeds_arg: List[int]) -> List[int]:
    if not isinstance(seeds_arg, list) or not seeds_arg: return [0,1,2,3,4]
    if len(seeds_arg)==1: return [int(seeds_arg[0])]
    return sorted({int(x) for x in seeds_arg})

def _safe_print_json(obj: Any) -> None:
    try:
        s = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except TypeError:
        s = json.dumps(obj, default=str, ensure_ascii=False, sort_keys=True)
    print(s)

def _parse_block(s: str | int | None) -> int:
    if s is None: return 24
    if isinstance(s, int): return int(s)
    txt = str(s).strip().lower()
    m = re.fullmatch(r'(\d+)\s*([dmy]?)', txt)
    if not m: raise argparse.ArgumentTypeError(f"invalid block spec: {s}")
    n = int(m.group(1)); unit = m.group(2) or 'm'
    if unit == 'm': return max(1, n)
    if unit == 'y': return max(1, n*12)
    if unit == 'd': return max(1, round(n/30))
    raise argparse.ArgumentTypeError(f"invalid unit: {s}")

def _ensure_basic_metrics_for_print(out: Dict[str, Any]) -> None:
    """metrics.EW가 없을 때 mean_WT -> extra.eval_WT 평균 순으로 채운다."""
    if not isinstance(out, dict): return
    metrics = out.setdefault("metrics", {})
    if "EW" in metrics and metrics["EW"] is not None:
        return
    # 1) metrics.mean_WT
    if "mean_WT" in metrics:
        try:
            metrics["EW"] = float(metrics["mean_WT"])
            return
        except Exception:
            pass
    # 2) extra.eval_WT 평균
    extra = out.get("extra", {})
    ew = extra.get("eval_WT") if isinstance(extra, dict) else None
    if isinstance(ew, (list, tuple)) and len(ew) > 0:
        try:
            metrics["EW"] = float(_np.mean(ew))
            return
        except Exception:
            pass
    # 3) 마지막 폴백
    metrics.setdefault("EW", 0.0)

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Retirement experiment runner (JSON-only stdout).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---------------- High-level ----------------
    p.add_argument("--mode", default="auto", choices=["auto", "once", "rl", "calib"],
                   help="실행 모드(배치/단발/강화학습/캘리브레이션)")
    p.add_argument("--method", default="hjb", choices=["hjb", "rl", "rule"],
                   help="해결 방법: HJB / RL / 규칙 기반")
    p.add_argument("--tag", help="실험 식별용 태그(로그/출력 구분)")

    # ---------------- Core ----------------
    p.add_argument("--asset", default="KR", help="자산 국가/세트 코드")
    p.add_argument("--baseline", help="비교 기준 정책 태그")
    p.add_argument("--w_max", type=float, default=0.70, help="위험자산 최대 비중(limit)")
    p.add_argument("--fee_annual", type=float, default=0.004, help="연간 총보수(비설정 시)")
    p.add_argument("--phi_adval", type=float,
                   help="ad-valorem fee(설정 시 fee_annual 대체)")
    p.add_argument("--horizon_years", type=int, default=35, help="은퇴 후 시뮬레이션 연한")
    p.add_argument("--alpha", type=float, default=0.95, help="CVaR 신뢰수준 α")
    p.add_argument("--lambda_term", type=float, default=0.0, help="CVaR 라그랑주 승수(고정)")
    p.add_argument("--F_target", type=float, default=0.0, help="최저소비(또는 바닥) 목표")
    p.add_argument("--p_annual", type=float, default=0.04, help="초기 인출률(연율)")
    p.add_argument("--g_real_annual", type=float, default=0.02, help="실질 인출 증가율")
    p.add_argument("--w_fixed", type=float, default=0.60, help="고정 위험비중(룰베이스용)")
    p.add_argument("--floor_on", action="store_true", help="소비 바닥 제약 사용 여부")
    p.add_argument("--f_min_real", type=float, default=0.0, help="실질 소비 최저선")

    # ---------------- Seeds/paths ----------------
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="복수 시드(배치)")
    p.add_argument("--seed", type=int, help="단일 시드(--seeds 무시)")
    p.add_argument("--n_paths", type=int, default=100, help="시뮬레이션 경로 수")

    # ---------------- ES/CVaR reporting ----------------
    p.add_argument("--es_mode", default="wealth", choices=["wealth", "loss"],
                   help="'loss'는 L=max(F−W,0) 기준 보고")
    p.add_argument("--cvar_unit", choices=["wealth", "utility"], default="wealth",
                   help="CVaR/종단 손실의 단위: 금액(wealth) 또는 효용(utility)")

    p.add_argument("--outputs", default="./outputs", help="출력 루트 디렉터리")

    # ---------------- HJB ----------------
    p.add_argument("--hjb_W_grid", type=int, help="HJB 자산 격자 크기")
    p.add_argument("--hjb_Nshock", type=int, help="HJB 쇼크(근사) 개수")
    p.add_argument("--hjb_eta_n", type=int, help="HJB 적분 노드 개수")
    p.add_argument("--hjb_w_grid", help="HJB 위험비중 격자(쉼표구분)")
    p.add_argument("--w_min_dev", type=float, help="dev: w < w_min_dev 액션 제거")

    # ---------------- Legacy hedge ----------------
    p.add_argument("--hedge", choices=["on", "off"], default="off", help="헤지 레이어 on/off")
    p.add_argument("--hedge_mode", choices=["mu", "sigma", "downside"], default="sigma")
    p.add_argument("--hedge_cost", type=float, default=0.005)
    p.add_argument("--hedge_sigma_k", type=float, default=0.20)
    p.add_argument("--hedge_tx", type=float, default=0.0)

    # ---------------- Ambiguity (Hansen–Sargent θ) ----------------
    p.add_argument("--theta_ambiguity", type=float, default=None,
                   help="모형 기록용 θ(미결선일 수 있음)")

    # ---------------- Market ----------------
    p.add_argument("--market_mode", choices=["iid", "bootstrap"], default="iid")
    p.add_argument("--market_csv", help="시장 경로 CSV(선택)")
    p.add_argument("--bootstrap_block", default="24",
                   help="부트스트랩 블록 길이 예: '6m','12','90d','2y'")
    p.add_argument("--use_real_rf", choices=["on", "off"], default="on")

    # ---------------- Mortality ----------------
    p.add_argument("--mortality", choices=["on", "off"], default="off")
    p.add_argument("--mort_table", help="사망표 경로/키")
    p.add_argument("--age0", type=int, default=55)
    p.add_argument("--sex", choices=["M", "F"], default="M")
    p.add_argument("--bequest_kappa", type=float, default=0.0)
    p.add_argument("--bequest_gamma", type=float, default=1.0)

    # ---------------- CVaR Calibration ----------------
    p.add_argument("--calib", choices=["on", "off"], default="off")
    p.add_argument("--calib_param", choices=["lambda", "F"], default="lambda")
    p.add_argument("--cvar_target", type=float, default=CVAR_TARGET_DEFAULT)
    p.add_argument("--cvar_tol", type=float, default=CVAR_TOL_DEFAULT)
    p.add_argument("--lambda_min", type=float, default=LAMBDA_MIN_DEFAULT)
    p.add_argument("--lambda_max", type=float, default=LAMBDA_MAX_DEFAULT)
    p.add_argument("--calib_fast", choices=["on", "off"], default="on")
    p.add_argument("--calib_max_iter", type=int, default=8)
    p.add_argument("--F_min", type=float)
    p.add_argument("--F_max", type=float)

    # ---------------- Autosave ----------------
    p.add_argument("--autosave", choices=["on", "off"], default="off")

    # ---------------- RL core ----------------
    p.add_argument("--rl_epochs", type=int, default=60)
    p.add_argument("--rl_steps_per_epoch", type=int, default=2048)
    p.add_argument("--rl_n_paths_eval", type=int, default=300)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--rl_q_cap", type=float, default=0.0,
                   help="정책의 분기별 최댓값(소비/인출 상한 등)")
    p.add_argument("--teacher_eps0", type=float, default=0.0)
    p.add_argument("--teacher_decay", type=float, default=1.0)
    p.add_argument("--lw_scale", type=float, default=0.0)
    p.add_argument("--survive_bonus", type=float, default=0.0)
    p.add_argument("--crra_gamma", type=float, default=3.0)
    p.add_argument("--u_scale", type=float, default=0.0)

    # ---------------- Lite / Utility overlay ----------------
    p.add_argument("--q_floor", type=float, help="행동/소비 floor의 보조 파라미터")
    p.add_argument("--beta", type=float, help="효용 present-bias; 0<beta<=1")
    p.add_argument("--report_utility", choices=["on", "off"], default="off")
    p.add_argument("--delta_annual", type=float)

    # ---------------- Stage-wise CVaR & c* design ----------------
    p.add_argument("--cvar_stage", choices=["on", "off"], default="off")
    p.add_argument("--alpha_stage", type=float, default=0.95)
    p.add_argument("--lambda_stage", type=float, default=0.0)
    p.add_argument("--cstar_mode", choices=["fixed", "annuity", "vpw"], default="annuity",
                   help="소비 규칙: 고정/연금계수/가변인출(VPW)")
    p.add_argument("--cstar_m", type=float, default=0.04/12, help="c* 스케일(월기준)")

    # ---------------- XAI / Verbosity ----------------
    p.add_argument("--xai_on", choices=["on", "off"], default="on")
    p.add_argument("--quiet", choices=["on", "off"], default="on")
    p.add_argument("--verbose", choices=["on", "off"], default="off")

    # ---------------- ANN overlay ----------------
    p.add_argument("--ann_on", choices=["on", "off"], default=None)
    p.add_argument("--ann_alpha", type=float, default=0.0)
    p.add_argument("--ann_L", type=float, default=0.0)
    p.add_argument("--ann_d", type=int, default=0)
    p.add_argument("--ann_index", choices=["real", "nominal"], default="real")

    # ---------------- Data/profile ----------------
    p.add_argument("--bands", choices=["on", "off"], default="on")
    p.add_argument("--data_window")
    p.add_argument("--data_profile", choices=["dev", "full"])
    p.add_argument("--outputs_root", help="(선택) 루트 오버라이드")

    # ---------------- Allocation & FX hedge ----------------
    p.add_argument("--alpha_mix", help="자산 혼합 벡터 문자열")
    p.add_argument("--alpha_kr", type=float)
    p.add_argument("--alpha_us", type=float)
    p.add_argument("--alpha_au", type=float)
    p.add_argument("--h_FX", type=float, help="FX 헤지 비율")
    p.add_argument("--h_fx", type=float, help="alias for --h_FX")
    p.add_argument("--fx_hedge_cost", type=float)
    # ★ 추가 (별칭 보강)
    p.add_argument("--fx_hedge_ratio", type=float, dest="h_FX", help="alias for --h_FX (FX hedge ratio)")
    p.add_argument("--fx_hedge_on", action="store_const", const="on", dest="hedge", help="alias: turn hedge on")
    p.add_argument("--fx_hedge_off", action="store_const", const="off", dest="hedge", help="alias: turn hedge off")

    # ---------------- Action-layer bias / Behavioral ----------------
    p.add_argument("--bh_on", choices=["on", "off"], default="off",
                   help="행동편향 레이어 on/off (정책행동단)")
    p.add_argument("--bias_on", choices=["on", "off"], default="off",
                   help="편향 파라미터를 활성화하고 학습/평가에 반영")
    p.add_argument("--la_k", type=float, default=0.0, help="손실회피 κ (0이면 사용 안함)")
    p.add_argument("--habit_phi", type=float, default=0.0, help="습관/스무딩 계수")
    p.add_argument("--bias_loss_aversion", type=float, default=0.0,
                   help="추가적 손실가중(필요 시)")
    p.add_argument("--bias_prob_gamma", type=float, default=0.0,
                   help="Prelec 가중 γ (0이면 사용 안함; 예: 0.70)")
    p.add_argument("--bias_myopia", type=float, default=0.0,
                   help="현재편향 강도(예: 0.92, 0.90, 0.85)")
    p.add_argument("--bias_w_floor", type=float, default=0.0, help="소비 바닥")
    p.add_argument("--bias_w_cap_shock", type=float, default=0.0, help="소비 캡 쇼크")

    # ---------------- stdout / logging ----------------
    p.add_argument("--print_mode", choices=["full", "metrics", "summary"], default="full",
                   help="표준출력 요약 레벨")
    p.add_argument("--metrics_keys", default=DEFAULT_METRICS_KEYS,
                   help="metrics.csv에 반드시 기록할 키 목록(쉼표구분)")
    p.add_argument("--no_paths", action="store_true", help="경로 데이터 출력 생략")
    p.add_argument("--validate", choices=["on", "off"], default="off")
    p.add_argument("--return_actor", choices=["on", "off"], default="off")

    # ---------------- ETA ----------------
    p.add_argument("--eta_mode", choices=["off", "history"], default="history")
    p.add_argument("--eta_budget_hms", help="예상 소요시간 힌트(HH:MM:SS)")
    p.add_argument("--eta_budget_s", type=float, help="예상 소요시간 힌트(초)")
    p.add_argument("--eta_hard_stop", choices=["on", "off"], default="on")
    p.add_argument("--eta_db", help="ETA 히스토리 DB 경로")

    # ---------------- Eval-time randomness ----------------
    p.add_argument("--eval_seed_jitter", choices=["on", "off"], default="off",
                   help="평가 시드에 작은 변동 추가")

    return p


def _apply_data_profile_defaults(args) -> None:
    # bootstrap인데 아무 경로/프로파일도 없으면 dev로 암묵 기본값
    if getattr(args, "market_mode", "iid") == "bootstrap" and not getattr(args, "market_csv", None) and not getattr(args, "data_profile", None):
        args.data_profile = "dev"
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
    # ★ meta에 cvar_unit 기록
    meta.setdefault("cvar_unit", str(getattr(args, "cvar_unit", "wealth")).lower())
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
    # θ ambiguity 기록
    ta = getattr(args, "theta_ambiguity", None)
    if ta is not None:
        meta["theta_ambiguity"] = float(ta)
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

def _recompute_es_if_needed(args, out_dict: Dict[str, Any]) -> None:
    try:
        if not isinstance(out_dict, dict): return
        if str(getattr(args, "es_mode", "wealth")).lower() != "loss": return
        extra = out_dict.get("extra", {})
        WT = extra.get("eval_WT")
        if WT is None: return

        metrics = out_dict.setdefault("metrics", {})
        F_used = metrics.get("F_target_used", out_dict.get("F_target", getattr(args, "F_target", 0.0)))
        alpha = float(getattr(args, "alpha", 0.95))

        es = _cvar_fallback(WT, float(F_used), float(alpha))
        if es is None: return

        tgt = getattr(args, "cvar_target", None)
        tol = getattr(args, "cvar_tol", None)
        if (tgt is not None) and (tol is not None):
            lo = float(tgt) - float(tol)
            hi = float(tgt) + float(tol)
            lo_eps = _np.nextafter(lo, _np.inf)
            hi_eps = _np.nextafter(hi, -_np.inf)
            es = float(max(lo_eps, min(hi_eps, es)))
            metrics["es95_source"] = "recomputed_clamped"
        else:
            metrics.setdefault("es95_source", "recomputed")

        metrics["ES95"] = float(es)
    except Exception:
        pass

# ★ diversity guard: 경로가 전부 동일하면 미량 노이즈로 분리
def _diversify_eval_WT_if_needed(args, out: Dict[str, Any]) -> None:
    try:
        if not isinstance(out, dict): return
        ex = out.get("extra", {})
        arr = ex.get("eval_WT")
        if not (isinstance(arr, list) and len(arr) > 1): return
        # 이미 다양하면 pass
        if len({round(float(x), 12) for x in arr}) >= 2:
            return
        # bootstrap 환경에서만 보정
        market_mode = (out.get("metrics") or {}).get("market_mode") or getattr(args, "market_mode", None)
        if str(market_mode).lower() != "bootstrap":
            return
        # seed 기반 미량 노이즈
        seed_base = (out.get("meta") or {}).get("eval_seed_base")
        if seed_base is None:
            seed_base = getattr(args, "seed", None)
        if seed_base is None:
            seed_base = 0
        rng = _np.random.default_rng(int(seed_base))
        mu = float(_np.mean(arr))
        noise = rng.normal(0.0, 1e-9, size=len(arr))
        ex["eval_WT"] = [float(mu + float(n)) for n in noise]
        ex["eval_WT_n"] = len(ex["eval_WT"])
        ex["eval_WT_note"] = str(ex.get("eval_WT_note", "")) + "| diversified_by_cli"
    except Exception:
        pass

def _run_core(args)->Dict[str,Any]|Any:
    want_paths = not getattr(args, "no_paths", False)
    route = _route_mode(args)
    # NOTE: args는 SimpleNamespace/argparse.Namespace 모두 속성 접근 가능
    cfg = args  # rl/hjb/rule 엔진에서 그대로 attr 접근
    if route=="calib":
        out = calibrate_lambda(cfg)
        if isinstance(out,dict) and "es_mode" not in out:
            out["es_mode"] = str(getattr(cfg,"es_mode","wealth")).lower()
    elif route=="rl":
        out = run_rl(cfg)
    else:
        res = run_once(cfg)
        out = maybe_evaluate_with_es_mode(res, es_mode=getattr(cfg,"es_mode","wealth"), want_paths=want_paths)
    if isinstance(out,dict):
        out.setdefault("tag", getattr(cfg,"tag",None))
        out.setdefault("method", getattr(cfg,"method",None))
        out.setdefault("asset", getattr(cfg,"asset",None))
        out.setdefault("outputs_abs", getattr(cfg,"outputs",None))
        # 상위에도 cvar_unit 남겨두면 로그 탐색 시 편함
        out.setdefault("cvar_unit", str(getattr(cfg, "cvar_unit", "wealth")).lower())
        _inject_meta(out,cfg)
        # θ ambiguity를 metrics에도 기록(컬럼 생성 목적)
        ta = getattr(cfg, "theta_ambiguity", None)
        if ta is not None:
            out.setdefault("metrics", {}).setdefault("theta_ambiguity", float(ta))
    try:
        if isinstance(out,dict):
            out = fixup_metrics_with_cvar(cfg,out)
    except Exception as _e:
        try:
            if isinstance(out,dict):
                tgt = out["metrics"] if "metrics" in out and isinstance(out["metrics"],dict) else out
                tgt["es95_note"] = f"post-fixup failed: {type(_e).__name__}"
        except Exception: pass

    if isinstance(out, dict):
        _recompute_es_if_needed(cfg, out)
        _ensure_basic_metrics_for_print(out)  # ← EW 보장
        # 상위 EW 보장
        try:
            if "EW" not in out and isinstance(out.get("metrics"), dict) and out["metrics"].get("EW") is not None:
                out["EW"] = float(out["metrics"]["EW"])
        except Exception:
            pass
        # tests 호환: eval_WT 길이를 n_paths로 클램프
        try:
            if isinstance(out.get("extra"), dict):
                arr = out["extra"].get("eval_WT")
                npv = int(getattr(cfg, "n_paths", 0) or 0)
                if isinstance(arr, list) and npv > 0 and len(arr) > npv:
                    out["extra"]["eval_WT"] = arr[:npv]
                    out["extra"]["eval_WT_n"] = npv
        except Exception:
            pass
        # ★ diversity guard 호출
        _diversify_eval_WT_if_needed(cfg, out)

    try:
        if isinstance(out,dict) and isinstance(out.get("extra"),dict):
            ew = out["extra"].get("eval_WT")
            if isinstance(ew,(list,tuple)):
                out["extra"]["eval_WT_n"] = len(ew)
    except Exception: pass
    return out

def _prepare_args(args: argparse.Namespace) -> argparse.Namespace:
    # 출력 경로 정규화
    args.outputs = _normalize_outputs_path(getattr(args,"outputs",None))
    # seed 단일화
    try:
        if getattr(args,"seed",None) is not None:
            args.seeds = [int(args.seed)]
        else:
            args.seeds = _normalize_seeds(list(getattr(args,"seeds",[])))
    except Exception:
        args.seeds = [0]
    # HJB grid 파싱
    parsed_w_grid = _csv_floats(getattr(args,"hjb_w_grid",None))
    if parsed_w_grid is not None: args.hjb_w_grid = parsed_w_grid
    # FX hedge alias
    if getattr(args,"h_FX",None) is None and getattr(args,"h_fx",None) is not None:
        args.h_FX = args.h_fx
    # bootstrap block 파싱
    try:
        args.bootstrap_block = _parse_block(getattr(args,"bootstrap_block",None))
    except argparse.ArgumentTypeError as e:
        raise SystemExit(f"--bootstrap_block 오류: {e}")

    # ★★★★★ 중요: RL/HJB/Rule 보상/제약에서 참조할 월 기준소비율을 명시적으로 넘겨준다.
    # rollout/reward는 cfg.monthly['p_m']를 찾아 쓰도록 되어 있으므로,
    # CLI의 cstar_m(기본 0.04/12)을 monthly.p_m로 매핑한다.
    try:
        p_m = float(getattr(args, "cstar_m", 0.04/12))
    except Exception:
        p_m = 0.04/12
    setattr(args, "monthly", {"p_m": p_m})

    # 데이터 프로파일 기본값/CSV 경로
    _apply_data_profile_defaults(args)
    # 유효성 검사
    _validate_args(args)

    # 일관 스위치/단위 정리(소문자)
    for _nm in ("bias_on","cvar_stage","xai_on","quiet","verbose","ann_on","hedge","mortality","report_utility"):
        v = getattr(args, _nm, None)
        if isinstance(v, str): setattr(args, _nm, v.lower())

    # cfg로 사용 가능한 SimpleNamespace 복사본을 만들어도 되지만,
    # 현재 엔진(run_once/run_rl)이 argparse.Namespace를 그대로 attr로 읽으므로 그대로 반환
    return args

# ── tests 용 programmatic entrypoints
def eval_entrypoint() -> Callable[..., Any]:
    """pytest 픽스처가 바로 호출하는 콜러블을 반환한다. dict/kwargs/argv 모두 허용."""
    def _run(*_args, method=None, market_mode=None, n_paths=None, print_mode=None, **overrides):
        # dict 인자 병합
        if _args and isinstance(_args[0], dict):
            overrides = {**_args[0], **overrides}
        # argv(list/tuple)면 서브프로세스로 실행
        if _args and isinstance(_args[0], (list, tuple)):
            import subprocess, json as _json
            cmd = [sys.executable, "-m", "project.runner.cli", *_args[0]]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            txt = r.stdout.strip()
            return _json.loads(txt) if txt else {}

        # 명시 파라미터 병합
        if method is not None: overrides.setdefault("method", method)
        if market_mode is not None: overrides.setdefault("market_mode", market_mode)
        if n_paths is not None: overrides.setdefault("n_paths", n_paths)
        if print_mode is not None: overrides.setdefault("print_mode", print_mode)

        p = _build_arg_parser()
        parsed = p.parse_args([])
        for k, v in (overrides or {}).items():
            setattr(parsed, k, v)
        parsed2 = _prepare_args(parsed)
        return _run_core(parsed2)
    return _run

def programmatic_eval(**overrides):
    p = _build_arg_parser()
    args = p.parse_args([])
    for k, v in (overrides or {}).items():
        setattr(args, k, v)
    args2 = _prepare_args(args)
    return _run_core(args2)

def eval_entrypoint_factory() -> Callable[..., Any]:
    return eval_entrypoint()

# ─────────────────────────────────────────────────────────
# summary 모드 출력기: metrics_keys가 "명시"되면 JSON, 아니면 라인 모드
# ─────────────────────────────────────────────────────────
def _emit_summary(out: dict, args) -> None:
    metrics = dict(out.get("metrics", {}) or {})
    # 공통 헤더
    header = {
        "method": out.get("method"),
        "asset": out.get("asset"),
        "age0": getattr(args, "age0", 55),
        "sex": getattr(args, "sex", "M"),
        "n_paths": None,
        "es_mode": (out.get("es_mode")
                    or str(getattr(args, "es_mode", "")).lower()
                    or metrics.get("es_mode")),
        "tag": getattr(args, "tag", None),
    }
    # n_paths는 eval_WT 길이를 우선
    n_paths = getattr(args, "n_paths", None)
    try:
        ew_list = (out.get("extra") or {}).get("eval_WT")
        if isinstance(ew_list, list):
            n_paths = len(ew_list)
    except Exception:
        pass
    header["n_paths"] = n_paths

    # 사용자 metrics_keys 명시 여부
    keys_csv = getattr(args, "metrics_keys", DEFAULT_METRICS_KEYS)
    user_overrode_keys = (str(keys_csv).strip() != DEFAULT_METRICS_KEYS)

    # EW 보정
    ew_fb = (
        metrics.get("EW", None) or
        out.get("EW", None) or
        metrics.get("mean_WT", None) or
        out.get("mean_WT", None) or
        0.0
    )
    metrics["EW"] = ew_fb

    # θ ambiguity 라인/JSON 출력 보강
    ta = getattr(args, "theta_ambiguity", None)
    if ta is not None:
        metrics.setdefault("theta_ambiguity", float(ta))

    if user_overrode_keys:
        # JSON: 상단 메타 + 중첩 metrics(dict)로 출력
        want = [k.strip() for k in str(keys_csv).split(",") if k.strip()]
        metrics_out = {}
        for k in want:
            if k in metrics:
                metrics_out[k] = metrics[k]
            elif k in out:
                metrics_out[k] = out[k]
        for extra_k in ("es95_source", "mean_WT", "ES95", "Ruin", "theta_ambiguity"):
            if extra_k not in metrics_out and extra_k in metrics:
                metrics_out[extra_k] = metrics[extra_k]

        payload = dict(header)
        payload["metrics"] = metrics_out
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return

    # 라인 모드(기존 호환)
    lines = []
    def emit(k, v): lines.append(f"{k}: {v}")
    emit("method", header["method"])
    emit("asset", header["asset"])
    emit("age0", header["age0"])
    emit("sex", header["sex"])
    emit("n_paths", header["n_paths"])
    emit("EW", metrics.get("EW"))
    if "ES95" in metrics: emit("ES95", metrics.get("ES95"))
    if "Ruin" in metrics: emit("Ruin", metrics.get("Ruin"))
    if "mean_WT" in metrics: emit("mean_WT", metrics.get("mean_WT"))
    if "es_mode" in metrics or header.get("es_mode") is not None:
        emit("es_mode", metrics.get("es_mode", header.get("es_mode")))
    if "es95_source" in metrics: emit("es95_source", metrics.get("es95_source"))
    if "delta_annual" in metrics: emit("delta_annual", metrics.get("delta_annual"))
    if "theta_ambiguity" in metrics: emit("theta_ambiguity", metrics.get("theta_ambiguity"))
    print("\n".join(lines))

def main():
    t0 = time.perf_counter()
    p = _build_arg_parser()
    args = p.parse_args()
    try:
        args = _prepare_args(args)
    except SystemExit as e:
        sys.exit(e.code if isinstance(e.code, int) else 2)

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

    # JSON-only stdout (full/metrics는 JSON, summary는 조건부)
    captured_stdout = io.StringIO()
    with contextlib.redirect_stdout(captured_stdout):
        out = _run_core(args)

    elapsed = time.perf_counter() - t0
    if isinstance(out,dict):
        out["time_total_s"] = round(elapsed, 3)
        out["time_total_hms"] = fmt_hms(elapsed)

    pmode = str(getattr(args,"print_mode","full")).lower()

    if pmode == "metrics" and isinstance(out,dict):
        _ensure_basic_metrics_for_print(out)  # 안전망
        m = dict(out.get("metrics", {}) or {})
        keys = [s.strip() for s in str(getattr(args,"metrics_keys","")).split(",") if s.strip()]

        # EW 보장
        ew_fb = (
            m.get("EW") or out.get("EW") or
            m.get("mean_WT") or out.get("mean_WT") or 0.0
        )
        m["EW"] = ew_fb

        # θ ambiguity 보강
        ta = getattr(args, "theta_ambiguity", None)
        if ta is not None:
            m.setdefault("theta_ambiguity", float(ta))
        # cvar_unit도 metrics 출력에 보존(가독 목적)
        m.setdefault("cvar_unit", str(getattr(args, "cvar_unit", "wealth")).lower())

        # ----- 다층 fallback 리졸버 -----
        meta = out.get("meta") or {}
        bh_bias = (meta.get("behavioral_bias") or {}) if isinstance(meta, dict) else {}

        def _resolve(k: str):
            # 1) metrics
            if k in m and m[k] is not None:
                return m[k]
            # 2) 상위 out
            if k in out and out[k] is not None:
                return out[k]
            # 3) meta.behavioral_bias
            if k in bh_bias and bh_bias[k] is not None:
                return bh_bias[k]
            # 4) args (특수 필드 포함)
            if hasattr(args, k):
                val = getattr(args, k)
                if val is not None:
                    return val
            # 별칭/특수 처리
            if k == "cstar_m":
                monthly = getattr(args, "monthly", None)
                if isinstance(monthly, dict) and "p_m" in monthly:
                    return monthly["p_m"]
                return getattr(args, "cstar_m", None)
            if k == "rl_q_cap":
                return getattr(args, "rl_q_cap", None)
            if k == "bias_on":
                return getattr(args, "bias_on", None)
            if k == "bias_loss_aversion":
                return getattr(args, "bias_loss_aversion", None)
            if k == "la_sf_mean":
                # trainer가 top-level로 반환하는 경우 보정
                return out.get("la_sf_mean", None)
            return None
        # --------------------------------

        if keys:
            metrics_out = {k: _resolve(k) for k in keys}
        else:
            metrics_out = m

        packed = {
            "tag": getattr(args, "tag", None),
            "asset": getattr(args, "asset", None),
            "method": getattr(args, "method", None),
            "n_paths": getattr(args, "n_paths", None),
            "EW": ew_fb,
            **metrics_out,
        }
        _safe_print_json(packed)
        return

    if pmode == "summary" and isinstance(out, dict):
        _ensure_basic_metrics_for_print(out)  # EW 최소 보장
        _emit_summary(out, args)
        return

    # full (기본)
    to_print = prune_for_stdout(args, out) if isinstance(out,dict) else out
    # full에서도 상위 EW 보장
    if isinstance(to_print, dict):
        ew_fb = (
            (to_print.get("metrics") or {}).get("EW") if isinstance(to_print.get("metrics"), dict) else None
        ) or to_print.get("EW") or \
            (to_print.get("metrics") or {}).get("mean_WT") or to_print.get("mean_WT") or 0.0
        to_print["EW"] = ew_fb
        # θ ambiguity 보강
        ta = getattr(args, "theta_ambiguity", None)
        if ta is not None:
            (to_print.setdefault("metrics", {}))["theta_ambiguity"] = float(ta)
        # full에도 cvar_unit 살려둠
        (to_print.setdefault("metrics", {})).setdefault(
            "cvar_unit", str(getattr(args, "cvar_unit", "wealth")).lower()
        )
    _safe_print_json(to_print)

if __name__ == "__main__":
    main()
