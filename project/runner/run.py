# project/runner/run.py
from __future__ import annotations

import contextlib
import os
import time
from typing import Any, Dict, Optional, Callable, Tuple, List, Union

import numpy as _np

from project.eval import evaluate, save_metrics_autocsv  # ← 절대경로
from project.config import SimConfig
from project.runner.config_build import make_cfg
from project.runner.actors import build_actor
from project.runner.annuity_wiring import setup_annuity_overlay
from project.runner.io_utils import ensure_dir, slim_args, do_autosave
from project.runner.logging_filters import silence_stdio
from project.data.loader import load_market_csv
from project.env.retirement_env import RetirementEnv  # type: ignore

# ✅ metrics.csv 기록 (주 기록 루트)
from project.utils.logging_io import write_metrics_csv

# ✅ 효용-레이어 행동편향(있으면 사용, 없으면 안전 폴백)
try:
    from project.policy.behavioral import parse_behavioral_from_args as _parse_bh, describe as _bh_describe  # type: ignore
except Exception:
    _parse_bh = None
    def _bh_describe(_spec) -> Dict[str, Any]:  # type: ignore
        try:
            return {
                "bh_on": bool(getattr(_spec, "on", False)),
                "lambda_loss": float(getattr(_spec, "lambda_loss", 1.0)),
                "beta": float(getattr(_spec, "beta", 1.0)),
                "habit_phi": float(getattr(_spec, "habit_phi", 0.0)),
            }
        except Exception:
            return {"bh_on": False, "lambda_loss": 1.0, "beta": 1.0, "habit_phi": 0.0}


# --------------------------
# Utilities
# --------------------------
def _fmt_hms(sec: float) -> str:
    try:
        total = int(round(float(sec)))
        m, s = divmod(total, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    except Exception:
        return "00:00:00"


def _onoff(v: Any, default: str = "on") -> str:
    s = str(v).strip().lower() if v is not None else default
    if s in ("on", "off"):
        return s
    if s in ("true", "1", "y", "yes"):
        return "on"
    if s in ("false", "0", "n", "no"):
        return "off"
    return default


# ---- ES/EV helpers (NEW) ----
def _es_tail_mean(arr, alpha=0.95):
    """Wealth 모드 ES: 하위 (1-α) 구간 평균."""
    x = _np.asarray(arr, dtype=float)
    x = x[_np.isfinite(x)]
    n = x.size
    if n == 0:
        return None
    k = max(1, int(_np.ceil((1.0 - float(alpha)) * n)))
    # 하위 k개 요소 평균
    part = _np.partition(x, k - 1)[:k]
    return float(_np.mean(part))


def _cvar_loss_from_wealth(arr, F_target, alpha=0.95):
    """Loss 모드 ES: L=max(F−W,0)의 상위 (1-α) 구간 평균."""
    x = _np.asarray(arr, dtype=float)
    x = x[_np.isfinite(x)]
    n = x.size
    if n == 0:
        return None
    loss = _np.maximum(float(F_target) - x, 0.0)
    k = max(1, int(_np.ceil((1.0 - float(alpha)) * n)))
    # 손실 상위 k개 평균
    part = _np.partition(loss, -(k))[-k:]
    return float(_np.mean(part))


# --------------------------
# Helpers: parsing mix / hedge
# --------------------------
def _parse_alpha_mix(args) -> Tuple[float, float, float]:
    def _as_float(x, default=None):
        try:
            return float(x)
        except Exception:
            return default

    if getattr(args, "alpha_mix", None):
        raw = str(args.alpha_mix).replace(" ", "")
        parts = [p for p in raw.split(",") if p != ""]
        if len(parts) == 3:
            kr = _as_float(parts[0], 1 / 3)
            us = _as_float(parts[1], 1 / 3)
            au = _as_float(parts[2], 1 / 3)
        else:
            kr = us = au = 1 / 3
    else:
        kr = _as_float(getattr(args, "alpha_kr", None), None)
        us = _as_float(getattr(args, "alpha_us", None), None)
        au = _as_float(getattr(args, "alpha_au", None), None)
        if kr is None or us is None or au is None:
            kr = us = au = 1 / 3

    s = kr + us + au
    if s <= 0:
        return (1/3, 1/3, 1/3)
    return (kr / s, us / s, au / s)


def _get_fx_hedge_params(args) -> Tuple[float, float]:
    h = getattr(args, "h_FX", getattr(args, "h_fx", None))
    try:
        h = float(h)
    except Exception:
        h = 0.0
    h = max(0.0, min(1.0, h))

    fx_cost_annual = getattr(args, "fx_hedge_cost", None)
    try:
        fx_cost_annual = float(fx_cost_annual)
    except Exception:
        fx_cost_annual = 0.002
    return h, fx_cost_annual


# --------------------------
# Market meta injection
# --------------------------
def _inject_market_meta(cfg: SimConfig, args, out_dict: Dict[str, Any]) -> None:
    if not isinstance(out_dict, dict):
        return
    market_mode = str(getattr(cfg, "market_mode", "iid") or "iid").lower()
    bootstrap_block = int(getattr(cfg, "bootstrap_block", 24) or 24)
    use_real_rf = _onoff(getattr(cfg, "use_real_rf", "on"), default="on")
    data_window = getattr(cfg, "data_window", None)
    market_meta_from_cfg = getattr(cfg, "meta", {}).get("market", {}) if getattr(cfg, "meta", None) else {}
    market_csv = getattr(args, "market_csv", None)
    data_profile = getattr(args, "data_profile", None)

    meta = out_dict.setdefault("meta", {})
    meta_market = dict(market_meta_from_cfg) if isinstance(market_meta_from_cfg, dict) else {}
    meta_market.update({
        "mode": market_mode,
        "bootstrap_block": bootstrap_block,
        "use_real_rf": use_real_rf,
        "data_window": data_window or "",
        "market_csv": os.path.abspath(market_csv) if market_csv else "",
        "data_profile": data_profile or "",
    })
    meta["market"] = meta_market
    out_dict["meta"] = meta

    metrics = out_dict.get("metrics")
    if isinstance(metrics, dict):
        metrics.setdefault("market_mode", market_mode)
        metrics.setdefault("bootstrap_block", bootstrap_block)
        metrics.setdefault("use_real_rf", use_real_rf)
        metrics.setdefault("data_window", data_window or "")
        metrics.setdefault("market_csv", meta_market["market_csv"])
        metrics.setdefault("data_profile", data_profile or "")

    if _onoff(getattr(args, "quiet", "on")) != "on":
        try:
            print(f"[market] mode={market_mode}, block={bootstrap_block}, use_real_rf={use_real_rf}, "
                  f"window='{data_window or ''}', profile='{data_profile or ''}', csv='{market_csv or ''}'")
        except Exception:
            pass


# --------------------------
# Data wiring (with mix & FX hedge)
# --------------------------
def _wire_market_data(cfg: SimConfig, args) -> None:
    setattr(cfg, "bands", getattr(args, "bands", "on"))
    setattr(cfg, "data_window", getattr(args, "data_window", None))
    setattr(cfg, "use_real_rf", _onoff(getattr(args, "use_real_rf", "on")))

    if getattr(cfg, "market_mode", "iid") != "bootstrap":
        return

    market_csv = getattr(args, "market_csv", None)
    if not market_csv:
        raise SystemExit(
            "market_mode=bootstrap 사용 시 --market_csv 또는 --data_profile(dev|full)이 필요합니다."
        )

    abs_csv = os.path.abspath(market_csv)
    if not os.path.exists(abs_csv):
        cwd = os.getcwd()
        raise SystemExit(
            "market_csv 파일을 찾을 수 없습니다.\n"
            f"  asked: {market_csv}\n"
            f"  abs:   {abs_csv}\n"
            f"  cwd:   {cwd}\n"
            "힌트: 경로/파일명을 확인하거나 --data_profile dev|full 사용"
        )

    blob = load_market_csv(
        path=abs_csv,
        asset=getattr(cfg, "asset", "KR"),
        use_real_rf=getattr(args, "use_real_rf", "on"),
        data_window=getattr(args, "data_window", None),
        cache=True,
    )

    ret_kr = blob.get("ret_kr_eq")
    ret_us_l = blob.get("ret_us_eq_krw")
    ret_au = blob.get("ret_gold_krw")
    rf_real = blob.get("rf_real")
    rf_nom = blob.get("rf_nom")
    dates = blob.get("dates")
    cpi = blob.get("cpi")
    ret_fx = blob.get("ret_fx", None)
    if ret_fx is None:
        ret_fx = blob.get("ret_fx_usdkrw", None)

    import numpy as np
    steps_per_year = int(getattr(cfg, "steps_per_year", 12) or 12)
    h_fx, fx_cost_ann = _get_fx_hedge_params(args)
    fx_cost_m = float(fx_cost_ann) / float(steps_per_year)

    if ret_us_l is None:
        mixed = blob.get("ret_asset")
    else:
        if ret_fx is not None:
            ret_us_hedged = (
                np.asarray(ret_us_l, dtype=float)
                - h_fx * np.asarray(ret_fx, dtype=float)
                - h_fx * fx_cost_m
            )
        else:
            ret_us_hedged = np.asarray(ret_us_l, dtype=float) - h_fx * fx_cost_m

        a_kr, a_us, a_au = _parse_alpha_mix(args)
        kr = np.asarray(ret_kr, dtype=float) if ret_kr is not None else 0.0
        us = np.asarray(ret_us_hedged, dtype=float)
        au = np.asarray(ret_au, dtype=float) if ret_au is not None else 0.0

        lens = [x.shape[0] for x in [kr, us, au] if isinstance(x, np.ndarray)]
        T = min(lens) if len(lens) >= 1 else 0
        if isinstance(kr, np.ndarray): kr = kr[:T]
        if isinstance(us, np.ndarray): us = us[:T]
        if isinstance(au, np.ndarray): au = au[:T]
        mixed = a_kr * kr + a_us * us + a_au * au

        setattr(cfg, "alpha_mix", (a_kr, a_us, a_au))
        setattr(cfg, "h_FX", h_fx)
        setattr(cfg, "fx_hedge_cost_annual", fx_cost_ann)

    setattr(cfg, "data_dates", dates)
    setattr(cfg, "data_cpi", cpi)
    setattr(cfg, "data_ret_series", mixed)
    setattr(cfg, "data_rf_series", rf_real if _onoff(getattr(args, "use_real_rf", "on")) == "on" else rf_nom)
    setattr(cfg, "data_ret_kr_eq", ret_kr)
    setattr(cfg, "data_ret_us_eq_krw", ret_us_l)
    setattr(cfg, "data_ret_gold_krw", ret_au)

    if _onoff(getattr(args, "quiet", "on")) != "on":
        try:
            ret_mean = float(_np.nanmean(mixed)) if mixed is not None else float("nan")
            rf_series = rf_real if _onoff(getattr(args, "use_real_rf", "on")) == "on" else rf_nom
            rf_mean = float(_np.nanmean(rf_series)) if rf_series is not None else float("nan")
            a = getattr(cfg, "alpha_mix", None)
            a_str = f"alpha={a}" if a is not None else "alpha=legacy"
            print(
                f"[data] len={len(mixed) if mixed is not None else 0}, "
                f"ret_mean={ret_mean:.4f}, rf_mean={rf_mean:.4f}, "
                f"h_FX={getattr(cfg,'h_FX',0.0):.2f}, {a_str}, "
                f"asset={getattr(cfg, 'asset', '?')}, window={getattr(cfg, 'data_window', None)}"
            )
        except Exception:
            pass


def _to_actor(policy_like: Any) -> Callable[[Any], tuple[float, float]]:
    if policy_like is None:
        raise RuntimeError("policy_like is None")

    def _actor(state: Any):
        out = None
        if hasattr(policy_like, "act"):
            out = policy_like.act(state)
        elif callable(policy_like):
            out = policy_like(state)
        elif hasattr(policy_like, "predict"):
            out = policy_like.predict(state)
        else:
            raise RuntimeError("No callable interface for actor: need .act/.predict/callable")

        if isinstance(out, dict) and "q" in out and "w" in out:
            q, w = out["q"], out["w"]
        elif isinstance(out, (tuple, list)) and len(out) >= 2:
            q, w = out[0], out[1]
        else:
            raise RuntimeError("actor must return (q, w) or dict with keys 'q','w'")
        return float(q), float(w)

    return _actor


def _normalize_evaluate_output(ret, es_mode: str):
    metrics, extras = {}, {}

    if isinstance(ret, dict):
        metrics = ret
    elif isinstance(ret, tuple):
        if len(ret) >= 1 and isinstance(ret[0], dict):
            metrics = ret[0]
        if len(ret) >= 2 and isinstance(ret[1], dict):
            extras = ret[1]
        if len(ret) > 2:
            extras["_rest"] = ret[2:]
    else:
        metrics = {"note": "unexpected evaluate return type", "type": str(type(ret))}

    if "es_mode" not in metrics:
        metrics["es_mode"] = str(es_mode).lower()
    metrics.setdefault("es95_source", "computed_in_evaluate")
    return metrics, extras


def _call_evaluate(cfg, actor, es_mode: str):
    try:
        ret = evaluate(cfg, actor, es_mode=str(es_mode).lower(), return_paths=True)
    except TypeError:
        ret = evaluate(cfg, actor, es_mode=str(es_mode).lower())
    return _normalize_evaluate_output(ret, es_mode)


def _as_ndarray_state(raw_obs: Union[Dict[str, Any], _np.ndarray, None], env: Any) -> _np.ndarray:
    if isinstance(raw_obs, dict):
        T = int(getattr(env, "T", 1) or 1)
        t_idx = float(raw_obs.get("t", 0.0))
        t_norm = t_idx / float(max(1, T - 1))
        W_now = float(raw_obs.get("W", getattr(env, "W", 0.0)))
        return _np.asarray([t_norm, W_now], dtype=float)
    if isinstance(raw_obs, _np.ndarray):
        arr = _np.asarray(raw_obs, dtype=float).ravel()
        if arr.size >= 2:
            return arr[:2]
        W_now = float(getattr(env, "W", 0.0))
        return _np.asarray([float(arr[0]) if arr.size >= 1 else 0.0, W_now], dtype=float)
    return _np.asarray([0.0, float(getattr(env, "W", 0.0))], dtype=float)


def _rollout_terminal_wealths(cfg: SimConfig,
                              actor: Callable[[Any], tuple[float, float]],
                              n_paths: int) -> List[float]:
    env = RetirementEnv(cfg)
    WTs: List[float] = []
    n_paths = int(max(1, n_paths))

    for _ in range(n_paths):
        env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            raw = env._obs() if hasattr(env, "_obs") else None
            state_nd = _as_ndarray_state(raw, env)
            q, w = actor(state_nd)
            _, _, done, truncated, _ = env.step(q=q, w=w)
        WTs.append(float(getattr(env, "W", 0.0)))
    return WTs


def _looks_degenerate_wt(xs) -> bool:
    try:
        arr = _np.asarray(xs, dtype=float).ravel()
        if arr.size <= 1:
            return True
        return bool(_np.allclose(arr, arr[0], atol=0, rtol=0))
    except Exception:
        return True


def run_once(args) -> Dict[str, Any]:
    t_all_0 = time.perf_counter()

    quiet_ctx = silence_stdio(also_stderr=True) if _onoff(getattr(args, "quiet", "on")) == "on" else contextlib.nullcontext()
    with quiet_ctx:
        t0 = time.perf_counter()
        cfg: SimConfig = make_cfg(args)

        bh_spec = None
        if _parse_bh is not None:
            try:
                bh_spec = _parse_bh(args)
                setattr(cfg, "behavioral_spec", bh_spec)
            except Exception:
                setattr(cfg, "behavioral_spec", None)
        time_make_cfg = time.perf_counter() - t0

        ensure_dir(args.outputs)
        if getattr(args, "tag", None) is not None:
            setattr(cfg, "tag", args.tag)

        t1 = time.perf_counter(); _wire_market_data(cfg, args); time_wire_data = time.perf_counter() - t1
        t2 = time.perf_counter()
        ann_enabled = (_onoff(getattr(args, "ann_on", "off")) == "on" and float(getattr(args, "ann_alpha", 0.0) or 0.0) > 0.0)
        if ann_enabled: setup_annuity_overlay(cfg, args)
        time_annuity = time.perf_counter() - t2

        t3 = time.perf_counter(); actor = build_actor(cfg, args); time_build_actor = time.perf_counter() - t3
        t4 = time.perf_counter(); m, extras = _call_evaluate(cfg, actor, es_mode=getattr(args, "es_mode", "wealth")); time_eval = time.perf_counter() - t4

    extras = extras or {}
    need_paths = (str(getattr(args, "print_mode", "full")).lower() == "full") and (not getattr(args, "no_paths", False))
    wt_from_eval = extras.get("eval_WT", None)
    if (not isinstance(wt_from_eval, (list, tuple))) or _looks_degenerate_wt(wt_from_eval):
        if need_paths:
            n_paths = getattr(args, "n_paths", None)
            if n_paths is None or int(n_paths) <= 0:
                n_paths = int(getattr(cfg, "n_paths_eval", 0)) or 100
            try:
                WTs = _rollout_terminal_wealths(cfg, _to_actor(actor), int(n_paths))
                extras["eval_WT"] = [float(x) for x in WTs]
                m.setdefault("mean_WT", float(_np.mean(WTs)))
                m.setdefault("EW", float(_np.mean(WTs)))
            except Exception as _e:
                extras.setdefault("eval_WT_note", f"local rollout failed: {type(_e).__name__}")

    if isinstance(m, dict):
        y_ann = float(getattr(cfg, "y_ann", 0.0) or 0.0)
        a_fac = float(getattr(cfg, "ann_a_factor", 0.0) or 0.0)
        P_val = float(getattr(cfg, "ann_P", 0.0) or 0.0)
        m.update({"y_ann": y_ann if y_ann != 0.0 else 0.0,
                  "ann_a_factor": a_fac if a_fac != 0.0 else 0.0,
                  "a_factor": a_fac if a_fac != 0.0 else 0.0,
                  "P": P_val if P_val != 0.0 else 0.0})

        if bh_spec is not None:
            try:
                m.update(_bh_describe(bh_spec))
            except Exception:
                pass

    n_paths_total = len(extras.get("eval_WT", [])) or (
        (getattr(cfg, "n_paths_eval", getattr(cfg, "n_paths", 0)) or 0) * max(1, len(getattr(cfg, "seeds", [])))
    )

    time_total = time.perf_counter() - t_all_0
    timing = {
        "make_cfg_s": round(time_make_cfg, 6),
        "wire_data_s": round(time_wire_data, 6),
        "annuity_setup_s": round(time_annuity, 6),
        "build_actor_s": round(time_build_actor, 6),
        "evaluate_s": round(time_eval, 6),
        "total_s": round(time_total, 6),
        "total_hms": _fmt_hms(time_total),
    }

    out = dict(
        asset=getattr(cfg, "asset", None),
        method=getattr(args, "method", ""),
        baseline=getattr(args, "baseline", ""),
        metrics=m,
        w_max=getattr(cfg, "w_max", None),
        fee_annual=getattr(cfg, "phi_adval", getattr(cfg, "fee_annual", None)),
        lambda_term=getattr(cfg, "lambda_term", None),
        alpha=getattr(cfg, "alpha", None),
        F_target=getattr(cfg, "F_target", None),
        es_mode=getattr(args, "es_mode", "wealth"),
        n_paths=int(n_paths_total),
        args=slim_args(args),
        extra=extras,
        timing=timing,
        time_total_s=timing["total_s"],
        time_total_hms=timing["total_hms"],
    )

    _inject_market_meta(cfg, args, out)

    meta_bh = None
    if bh_spec is not None:
        try:
            meta_bh = _bh_describe(bh_spec)
        except Exception:
            meta_bh = None
    out.setdefault("meta", {})
    if meta_bh:
        out["meta"]["behavioral"] = meta_bh

    metrics_csv = os.path.join(args.outputs, "_logs", "metrics.csv")
    meta = {
        "tag": getattr(args, "tag", None),
        "method": out["method"],
        "asset": getattr(cfg, "asset", None),
        "outputs_abs": os.path.abspath(args.outputs),
        "time_total_hms": timing["total_hms"],
        "market_mode": getattr(cfg, "market_mode", None),
        "bootstrap_block": getattr(cfg, "bootstrap_block", None),
        "use_real_rf": getattr(cfg, "use_real_rf", None),
        "data_window": getattr(cfg, "data_window", None),
    }
    if meta_bh:
        meta.update({f"bh_{k}": v for k, v in meta_bh.items()})

    try:
        write_metrics_csv(metrics_csv, args, out, meta=meta)
    except Exception:
        pass

    try:
        setattr(cfg, "method", getattr(args, "method", ""))
        setattr(cfg, "es_mode", getattr(args, "es_mode", "wealth"))
        save_metrics_autocsv(out.get("metrics", {}), cfg, outputs=args.outputs)
    except Exception:
        pass

    if _onoff(getattr(args, "autosave", "off")) == "on":
        do_autosave(m, cfg, args, out)

    return out


# ==========================
# RL (RLTrainer 경로, 절대임포트 고정)
# ==========================
# (중략) 파일 상단 및 다른 함수들은 그대로 유지

def _build_env_factory_from_args(args):
    """IRPEnvAdapter(env) + RetirementEnv kwargs 전달."""
    from project.env.irp_env import IRPEnvAdapter  # ← 절대경로

    fee_annual = float(args.phi_adval) if (getattr(args, "phi_adval", None) not in (None, 0.0)) else float(args.fee_annual)

    # RetirementEnv가 직접 받는 값들은 base_kwargs에만 넣는다.
    base_kwargs = dict(
        horizon_years   = int(args.horizon_years),
        # w_max는 IRPEnvAdapter에도 전달하지만, 혹시 RetirementEnv도 받는다면 setdefault로 중복 안전
        w_max           = float(args.w_max),
        fee_annual      = float(fee_annual),
        floor_on        = "on" if bool(getattr(args, "floor_on", False)) else "off",
        f_min_real      = float(getattr(args, "f_min_real", 0.0) or 0.0),
        market_mode     = str(args.market_mode),
        market_csv      = str(getattr(args, "market_csv", "" ) or ""),
        bootstrap_block = int(args.bootstrap_block),
        use_real_rf     = str(args.use_real_rf),
        survive_bonus   = float(getattr(args, "survive_bonus", 0.0) or 0.0),
        u_scale         = float(getattr(args, "u_scale", 0.05) or 0.05),
        crra_gamma      = float(getattr(args, "crra_gamma", 3.0) or 3.0),
        age0            = int(getattr(args, "age0", 65)),
        sex             = str(getattr(args, "sex", "M")),
        seeds=[int(getattr(args, "seed", 0))],   # 또는 seeds=args.seeds
        F_target        = float(getattr(args, "F_target", 0.0) or 0.0),
    )

    # IRPEnvAdapter에 허용된 인자만 직접 전달: f_target, w_max, q_floor, base_kwargs
    def env_factory():
        return IRPEnvAdapter(
            f_target   = base_kwargs.get("F_target", 0.0),
            w_max      = base_kwargs.get("w_max", None),
            q_floor    = float(getattr(args, "q_floor", 0.0) or 0.0),
            base_kwargs= base_kwargs,
            q_cap      = float(getattr(args, "rl_q_cap", 0.01) or 0.01),  # ★ 추가: 기본 1%/월
        )
    return env_factory

def _deterministic_policy_step(tr, obs, device):
    import torch
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    dist_q, dist_w, _ = tr.actor(obs_t)
    a_q = (dist_q.concentration1 / (dist_q.concentration1 + dist_q.concentration0)).squeeze(-1)
    a_w = (dist_w.concentration1 / (dist_w.concentration1 + dist_w.concentration0)).squeeze(-1)
    a_q = float(a_q.detach().cpu().item())
    a_w = float(a_w.detach().cpu().item())
    return {"q": a_q, "w": a_w}


def _evaluate_collect_WT(tr, env_factory, n_episodes: int) -> Dict[str, Any]:
    WT = []
    returns = []
    # 평가 시드: 트레이너 시드 기준으로 에피소드마다 분기
    base_seed = int(getattr(tr.cfg, "seed", 0))
    for ep in range(int(n_episodes)):
        env = env_factory()
        # 에피소드별 고유 시드
        eval_seed = base_seed + ep
        obs = env.reset(seed=eval_seed)   # <<< 핵심 수정
        done = False
        ret_sum = 0.0
        info = {}
        while not done:
            act = _deterministic_policy_step(tr, obs, tr.device)
            step_out = env.step(act)
            if isinstance(step_out, tuple) and len(step_out) == 5:
                obs, rew, done, trunc, info = step_out
                done = bool(done) or bool(trunc)
            else:
                obs, rew, done, info = step_out
            ret_sum += float(rew)
        W_T = info.get("W_T") or info.get("terminal_wealth") or info.get("W")
        WT.append(float(W_T) if W_T is not None else 0.0)
        returns.append(ret_sum)
    out = {
        "eval_WT": WT,
        "eval_return_mean": float(_np.mean(returns)) if len(returns) else 0.0,
        "eval_return_std": float(_np.std(returns)) if len(returns) else 0.0,
        "episodes": int(n_episodes),
    }
    return out

def run_rl(args):
    t_all_0 = time.perf_counter()

    t0 = time.perf_counter()
    cfg: SimConfig = make_cfg(args)

    bh_spec = None
    if _parse_bh is not None:
        try:
            bh_spec = _parse_bh(args)
            setattr(cfg, "behavioral_spec", bh_spec)
        except Exception:
            setattr(cfg, "behavioral_spec", None)
    time_make_cfg = time.perf_counter() - t0

    ensure_dir(args.outputs)
    if getattr(args, "tag", None) is not None:
        setattr(cfg, "tag", args.tag)

    t1 = time.perf_counter(); _wire_market_data(cfg, args); time_wire_data = time.perf_counter() - t1

    ann_enabled = (_onoff(getattr(args, "ann_on", "off")) == "on" and float(getattr(args, "ann_alpha", 0.0) or 0.0) > 0.0)
    t2 = time.perf_counter()
    if ann_enabled: setup_annuity_overlay(cfg, args)
    time_annuity = time.perf_counter() - t2

    env_factory = _build_env_factory_from_args(args)

    # 🔒 절대경로 임포트 고정
    try:
        from project.trainer.rl_trainer import RLConfig, RLTrainer
    except Exception as e1:
        try:
            from ..trainer.rl_trainer import RLConfig, RLTrainer  # type: ignore
        except Exception as e2:
            raise SystemExit(f"RL trainer import failed: {e1}")

    max_steps = int(args.rl_epochs) * int(args.rl_steps_per_epoch)
    cfg_rl = RLConfig(
        obs_dim = -1,
        hidden_dims = [128, 128],
        gamma = float(getattr(args, "beta", 0.996) or 0.996),
        lam = float(getattr(args, "gae_lambda", 0.95) or 0.95),
        ent_coef = float(getattr(args, "entropy_coef", 0.005) or 0.005),
        vf_coef = float(getattr(args, "value_coef", 0.5) or 0.5),
        lr = float(getattr(args, "lr", 3e-4) or 3e-4),
        max_grad_norm = float(getattr(args, "max_grad_norm", 0.5) or 0.5),
        max_steps = max_steps,
        rollout_len = int(getattr(args, "rl_steps_per_epoch", 512) or 512),
        batch_size = int(getattr(args, "rl_steps_per_epoch", 512) or 512),
        seed = int(args.seeds[0] if isinstance(args.seeds, (list, tuple)) else int(args.seeds)),
        log_dir = os.path.join(os.path.abspath(args.outputs), "_logs"),
        tag = str(getattr(args, "tag", "rl_run") or "rl_run"),
        device = "auto",
        value_clip = 0.0,
        entropy_clip = 0.0,
    )

    t3 = time.perf_counter()
    trainer = RLTrainer(cfg_rl, env_factory)
    trainer.train()
    time_train_call = time.perf_counter() - t3

    t4 = time.perf_counter()
    extras_dict = _evaluate_collect_WT(trainer, env_factory, int(getattr(args, "rl_n_paths_eval", 64) or 64))
    time_eval = time.perf_counter() - t4

    # ---- Build metrics from eval_WT (NEW) ----
    wt = extras_dict.get("eval_WT", []) or []
    alpha = float(getattr(cfg, "alpha", 0.95) or 0.95)
    es_mode = str(getattr(args, "es_mode", "wealth")).lower()
    F_target = float(getattr(cfg, "F_target", 0.0) or 0.0)

    metrics_dict: Dict[str, Any] = {}
    if len(wt) > 0:
        ew = float(_np.mean(wt))
        ruin = float(_np.mean(_np.asarray(wt, dtype=float) <= 0.0))
        if es_mode == "wealth":
            es95 = _es_tail_mean(wt, alpha=alpha)
        else:  # "loss"
            es95 = _cvar_loss_from_wealth(wt, F_target=F_target, alpha=alpha)
        metrics_dict.update({
            "EW": ew,
            "mean_WT": ew,
            "ES95": es95,
            "Ruin": ruin,
            "es95_source": f"computed_from_eval_WT_{es_mode}",
            "eval_episodes": int(extras_dict.get("episodes", 0)),
        })
    else:
        metrics_dict.update({
            "EW": None, "mean_WT": None, "ES95": None, "Ruin": None,
            "es95_source": "no_eval_WT",
            "eval_episodes": int(extras_dict.get("episodes", 0)),
        })

    time_total = time.perf_counter() - t_all_0
    timing = {
        "make_cfg_s": round(time_make_cfg, 6),
        "wire_data_s": round(time_wire_data, 6),
        "annuity_setup_s": round(time_annuity, 6),
        "train_call_s": round(time_train_call, 6),
        "evaluate_s": round(time_eval, 6),
        "total_s": round(time_total, 6),
        "total_hms": _fmt_hms(time_total),
    }

    out = dict(
        asset=getattr(cfg, "asset", None),
        method="rl",
        baseline="",
        metrics=metrics_dict,  # ← filled
        w_max=getattr(cfg, "w_max", None),
        fee_annual=getattr(cfg, "phi_adval", getattr(cfg, "fee_annual", None)),
        lambda_term=getattr(cfg, "lambda_term", None),
        alpha=getattr(cfg, "alpha", None),
        F_target=getattr(cfg, "F_target", None),
        es_mode=es_mode,
        n_paths=int(getattr(args, "rl_n_paths_eval", 64) or 64),
        args=slim_args(args) | {
            "rl_q_cap": getattr(args, "rl_q_cap", None),
            "teacher_eps0": getattr(args, "teacher_eps0", None),
            "teacher_decay": getattr(args, "teacher_decay", None),
            "survive_bonus": getattr(args, "survive_bonus", None),
            "u_scale": getattr(args, "u_scale", None),
            "lw_scale": getattr(args, "lw_scale", None),
            "tag": getattr(args, "tag", None),
            "alpha_mix": getattr(cfg, "alpha_mix", None),
            "h_FX": getattr(cfg, "h_FX", None),
        },
        ckpt_path=None,
        extra=extras_dict,
        timing=timing,
        time_total_s=timing["total_s"],
        time_total_hms=timing["total_hms"],
    )

    _inject_market_meta(cfg, args, out)

    if bh_spec is not None:
        try:
            out.setdefault("meta", {})
            out["meta"]["behavioral"] = _bh_describe(bh_spec)
        except Exception:
            pass

    metrics_csv = os.path.join(args.outputs, "_logs", "metrics.csv")
    meta = {
        "tag": getattr(args, "tag", None),
        "method": "rl",
        "asset": getattr(cfg, "asset", None),
        "outputs_abs": os.path.abspath(args.outputs),
        "time_total_hms": timing["total_hms"],
        "market_mode": getattr(cfg, "market_mode", None),
        "bootstrap_block": getattr(cfg, "bootstrap_block", None),
        "use_real_rf": getattr(cfg, "use_real_rf", None),
        "data_window": getattr(cfg, "data_window", None),
    }
    if bh_spec is not None:
        try:
            meta.update({f"bh_{k}": v for k, v in _bh_describe(bh_spec).items()})
        except Exception:
            pass

    try:
        write_metrics_csv(metrics_csv, args, out, meta=meta)
    except Exception:
        pass

    try:
        setattr(cfg, "method", "rl")
        setattr(cfg, "es_mode", es_mode)
        save_metrics_autocsv(out.get("metrics", {}), cfg, outputs=args.outputs)
    except Exception:
        pass

    if _onoff(getattr(args, "autosave", "off")) == "on":
        do_autosave(out.get("metrics") or {}, cfg, args, out)

    return out
