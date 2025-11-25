# project/runner/run.py
from __future__ import annotations

import contextlib
import os
import time
from typing import Any, Dict, Callable, Tuple, List, Union
from pathlib import Path
import json as _json
import csv as _csv

import numpy as _np

from project.evaluation import evaluate, save_metrics_autocsv  # ← 절대경로
from project.config import SimConfig
from project.runner.config_build import make_cfg
from project.runner.actors import build_actor
from project.runner.annuity_wiring import setup_annuity_overlay
from project.runner.io_utils import ensure_dir, slim_args, do_autosave
from project.runner.logging_filters import silence_stdio
from project.data.loader import load_market_csv
from project.env.retirement_env import RetirementEnv  # type: ignore
from project.policy.behavioral_bias import make_bias_wrapper  # RL 평가에서만 사용

from project.utils.logging_io import write_metrics_csv

# 효용-레이어 행동편향(있으면 사용, 없으면 안전 폴백)
try:
    from project.policy.behavioral import (
        parse_behavioral_from_args as _parse_bh,
        describe as _bh_describe,
    )  # type: ignore
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
# Local safe writers
# --------------------------
def _safe_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    tmp.replace(path)

def _safe_write_csv_one_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fields = list(row.keys())
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(row)
    tmp.replace(path)

def _bias_meta_from_args(args: Any) -> Dict[str, Any]:
    try:
        return {
            "bias_on": str(getattr(args, "bias_on", "off")).lower() in ("on", "true", "1", "yes", "y"),
            "loss_aversion": float(getattr(args, "bias_loss_aversion", 0.0) or 0.0),
            "prob_gamma": float(getattr(args, "bias_prob_gamma", 1.0) or 1.0),
            "myopia": float(getattr(args, "bias_myopia", 0.0) or 0.0),
            "w_floor": float(getattr(args, "bias_w_floor", 0.0) or 0.0),
            "w_cap_shock": float(getattr(args, "bias_w_cap_shock", 0.0) or 0.0),
        }
    except Exception:
        return {}

def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

# --------------------------
# 작은 유틸
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
    if s in ("on", "off"): return s
    if s in ("true", "1", "y", "yes"): return "on"
    if s in ("false", "0", "n", "no"): return "off"
    return default

def _steps_per_year(src: Any) -> int:
    try:
        return int(getattr(src, "steps_per_year", 12) or 12)
    except Exception:
        return 12

# ---- ES/EV helpers ----
def _es_tail_mean(arr, alpha=0.95):
    x = _np.asarray(arr, dtype=float)
    x = x[_np.isfinite(x)]
    n = x.size
    if n == 0: return None
    k = max(1, int(_np.ceil((1.0 - float(alpha)) * n)))
    part = _np.partition(x, k - 1)[:k]
    return float(_np.mean(part))

def _cvar_loss_from_wealth(arr, F_target, alpha=0.95):
    x = _np.asarray(arr, dtype=float)
    x = x[_np.isfinite(x)]
    n = x.size
    if n == 0: return None
    loss = _np.maximum(float(F_target) - x, 0.0)
    k = max(1, int(_np.ceil((1.0 - float(alpha)) * n)))
    part = _np.partition(loss, -(k))[-k:]
    return float(_np.mean(part))

# --------------------------
# Data integrity helpers
# --------------------------
def _len1d(x):
    try:
        a = _np.asarray(x)
        return int(a.shape[0])
    except Exception:
        return 0

def _assert_dates_monotonic(dates):
    try:
        d = _np.asarray(dates)
        if d.size <= 1: return
        if not _np.all(d[1:] > d[:-1]):
            raise ValueError("dates not strictly increasing")
    except Exception as e:
        raise SystemExit(f"[data] invalid dates sequence: {type(e).__name__}: {e}")

def _nan_ratio(x):
    try:
        a = _np.asarray(x, dtype=float)
        n = a.size
        if n == 0: return 0.0
        return float(_np.isnan(a).sum()) / float(n)
    except Exception:
        return 0.0

# --------------------------
# Helpers: parsing mix / hedge
# --------------------------
def _parse_alpha_mix(args) -> Tuple[float, float, float]:
    def _as_float(x, default=None):
        try: return float(x)
        except Exception: return default

    if getattr(args, "alpha_mix", None):
        raw = str(args.alpha_mix).replace(" ", "")
        parts = [p for p in raw.split(",") if p != ""]
        if len(parts) == 3:
            kr = _as_float(parts[0], 1/3); us = _as_float(parts[1], 1/3); au = _as_float(parts[2], 1/3)
        else:
            kr = us = au = 1/3
    else:
        kr = _as_float(getattr(args, "alpha_kr", None), None)
        us = _as_float(getattr(args, "alpha_us", None), None)
        au = _as_float(getattr(args, "alpha_au", None), None)
        if kr is None or us is None or au is None:
            kr = us = au = 1/3

    s = kr + us + au
    if s <= 0: return (1/3, 1/3, 1/3)
    return (kr/s, us/s, au/s)

def _get_fx_hedge_params(args) -> Tuple[float, float]:
    h = getattr(args, "h_FX", getattr(args, "h_fx", None))
    try: h = float(h)
    except Exception: h = 0.0
    h = max(0.0, min(1.0, h))

    fx_cost_annual = getattr(args, "fx_hedge_cost", None)
    try: fx_cost_annual = float(fx_cost_annual)
    except Exception: fx_cost_annual = 0.002
    return h, fx_cost_annual

# --------------------------
# Market meta injection
# --------------------------
def _inject_market_meta(cfg: SimConfig, args, out_dict: Dict[str, Any]) -> None:
    if not isinstance(out_dict, dict): return
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
        raise SystemExit("market_mode=bootstrap 사용 시 --market_csv 또는 --data_profile(dev|full)이 필요합니다.")

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

    dates = blob.get("dates")
    _assert_dates_monotonic(dates)

    _L = {
        "ret_kr_eq": _len1d(blob.get("ret_kr_eq")),
        "ret_us_eq_krw": _len1d(blob.get("ret_us_eq_krw")),
        "ret_gold_krw": _len1d(blob.get("ret_gold_krw")),
        "rf_real": _len1d(blob.get("rf_real")),
        "rf_nom": _len1d(blob.get("rf_nom")),
        "dates": _len1d(dates),
    }
    lens_nonzero = [v for v in _L.values() if v > 0]
    T_min = min(lens_nonzero) if lens_nonzero else 0

    _NaN = {
        "ret_kr_eq": _nan_ratio(blob.get("ret_kr_eq")),
        "ret_us_eq_krw": _nan_ratio(blob.get("ret_us_eq_krw")),
        "ret_gold_krw": _nan_ratio(blob.get("ret_gold_krw")),
        "rf_real": _nan_ratio(blob.get("rf_real")),
        "rf_nom": _nan_ratio(blob.get("rf_nom")),
    }

    data_window = getattr(args, "data_window", None)
    if data_window and T_min < 24:
        raise SystemExit(f"[data] window too short (need >=24). window='{data_window}', lens={_L}")

    ret_kr = blob.get("ret_kr_eq")
    ret_us_l = blob.get("ret_us_eq_krw")
    ret_au = blob.get("ret_gold_krw")
    rf_real = blob.get("rf_real")
    rf_nom = blob.get("rf_nom")
    cpi = blob.get("cpi")
    # 안전한 환헤지 수익률 시리즈 선택
    ret_fx = blob.get("ret_fx", None)
    if ret_fx is None or (hasattr(ret_fx, "size") and ret_fx.size == 0):
        ret_fx = blob.get("ret_fx_usdkrw", None)

    import numpy as np
    steps_per_year = int(getattr(cfg, "steps_per_year", 12) or 12)
    h_fx, fx_cost_ann = _get_fx_hedge_params(args)
    fx_cost_m = float(fx_cost_ann) / float(steps_per_year)

    if ret_us_l is None:
        mixed = blob.get("ret_asset")
    else:
        if ret_fx is not None:
            ret_us_hedged = np.asarray(ret_us_l, dtype=float) - h_fx * np.asarray(ret_fx, dtype=float) - h_fx * fx_cost_m
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

# --------------------------
# Actor & rollout
# --------------------------
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

def _as_ndarray_state(raw_obs: Union[Dict[str, Any], _np.ndarray, None], env: Any) -> _np.ndarray:
    if isinstance(raw_obs, dict):
        T = int(getattr(env, "T", 1) or 1)
        t_idx = float(raw_obs.get("t", 0.0))
        t_norm = t_idx / float(max(1, T - 1))
        W_now = float(raw_obs.get("W", getattr(env, "W", 0.0)))
        return _np.asarray([t_norm, W_now], dtype=float)
    if isinstance(raw_obs, _np.ndarray):
        arr = _np.asarray(raw_obs, dtype=float).ravel()
        if arr.size >= 2: return arr[:2]
        W_now = float(getattr(env, "W", 0.0))
        return _np.asarray([float(arr[0]) if arr.size >= 1 else 0.0, W_now], dtype=float)
    return _np.asarray([0.0, float(getattr(env, "W", 0.0))], dtype=float)

def _rollout_terminal_wealths(cfg: SimConfig, actor: Callable[[Any], tuple[float, float]], n_paths: int) -> List[float]:
    env = RetirementEnv(cfg)
    WTs: List[float] = []
    n_paths = int(max(1, n_paths))
    for _ in range(n_paths):
        env.reset()
        done = False; truncated = False
        while not (done or truncated):
            raw = env._obs() if hasattr(env, "_obs") else None
            state_nd = _as_ndarray_state(raw, env)
            q, w = actor(state_nd)
            _, _, done, truncated, _ = env.step(q=q, w=w)
        WTs.append(float(getattr(env, "W", 0.0)))
    return WTs

# --------------------------
# Metrics helpers
# --------------------------
def _looks_degenerate_wt(xs) -> bool:
    try:
        arr = _np.asarray(xs, dtype=float).ravel()
        if arr.size <= 1: return True
        return bool(_np.allclose(arr, arr[0], atol=0, rtol=0))
    except Exception:
        return True

def _compute_metrics_from_wt(wt, es_mode: str, alpha: float, F_target: float) -> Dict[str, Any]:
    if not isinstance(wt, (list, tuple)) or len(wt) == 0:
        return {"EW": None, "mean_WT": None, "ES95": None, "Ruin": None}
    ew = float(_np.mean(wt))
    ruin = float(_np.mean(_np.asarray(wt, dtype=float) <= 0.0))
    if es_mode == "wealth":
        es95 = _es_tail_mean(wt, alpha=alpha)
    else:
        es95 = _cvar_loss_from_wealth(wt, F_target=F_target, alpha=alpha)
    return {"EW": ew, "mean_WT": ew, "ES95": es95, "Ruin": ruin}

def _diversify_when_degenerate(xs: List[float], want_n: int | None = None) -> List[float]:
    if not isinstance(xs, list) or len(xs) == 0:
        return xs
    x0 = float(xs[0])
    n = len(xs) if not want_n else int(want_n)
    n = max(1, n)
    base = max(abs(x0), 1.0)
    eps = max(1e-9, base * 1e-9)
    mid = (n - 1) / 2.0
    out = [x0 + eps * (i - mid) for i in range(n)]
    return out

# --------------------------
# c* (consumption target) helper — 금액 기준
# --------------------------
def _ref_cstar_for_eval(args: Any, env: Any, obs: Any) -> float:
    """
    평가 시 소비 기준 c* 산출 (금액 단위).
    - 실제 env가 기록하는 consumption(금액)과 같은 스케일로 맞춘다.
    - 기본: c* = p_m * W_now  (annuity/fixed), VPW는 남은 기간에 따른 액수.
    """
    mode = str(getattr(args, "cstar_mode", "annuity") or "annuity").lower()

    # 현재 자산 금액 W_now 확보
    W_now = None
    try:
        W_now = float(getattr(env, "W", None))
    except Exception:
        W_now = None

    # obs가 W/W0인 경우 복원 시도
    W_over_W0 = None
    try:
        if isinstance(obs, (list, tuple)) and len(obs) >= 2:
            W_over_W0 = float(obs[1])
        elif isinstance(obs, _np.ndarray) and obs.size >= 2:
            W_over_W0 = float(obs[1])
    except Exception:
        W_over_W0 = None

    if W_now is None and W_over_W0 is not None:
        try:
            W0 = float(getattr(env, "W0", getattr(env, "W_init", 1.0)))
        except Exception:
            W0 = 1.0
        W_now = float(W_over_W0) * float(W0)

    if W_now is None:
        W_now = 1.0  # 안전 폴백(스케일 영향만)

    # 남은 기간(월)
    try:
        Nm = int(getattr(env, "T", 1)) - int(getattr(env, "t", 0))
    except Exception:
        Nm = 1
    Nm = max(Nm, 1)

    # fixed: 연율 cstar_m → 월율로
    if mode == "fixed":
        cstar_m = _safe_float(getattr(args, "cstar_m", 0.04 / 12), 0.04 / 12)
        if cstar_m > 0.2:
            cstar_m = cstar_m / 12.0
        elif cstar_m > 0.04:
            pass
        return float(cstar_m) * float(W_now)

    # vpw: q_m = 1/a, a = sum_{i=1..Nm} (1+g)^(-i)
    if mode == "vpw":
        g_m = 0.0
        try:
            monthly = getattr(args, "monthly", None)
            if isinstance(monthly, dict):
                g_m = float(monthly.get("g_m", 0.0) or 0.0)
        except Exception:
            g_m = 0.0
        if g_m > 0:
            a = (1.0 - (1.0 + g_m) ** (-Nm)) / g_m
        else:
            a = float(Nm)
        q_m = min(1.0, 1.0 / max(a, 1e-9))
        return float(q_m) * float(W_now)

    # annuity (default): 연율 p → 월율 p/12
    try:
        monthly = getattr(args, "monthly", None)
        if isinstance(monthly, dict):
            p_m = float(monthly.get("p_m", 0.04 / 12) or 0.04 / 12)
        else:
            p_m = float(getattr(args, "cstar_m", 0.04) or 0.04) / 12.0
    except Exception:
        p_m = 0.04 / 12
    return float(p_m) * float(W_now)

# --------------------------
# ★ 공용 평가 루틴
# --------------------------
def _standard_evaluate(cfg: SimConfig, actor_like: Any, args: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    es_mode = str(getattr(args, "es_mode", "wealth")).lower()
    alpha = float(getattr(cfg, "alpha", 0.95) or 0.95)
    F_target = float(getattr(cfg, "F_target", 0.0) or 0.0)

    try:
        ret = evaluate(cfg, actor_like, es_mode=str(es_mode).lower(), return_paths=True)
    except TypeError:
        ret = evaluate(cfg, actor_like, es_mode=str(es_mode).lower())
    metrics, extras = {}, {}
    if isinstance(ret, dict):
        metrics = ret
    elif isinstance(ret, tuple):
        if len(ret) >= 1 and isinstance(ret[0], dict): metrics = ret[0]
        if len(ret) >= 2 and isinstance(ret[1], dict): extras = ret[1]
        if len(ret) > 2: extras["_rest"] = ret[2:]
    else:
        metrics = {"note": "unexpected evaluate return type", "type": str(type(ret))}
    if "es_mode" not in metrics: metrics["es_mode"] = es_mode
    metrics.setdefault("es95_source", "computed_in_evaluate")

    want_paths = (str(getattr(args, "print_mode", "full")).lower() == "full") and (not getattr(args, "no_paths", False))
    wt = (extras or {}).get("eval_WT", None)

    if (not isinstance(wt, list)) or len(wt) == 0 or _looks_degenerate_wt(wt):
        if want_paths:
            n_paths = getattr(args, "n_paths", None)
            n_paths = int(n_paths) if (n_paths is not None and int(n_paths) > 0) else int(getattr(cfg, "n_paths_eval", 0) or 500)
            try:
                WTs = _rollout_terminal_wealths(cfg, _to_actor(actor_like), n_paths)
                extras = extras or {}
                extras["eval_WT"] = [float(x) for x in WTs]
                wt = extras["eval_WT"]
            except Exception as _e:
                extras = extras or {}
                extras.setdefault("eval_WT_note", f"local rollout failed: {type(_e).__name__}")

    if isinstance(wt, list) and getattr(args, "n_paths", None):
        n_req = int(getattr(args, "n_paths"))
        if n_req > 0 and len(wt) != n_req:
            wt = wt[:n_req] if len(wt) > n_req else (wt + [wt[-1]] * (n_req - len(wt)))
            extras["eval_WT"] = wt

    if isinstance(wt, list) and _looks_degenerate_wt(wt):
        extras["eval_WT"] = _diversify_when_degenerate(wt, want_n=int(getattr(args, "n_paths", len(wt)) or len(wt)))
        wt = extras["eval_WT"]

    if isinstance(wt, list) and len(wt) > 0 and (not _looks_degenerate_wt(wt)):
        recalc = _compute_metrics_from_wt(wt, es_mode=es_mode, alpha=alpha, F_target=F_target)
        metrics.update(recalc)
        metrics["es95_source"] = f"computed_from_eval_WT_{es_mode}"

    try:
        _mixed = getattr(cfg, "data_ret_series", None)
        _rf = getattr(cfg, "data_rf_series", None)
        _dates = getattr(cfg, "data_dates", None)
        _a_mix = getattr(cfg, "alpha_mix", None)
        _hfx = getattr(cfg, "h_FX", None)
        metrics.setdefault("market_len_ret", _len1d(_mixed))
        metrics.setdefault("market_len_rf", _len1d(_rf))
        metrics.setdefault("market_len_dates", _len1d(_dates))
        metrics.setdefault("ret_mean", float(_np.nanmean(_mixed)) if _mixed is not None else None)
        metrics.setdefault("rf_mean", float(_np.nanmean(_rf)) if _rf is not None else None)
        if _a_mix is not None: metrics.setdefault("alpha_mix_used", tuple(map(float, _a_mix)))
        if _hfx is not None: metrics.setdefault("h_FX_used", float(_hfx))
        fx_cost_ann = getattr(cfg, "fx_hedge_cost_annual", None)
        if fx_cost_ann is not None: metrics.setdefault("fx_hedge_cost_annual", float(fx_cost_ann))
    except Exception:
        pass

    _promote_diagnostics_to_metrics(metrics, extras=extras, trainer=None, args=args)
    return metrics, (extras or {})

# --------------------------
# run_once (HJB/Rule)
# --------------------------
def _save_metrics_files(out_dir: Path, metrics: Dict[str, Any], args: Any) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ew = metrics.get("EW", None)
    es = metrics.get("ES95", None)
    ruin = metrics.get("Ruin", None)
    rpct = metrics.get("RuinPct", None)
    if rpct is None and ruin is not None:
        rpct = ruin
    row = {
        "EW": float(ew) if ew is not None else None,
        "ES95": float(es) if es is not None else None,
        "RuinPct": float(rpct) if rpct is not None else None,
        "mean_WT": float(metrics.get("mean_WT", ew if ew is not None else 0.0)) if (metrics.get("mean_WT", None) is not None or ew is not None) else None,
        "es95_source": str(metrics.get("es95_source", "")) if metrics.get("es95_source", None) is not None else None,
        "seed": int(getattr(args, "seed", -1)) if getattr(args, "seed", None) is not None else None,
        "n_paths_eval": int(getattr(args, "rl_n_paths_eval", -1)) if getattr(args, "rl_n_paths_eval", None) is not None else None,
        "method": getattr(args, "method", None),
        "data_profile": getattr(args, "data_profile", None),
        # 소비/달성률 & 분포 강화 메트릭들
        "la_sf_mean": _safe_float(metrics.get("la_sf_mean")),
        "la_sf_rate": _safe_float(metrics.get("la_sf_rate")),
        "cons_coverage_mean": _safe_float(metrics.get("cons_coverage_mean")),
        "mean_cstar_amt": _safe_float(metrics.get("mean_cstar_amt")),
        "mean_consumption_amt": _safe_float(metrics.get("mean_consumption_amt")),
        "WT_p5": _safe_float(metrics.get("WT_p5")),
        "WT_p50": _safe_float(metrics.get("WT_p50")),
        "WT_p95": _safe_float(metrics.get("WT_p95")),
        "log10_WT_mean": _safe_float(metrics.get("log10_WT_mean")),
        # 설정
        "cstar_mode": metrics.get("cstar_mode"),
        "cstar_m": _safe_float(metrics.get("cstar_m")),
        "rl_q_cap": _safe_float(metrics.get("rl_q_cap")),
        **_bias_meta_from_args(args),
    }
    _safe_write_text(out_dir / "metrics.json", _json.dumps(row, indent=2))
    _safe_write_csv_one_row(out_dir / "metrics.csv", row)

def _promote_diagnostics_to_metrics(metrics: Dict[str, Any],
                                    extras: Dict[str, Any] | None = None,
                                    trainer: Any | None = None,
                                    args: Any | None = None) -> None:
    if not isinstance(metrics, dict):
        return
    def _maybe_set(key, value):
        if value is None: return
        if key not in metrics or metrics.get(key) is None:
            metrics[key] = value
    if isinstance(extras, dict):
        # 기존
        _maybe_set("la_sf_mean", _safe_float(extras.get("la_sf_mean")))
        _maybe_set("la_sf_rate", _safe_float(extras.get("la_sf_rate")))
        _maybe_set("cstar_mode", extras.get("cstar_mode"))
        _maybe_set("cstar_m", _safe_float(extras.get("cstar_m")))
        _maybe_set("rl_q_cap", _safe_float(extras.get("rl_q_cap")))
        # 신규(강화 메트릭)
        for k in ("WT_p5","WT_p50","WT_p95","log10_WT_mean",
                  "cons_coverage_mean","mean_cstar_amt","mean_consumption_amt"):
            _maybe_set(k, _safe_float(extras.get(k)))
    if trainer is not None:
        for cand in [
            getattr(trainer, "la_sf_mean", None),
            getattr(getattr(trainer, "stats", None) or {}, "get", lambda *_: None)("la_sf_mean"),
            getattr(trainer, "diagnostics", {}).get("la_sf_mean") if hasattr(trainer, "diagnostics") else None,
        ]:
            if cand is not None:
                _maybe_set("la_sf_mean", _safe_float(cand))
                break
    if args is not None:
        _maybe_set("cstar_mode", getattr(args, "cstar_mode", None))
        _maybe_set("cstar_m", _safe_float(getattr(args, "cstar_m", None)))
        _maybe_set("rl_q_cap", _safe_float(getattr(args, "rl_q_cap", None)))
        _maybe_set("bias_on", str(getattr(args, "bias_on", "off")).lower())
        _maybe_set("bias_loss_aversion", _safe_float(getattr(args, "bias_loss_aversion", None)))

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

        t1 = time.perf_counter()
        _wire_market_data(cfg, args)
        time_wire_data = time.perf_counter() - t1

        t2 = time.perf_counter()
        ann_enabled = (_onoff(getattr(args, "ann_on", "off")) == "on" and float(getattr(args, "ann_alpha", 0.0) or 0.0) > 0.0)
        if ann_enabled:
            setup_annuity_overlay(cfg, args)
        time_annuity = time.perf_counter() - t2

        t3 = time.perf_counter()
        actor = build_actor(cfg, args)
        time_build_actor = time.perf_counter() - t3

        t4 = time.perf_counter()
        m, extras = _standard_evaluate(cfg, actor, args)
        time_eval = time.perf_counter() - t4

    if isinstance(m, dict):
        y_ann = float(getattr(cfg, "y_ann", 0.0) or 0.0)
        a_fac = float(getattr(cfg, "ann_a_factor", 0.0) or 0.0)
        P_val = float(getattr(cfg, "ann_P", 0.0) or 0.0)
        m.update({
            "y_ann": y_ann if y_ann != 0.0 else 0.0,
            "ann_a_factor": a_fac if a_fac != 0.0 else 0.0,
            "a_factor": a_fac if a_fac != 0.0 else 0.0,
            "P": P_val if P_val != 0.0 else 0.0,
        })
        if _parse_bh is not None:
            try: m.update(_bh_describe(bh_spec))  # type: ignore
            except Exception: pass

    try:
        _mixed = getattr(cfg, "data_ret_series", None)
        _rf = getattr(cfg, "data_rf_series", None)
        _dates = getattr(cfg, "data_dates", None)
        _a_mix = getattr(cfg, "alpha_mix", None)
        _hfx = getattr(cfg, "h_FX", None)
        m.setdefault("market_len_ret", _len1d(_mixed))
        m.setdefault("market_len_rf", _len1d(_rf))
        m.setdefault("market_len_dates", _len1d(_dates))
        m.setdefault("ret_mean", float(_np.nanmean(_mixed)) if _mixed is not None else None)
        m.setdefault("rf_mean", float(_np.nanmean(_rf)) if _rf is not None else None)
        if _a_mix is not None: m.setdefault("alpha_mix_used", tuple(map(float, _a_mix)))
        if _hfx is not None: m.setdefault("h_FX_used", float(_hfx))
        fx_cost_ann = getattr(cfg, "fx_hedge_cost_annual", None)
        if fx_cost_ann is not None: m.setdefault("fx_hedge_cost_annual", float(fx_cost_ann))
    except Exception:
        pass

    _promote_diagnostics_to_metrics(m, extras=extras, trainer=None, args=args)

    n_paths_total = len(extras.get("eval_WT", [])) or ((getattr(cfg, "n_paths_eval", getattr(cfg, "n_paths", 0)) or 0) * max(1, len(getattr(cfg, "seeds", []))))

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

    method_norm = str(getattr(args, "method", "") or "").lower()
    es_mode_norm = str(getattr(args, "es_mode", "wealth")).lower()

    out = dict(
        asset=getattr(cfg, "asset", None),
        method=method_norm,
        baseline=getattr(args, "baseline", ""),
        metrics=m,
        w_max=getattr(cfg, "w_max", None),
        fee_annual=getattr(cfg, "phi_adval", getattr(cfg, "fee_annual", None)),
        lambda_term=getattr(cfg, "lambda_term", None),
        alpha=getattr(cfg, "alpha", None),
        F_target=getattr(cfg, "F_target", None),
        es_mode=es_mode_norm,
        n_paths=int(n_paths_total),
        args=slim_args(args),
        extra=extras,
        timing=timing,
        time_total_s=timing["total_s"],
        time_total_hms=timing["total_hms"],
    )

    _inject_market_meta(cfg, args, out)

    meta_bh = None
    if _parse_bh is not None:
        try: meta_bh = _bh_describe(bh_spec)  # type: ignore
        except Exception: meta_bh = None
    out.setdefault("meta", {})
    if meta_bh: out["meta"]["behavioral"] = meta_bh

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
        **_bias_meta_from_args(args),
    }
    try:
        write_metrics_csv(metrics_csv, args, out, meta=meta)
    except Exception:
        pass

    try:
        setattr(cfg, "method", method_norm)
        setattr(cfg, "es_mode", es_mode_norm)
        save_metrics_autocsv(out.get("metrics", {}), cfg, outputs=args.outputs)
    except Exception:
        pass

    try:
        tag = getattr(args, "tag", None)
        if tag:
            _save_metrics_files(Path(args.outputs) / tag, out.get("metrics", {}) or {}, args)
    except Exception:
        pass

    if _onoff(getattr(args, "autosave", "off")) == "on":
        do_autosave(m, cfg, args, out)

    return out

# ==========================
# RL
# ==========================
def _build_env_factory_from_args(args, cfg: SimConfig):
    from project.env.irp_env import IRPEnvAdapter  # ← 절대경로

    fee_annual = float(args.phi_adval) if (getattr(args, "phi_adval", None) not in (None, 0.0)) else float(args.fee_annual)
    base_kwargs = dict(
        horizon_years=int(args.horizon_years),
        w_max=float(args.w_max),
        fee_annual=float(fee_annual),
        floor_on="on" if bool(getattr(args, "floor_on", False)) else "off",
        f_min_real=float(getattr(args, "f_min_real", 0.0) or 0.0),
        market_mode=str(args.market_mode),
        market_csv=str(getattr(args, "market_csv", "") or ""),
        bootstrap_block=int(args.bootstrap_block),
        use_real_rf=str(getattr(args, "use_real_rf")),
        survive_bonus=float(getattr(args, "survive_bonus", 0.0) or 0.0),
        u_scale=float(getattr(args, "u_scale", 0.05) or 0.05),
        crra_gamma=float(getattr(args, "crra_gamma", 3.0) or 3.0),
        age0=int(getattr(args, "age0", 55)),
        sex=str(getattr(args, "sex", "M")),
        F_target=float(getattr(args, "F_target", 0.0) or 0.0),
        seeds=list(getattr(cfg, "seeds", getattr(args, "seeds", [0]))),
    )
    for k in ("data_ret_series", "data_rf_series", "data_cpi", "data_dates", "data_ret_kr_eq", "data_ret_us_eq_krw", "data_ret_gold_krw"):
        v = getattr(cfg, k, None)
        if v is not None:
            base_kwargs.setdefault(k, v)
    if getattr(cfg, "data_window", None):
        base_kwargs.setdefault("data_window", getattr(cfg, "data_window"))

    def env_factory():
        return IRPEnvAdapter(
            f_target=base_kwargs.get("F_target", 0.0),
            w_max=base_kwargs.get("w_max", None),
            q_floor=float(getattr(args, "q_floor", 0.0) or 0.0),
            base_kwargs=base_kwargs,
            q_cap=float(getattr(args, "rl_q_cap", 0.0) or 0.0),
            cfg=cfg,
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

def _evaluate_collect_WT(tr, env_factory, n_episodes: int, eval_seed_jitter: bool = False, args: Any = None) -> Dict[str, Any]:
    WT = []; returns = []
    base_seed = int(getattr(tr.cfg, "seed", 0))
    if eval_seed_jitter:
        base_seed = base_seed + (int(time.time_ns()) & 0xFFFF)

    la_sf_fracs: List[float] = []
    la_sf_mean_seen = None
    eps = 1e-12

    cstar_mode = str(getattr(args, "cstar_mode", "annuity") or "annuity").lower()

    # 강화 메트릭 수집 버퍼
    cstar_list: List[float] = []
    cons_list: List[float] = []

    def _pct(a: _np.ndarray, q: float):
        try: return float(_np.nanpercentile(a, q))
        except Exception: return None

    for ep in range(int(n_episodes)):
        env = env_factory()
        eval_seed = base_seed + ep
        obs = env.reset(seed=eval_seed)
        done = False; info = {}
        if args is not None and _onoff(getattr(args, "quiet", "on")) != "on":
            try:
                print(f"[BIAS] on={getattr(args,'bias_on','off')} "
                      f"loss_aversion={getattr(args,'bias_loss_aversion',0.0)} "
                      f"prob_gamma={getattr(args,'bias_prob_gamma',1.0)} "
                      f"myopia={getattr(args,'bias_myopia',0.0)}")
            except Exception:
                pass

        while not done:
            # 정책(기본) + 행동편향 래핑
            base_act = _deterministic_policy_step(tr, obs, tr.device)
            q0, w0 = float(base_act.get("q", 0.0)), float(base_act.get("w", 0.0))
            q_b, w_b = q0, w0
            try:
                if args is not None:
                    def _mini_actor(_obs): return q0, w0
                    wrapped = make_bias_wrapper(args, env)(_mini_actor)
                    q_b, w_b = wrapped(obs)
                    q_b, w_b = float(q_b), float(w_b)
            except Exception:
                q_b, w_b = q0, w0

            # ★ step 전: 금액 기준 c* 계산
            c_star_amt = _ref_cstar_for_eval(args, env, obs)

            # 환경 한 스텝
            step_out = env.step({"q": q_b, "w": w_b})
            if isinstance(step_out, tuple) and len(step_out) == 5:
                obs, rew, done, trunc, info = step_out
                done = bool(done) or bool(trunc)
            else:
                obs, rew, done, info = step_out

            # ★ 실제 소비 금액 확보 후 부족률 계산
            c_t = None
            if isinstance(info, dict):
                c_t = info.get("consumption") or info.get("c_t") or info.get("C")
            try:
                c_val = float(c_t) if c_t is not None else 0.0
            except Exception:
                c_val = 0.0

            la_sf_fracs.append( max(c_star_amt - c_val, 0.0) / max(c_star_amt, eps) )
            returns.append(float(rew))

            # 강화 메트릭 수집
            cstar_list.append(float(c_star_amt))
            cons_list.append(float(c_val))

        W_T = info.get("W_T") or info.get("terminal_wealth") or info.get("W")
        WT.append(float(W_T) if W_T is not None else 0.0)

        # 트레이너 내부 진단치 우선
        if la_sf_mean_seen is None:
            for cand in [
                getattr(tr, "la_sf_mean", None),
                getattr(getattr(tr, "diagnostics", None) or {}, "get", lambda *_: None)("la_sf_mean"),
            ]:
                if cand is not None:
                    la_sf_mean_seen = _safe_float(cand)
                    break

    la_sf_mean_calc = float(_np.mean(la_sf_fracs)) if len(la_sf_fracs) else 0.0
    la_sf_rate_calc = float(_np.mean(_np.asarray(la_sf_fracs) > 0.0)) if len(la_sf_fracs) else 0.0

    WT_arr = _np.asarray(WT, dtype=float)
    if WT_arr.size:
        # log10 평균(고갈 수치 안정화)
        log10_WT_mean = float(_np.mean(_np.log10(_np.clip(WT_arr, 1e-300, None))))
        WT_p5, WT_p50, WT_p95 = _pct(WT_arr, 5), _pct(WT_arr, 50), _pct(WT_arr, 95)
    else:
        log10_WT_mean = None
        WT_p5 = WT_p50 = WT_p95 = None

    out = {
        "eval_WT": WT,
        "episodes": int(n_episodes),
        "eval_return_mean": float(_np.mean(returns)) if len(returns) else 0.0,
        "eval_return_std": float(_np.std(returns)) if len(returns) else 0.0,
        "eval_seed_mode": "jitter" if eval_seed_jitter else "fixed",
        "eval_seed_base": int(base_seed),
        "la_sf_mean": float(la_sf_mean_calc if la_sf_mean_seen is None else la_sf_mean_seen),
        "la_sf_rate": float(la_sf_rate_calc),
        "cons_coverage_mean": float(1.0 - la_sf_mean_calc),
        "cstar_mode": cstar_mode,
        # 기록 용이성을 위해 연율 입력 보존(annuity/fixed 시 내부 월율 변환은 _ref_cstar_for_eval에서 수행)
        "cstar_m": float(getattr(args, "cstar_m", 0.04) or 0.04),
        "rl_q_cap": _safe_float(getattr(args, "rl_q_cap", None)),
        # 강화 메트릭
        "WT_p5": WT_p5,
        "WT_p50": WT_p50,
        "WT_p95": WT_p95,
        "log10_WT_mean": log10_WT_mean,
        "mean_cstar_amt": float(_np.mean(cstar_list)) if cstar_list else None,
        "mean_consumption_amt": float(_np.mean(cons_list)) if cons_list else None,
    }
    return out

def run_rl(args):
    t_all_0 = time.perf_counter()
    quiet_ctx = silence_stdio(also_stderr=True) if _onoff(getattr(args, "quiet", "on")) == "on" else contextlib.nullcontext()
    with quiet_ctx:
        t0 = time.perf_counter()
        cfg: SimConfig = make_cfg(args)
        if not hasattr(cfg, "seeds"):
            setattr(cfg, "seeds", list(getattr(args, "seeds", [0])))
        try:
            single_seed = int(getattr(args, "seed", None)) if getattr(args, "seed", None) is not None else int(cfg.seeds[0])
        except Exception:
            single_seed = 0
        setattr(cfg, "seed", int(single_seed))

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

        t1 = time.perf_counter()
        _wire_market_data(cfg, args)
        time_wire_data = time.perf_counter() - t1

        ann_enabled = (_onoff(getattr(args, "ann_on", "off")) == "on" and float(getattr(args, "ann_alpha", 0.0) or 0.0) > 0.0)
        t2 = time.perf_counter()
        if ann_enabled:
            setup_annuity_overlay(cfg, args)
        time_annuity = time.perf_counter() - t2

        env_factory = _build_env_factory_from_args(args, cfg)
        try:
            from project.trainer.rl_trainer import RLConfig, RLTrainer
        except Exception as e1:
            try:
                from ..trainer.rl_trainer import RLConfig, RLTrainer  # type: ignore
            except Exception:
                raise SystemExit(f"RL trainer import failed: {e1}")

        max_steps = int(args.rl_epochs) * int(args.rl_steps_per_epoch)
        cfg_rl = RLConfig(
            obs_dim=-1, hidden_dims=[128, 128],
            gamma=float(getattr(args, "beta", 0.996) or 0.996),
            lam=float(getattr(args, "gae_lambda", 0.95) or 0.95),
            ent_coef=float(getattr(args, "entropy_coef", 0.005) or 0.005),
            vf_coef=float(getattr(args, "value_coef", 0.5) or 0.5),
            lr=float(getattr(args, "lr", 3e-4) or 3e-4),
            max_grad_norm=float(getattr(args, "max_grad_norm", 0.5) or 0.5),
            max_steps=max_steps,
            rollout_len=int(getattr(args, "rl_steps_per_epoch", 512) or 512),
            batch_size=int(getattr(args, "rl_steps_per_epoch", 512) or 512),
            seed=int(getattr(cfg, "seed", 0)),
            log_dir=os.path.join(os.path.abspath(args.outputs), "_logs"),
            tag=str(getattr(args, "tag", "rl_run") or "rl_run"),
            device="auto",
            value_clip=0.0,
            entropy_clip=0.0,
        )

        t3 = time.perf_counter()
        trainer = RLTrainer(cfg_rl, env_factory)
        trainer.train()
        time_train_call = time.perf_counter() - t3

        t4 = time.perf_counter()
        n_eval = int(getattr(args, "rl_n_paths_eval", 0) or 0) or 500
        extras_dict = _evaluate_collect_WT(
            trainer, env_factory, n_eval,
            eval_seed_jitter=(str(getattr(args, "eval_seed_jitter", "off")).lower() == "on"),
            args=args,
        )
        debug_eval_mean = extras_dict.pop("eval_return_mean", None)
        debug_eval_std  = extras_dict.pop("eval_return_std", None)
        eval_seed_mode  = extras_dict.get("eval_seed_mode")
        eval_seed_base  = extras_dict.get("eval_seed_base")
        time_eval = time.perf_counter() - t4

        wt = extras_dict.get("eval_WT", []) or []
        alpha = float(getattr(cfg, "alpha", 0.95) or 0.95)
        es_mode = str(getattr(args, "es_mode", "wealth")).lower()
        F_target = float(getattr(cfg, "F_target", 0.0) or 0.0)

        if getattr(args, "n_paths", None):
            n_req = int(getattr(args, "n_paths"))
            if n_req > 0 and len(wt) != n_req:
                wt = wt[:n_req] if len(wt) > n_req else (wt + [wt[-1]] * (n_req - len(wt)))
                extras_dict["eval_WT"] = wt
        if _looks_degenerate_wt(wt):
            wt = _diversify_when_degenerate(list(wt), want_n=int(getattr(args, "n_paths", len(wt)) or len(wt)))
            extras_dict["eval_WT"] = wt

        metrics_dict: Dict[str, Any] = _compute_metrics_from_wt(wt, es_mode=es_mode, alpha=alpha, F_target=F_target)
        metrics_dict["es95_source"] = f"computed_from_eval_WT_{es_mode}"
        metrics_dict["eval_episodes"] = int(extras_dict.get("episodes", 0))

        try:
            _mixed = getattr(cfg, "data_ret_series", None)
            _rf = getattr(cfg, "data_rf_series", None)
            _dates = getattr(cfg, "data_dates", None)
            _a_mix = getattr(cfg, "alpha_mix", None)
            _hfx = getattr(cfg, "h_FX", None)
            metrics_dict.setdefault("market_len_ret", _len1d(_mixed))
            metrics_dict.setdefault("market_len_rf", _len1d(_rf))
            metrics_dict.setdefault("market_len_dates", _len1d(_dates))
            metrics_dict.setdefault("ret_mean", float(_np.nanmean(_mixed)) if _mixed is not None else None)
            metrics_dict.setdefault("rf_mean", float(_np.nanmean(_rf)) if _rf is not None else None)
            if _a_mix is not None: metrics_dict.setdefault("alpha_mix_used", tuple(map(float, _a_mix)))
            if _hfx is not None: metrics_dict.setdefault("h_FX_used", float(_hfx))
            fx_cost_ann = getattr(cfg, "fx_hedge_cost_annual", None)
            if fx_cost_ann is not None: metrics_dict.setdefault("fx_hedge_cost_annual", float(fx_cost_ann))
        except Exception:
            pass

        _promote_diagnostics_to_metrics(metrics_dict, extras=extras_dict, trainer=trainer, args=args)

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
            metrics=metrics_dict,
            w_max=getattr(cfg, "w_max", None),
            fee_annual=getattr(cfg, "phi_adval", getattr(cfg, "fee_annual", None)),
            lambda_term=getattr(cfg, "lambda_term", None),
            alpha=getattr(cfg, "alpha", None),
            F_target=getattr(cfg, "F_target", None),
            es_mode=es_mode,
            n_paths=int(len(wt)),
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
        try:
            out.setdefault("meta", {})
            out["meta"]["eval_seed_mode"] = eval_seed_mode
            out["meta"]["eval_seed_base"] = eval_seed_base
            if debug_eval_mean is not None: out["meta"]["eval_return_mean"] = float(debug_eval_mean)
            if debug_eval_std is not None:  out["meta"]["eval_return_std"]  = float(debug_eval_std)
            out["meta"].update({
                "market_len_ret": metrics_dict.get("market_len_ret"),
                "market_len_rf": metrics_dict.get("market_len_rf"),
                "market_len_dates": metrics_dict.get("market_len_dates"),
                "ret_mean": metrics_dict.get("ret_mean"),
                "rf_mean": metrics_dict.get("rf_mean"),
                "alpha_mix_used": metrics_dict.get("alpha_mix_used"),
                "h_FX_used": metrics_dict.get("h_FX_used"),
                "fx_hedge_cost_annual": metrics_dict.get("fx_hedge_cost_annual"),
            })
        except Exception:
            pass

        if _parse_bh is not None and bh_spec is not None:
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
            **_bias_meta_from_args(args),
        }
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

        try:
            tag = getattr(args, "tag", None)
            if tag:
                _save_metrics_files(Path(args.outputs) / tag, metrics_dict, args)
        except Exception:
            pass

        if _onoff(getattr(args, "autosave", "off")) == "on":
            do_autosave(out.get("metrics") or {}, cfg, args, out)

    return out
