# run.py (refactored)
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
    """액션-레이어 편향 파라미터를 metrics 파일에 기록하기 위한 helper."""
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

def _save_metrics_files(out_dir: Path, metrics: Dict[str, Any], args: Any) -> None:
    """
    outputs/<tag>/metrics.json & metrics.csv 생성.
    - Ruin만 있을 때 RuinPct 자동 보정
    - seed/n_paths_eval/편향 파라미터 등 메타 포함
    """
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
        # 액션-레이어 편향 파라미터 메타 추가
        **_bias_meta_from_args(args),
    }
    _safe_write_text(out_dir / "metrics.json", _json.dumps(row, indent=2))
    _safe_write_csv_one_row(out_dir / "metrics.csv", row)


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


# ---- ES/EV helpers ----
def _es_tail_mean(arr, alpha=0.95):
    """Wealth 모드 ES: 하위 (1-α) 구간 평균."""
    x = _np.asarray(arr, dtype=float)
    x = x[_np.isfinite(x)]
    n = x.size
    if n == 0:
        return None
    k = max(1, int(_np.ceil((1.0 - float(alpha)) * n)))
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
        if d.size <= 1:
            return
        if not _np.all(d[1:] > d[:-1]):
            raise ValueError("dates not strictly increasing")
    except Exception as e:
        raise SystemExit(f"[data] invalid dates sequence: {type(e).__name__}: {e}")

def _nan_ratio(x):
    try:
        a = _np.asarray(x, dtype=float)
        n = a.size
        if n == 0:
            return 0.0
        return float(_np.isnan(a).sum()) / float(n)
    except Exception:
        return 0.0


# --------------------------
# Helpers: parsing mix / hedge
# --------------------------
def _parse_alpha_mix(args) -> Tuple[float, float, float]:
    """alpha 믹스를 (--alpha_mix 'kr,us,au' | 세 개의 개별 플래그)에서 읽어 정규화."""
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
        return (1 / 3, 1 / 3, 1 / 3)
    return (kr / s, us / s, au / s)

def _get_fx_hedge_params(args) -> Tuple[float, float]:
    """환헤지 비중 h_FX(0~1), 연간 헤지비용을 읽어 반환."""
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
            print(
                f"[market] mode={market_mode}, block={bootstrap_block}, use_real_rf={use_real_rf}, "
                f"window='{data_window or ''}', profile='{data_profile or ''}', csv='{market_csv or ''}'"
            )
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

    # === integrity checks & quick stats ===
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
        raise SystemExit(
            f"[data] window too short (need >=24). window='{data_window}', lens={_L}"
        )

    ret_kr = blob.get("ret_kr_eq")
    ret_us_l = blob.get("ret_us_eq_krw")
    ret_au = blob.get("ret_gold_krw")
    rf_real = blob.get("rf_real")
    rf_nom = blob.get("rf_nom")
    cpi = blob.get("cpi")
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
        if isinstance(kr, np.ndarray):
            kr = kr[:T]
        if isinstance(us, np.ndarray):
            us = us[:T]
        if isinstance(au, np.ndarray):
            au = au[:T]
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
                "[data] "
                f"len_total={len(mixed) if mixed is not None else 0}, "
                f"ret_mean={ret_mean:.4f}, rf_mean={rf_mean:.4f}, "
                f"h_FX={getattr(cfg,'h_FX',0.0):.2f}, {a_str}, "
                f"asset={getattr(cfg, 'asset', '?')}, window={getattr(cfg, 'data_window', None)}"
            )
            print("[data] lens=", _L, " NaN%=", {k: round(v * 100, 3) for k, v in _NaN.items()})
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


def _rollout_terminal_wealths(cfg: SimConfig, actor: Callable[[Any], tuple[float, float]], n_paths: int) -> List[float]:
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


def _compute_metrics_from_wt(wt, es_mode: str, alpha: float, F_target: float) -> Dict[str, Any]:
    """WT 샘플에서 EW/ES95/Ruin 재계산(wealth/loss 모드 모두 지원)."""
    out = {}
    if not isinstance(wt, (list, tuple)) or len(wt) == 0:
        return {"EW": None, "mean_WT": None, "ES95": None, "Ruin": None}
    ew = float(_np.mean(wt))
    ruin = float(_np.mean(_np.asarray(wt, dtype=float) <= 0.0))
    if es_mode == "wealth":
        es95 = _es_tail_mean(wt, alpha=alpha)
    else:
        es95 = _cvar_loss_from_wealth(wt, F_target=F_target, alpha=alpha)
    out.update({"EW": ew, "mean_WT": ew, "ES95": es95, "Ruin": ruin})
    return out


# --------------------------
# ★ 공용 평가 루틴
# --------------------------
def _standard_evaluate(cfg: SimConfig, actor_like: Any, args: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    모든 방법론(rule/hjb/rl)에서 동일 포맷으로 평가·지표 산출:
      1) evaluate(..., return_paths=True) 시도
      2) eval_WT 있으면 EW/ES95/Ruin을 eval_WT 기준으로 재계산
      3) 없거나 퇴화 시 local rollout 보조 (print_mode=full & no_paths=False일 때만)
    """
    es_mode = str(getattr(args, "es_mode", "wealth")).lower()
    alpha = float(getattr(cfg, "alpha", 0.95) or 0.95)
    F_target = float(getattr(cfg, "F_target", 0.0) or 0.0)

    m, extras = _call_evaluate(cfg, actor_like, es_mode=es_mode)

    need_paths = (str(getattr(args, "print_mode", "full")).lower() == "full") and (not getattr(args, "no_paths", False))
    wt_from_eval = (extras or {}).get("eval_WT", None)

    if (not isinstance(wt_from_eval, (list, tuple))) or _looks_degenerate_wt(wt_from_eval):
        if need_paths:
            n_paths = getattr(args, "n_paths", None)
            if n_paths is None or int(n_paths) <= 0:
                n_paths = int(getattr(cfg, "n_paths_eval", 0)) or 500  # 기본 500으로 통일
            try:
                WTs = _rollout_terminal_wealths(cfg, _to_actor(actor_like), int(n_paths))
                extras = extras or {}
                extras["eval_WT"] = [float(x) for x in WTs]
                wt_from_eval = extras["eval_WT"]
            except Exception as _e:
                extras = extras or {}
                extras.setdefault("eval_WT_note", f"local rollout failed: {type(_e).__name__}")

    # WT가 있으면 항상 재계산
    if isinstance(wt_from_eval, (list, tuple)) and len(wt_from_eval) > 0 and (not _looks_degenerate_wt(wt_from_eval)):
        recalc = _compute_metrics_from_wt(wt_from_eval, es_mode=es_mode, alpha=alpha, F_target=F_target)
        m.update(recalc)
        m["es95_source"] = f"computed_from_eval_WT_{es_mode}"

    # 시장/믹스 메타 복제
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
        if _a_mix is not None:
            m.setdefault("alpha_mix_used", tuple(map(float, _a_mix)))
        if _hfx is not None:
            m.setdefault("h_FX_used", float(_hfx))
        fx_cost_ann = getattr(cfg, "fx_hedge_cost_annual", None)
        if fx_cost_ann is not None:
            m.setdefault("fx_hedge_cost_annual", float(fx_cost_ann))
    except Exception:
        pass

    return m, (extras or {})


def run_once(args) -> Dict[str, Any]:
    """
    rule/hjb 경로:
      - actors.build_actor가 필요 시 편향 래퍼 단일 적용(이중 적용 방지).
      - 평가는 ★공용 루틴(_standard_evaluate)★로 강제.
    """
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
        actor = build_actor(cfg, args)   # ← actors.build_actor가 필요 시 래퍼 단일 적용
        time_build_actor = time.perf_counter() - t3

        # ★ 공용 평가
        t4 = time.perf_counter()
        m, extras = _standard_evaluate(cfg, actor, args)
        time_eval = time.perf_counter() - t4

    # ann 메타/행동편향 메타
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
            try:
                m.update(_bh_describe(bh_spec))  # type: ignore
            except Exception:
                pass

    # 시장/믹스 메타도 metrics에 일부 복제(요약 스크립트 편의)
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
        if _a_mix is not None:
            m.setdefault("alpha_mix_used", tuple(map(float, _a_mix)))
        if _hfx is not None:
            m.setdefault("h_FX_used", float(_hfx))
        fx_cost_ann = getattr(cfg, "fx_hedge_cost_annual", None)
        if fx_cost_ann is not None:
            m.setdefault("fx_hedge_cost_annual", float(fx_cost_ann))
    except Exception:
        pass

    # n_paths 계산(가능한 한 실제 WT 샘플 수로)
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
        try:
            meta_bh = _bh_describe(bh_spec)  # type: ignore
        except Exception:
            meta_bh = None
    out.setdefault("meta", {})
    if meta_bh:
        out["meta"]["behavioral"] = meta_bh

    # 중앙 로그 메타
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
        # 액션-레이어 편향 메타도 중앙 로그에 얹기
        **_bias_meta_from_args(args),
    }

    # (A) 중앙 로그 파일 갱신
    try:
        write_metrics_csv(metrics_csv, args, out, meta=meta)
    except Exception:
        pass

    # (B) 기존 자동 CSV 경로 유지
    try:
        setattr(cfg, "method", method_norm)
        setattr(cfg, "es_mode", es_mode_norm)
        save_metrics_autocsv(out.get("metrics", {}), cfg, outputs=args.outputs)
    except Exception:
        pass

    # (C) outputs/<tag>/metrics.json & metrics.csv 강제 생성 (+편향 메타)
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
# RL (RLTrainer 경로, 절대임포트 고정)
# ==========================
def _build_env_factory_from_args(args, cfg: SimConfig):
    """
    IRPEnvAdapter(env) + RetirementEnv kwargs 전달.
    cfg를 클로저로 캡처하여 RetirementEnv(cfg, **kwargs) 경로를 강제.
    """
    from project.env.irp_env import IRPEnvAdapter  # ← 절대경로

    fee_annual = float(args.phi_adval) if (getattr(args, "phi_adval", None) not in (None, 0.0)) else float(args.fee_annual)

    # RetirementEnv가 직접 받는 값들은 base_kwargs에 넣는다.
    base_kwargs = dict(
        horizon_years=int(args.horizon_years),
        w_max=float(args.w_max),
        fee_annual=float(fee_annual),
        floor_on="on" if bool(getattr(args, "floor_on", False)) else "off",
        f_min_real=float(getattr(args, "f_min_real", 0.0) or 0.0),
        market_mode=str(args.market_mode),
        market_csv=str(getattr(args, "market_csv", "") or ""),
        bootstrap_block=int(args.bootstrap_block),
        use_real_rf=str(args.use_real_rf),
        survive_bonus=float(getattr(args, "survive_bonus", 0.0) or 0.0),
        u_scale=float(getattr(args, "u_scale", 0.05) or 0.05),
        crra_gamma=float(getattr(args, "crra_gamma", 3.0) or 3.0),
        age0=int(getattr(args, "age0", 65)),
        sex=str(getattr(args, "sex", "M")),
        F_target=float(getattr(args, "F_target", 0.0) or 0.0),
        seeds=list(getattr(cfg, "seeds", getattr(args, "seeds", [0]))),
    )

    # cfg에 주입된 데이터 시리즈가 있을 경우 base_kwargs에도 복사
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
            cfg=cfg,  # ★ 핵심: cfg 전달
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
    """
    RL 평가 루프.
    - trainer.actor에서 얻은 (q,w)에 대해 행동편향 래퍼(make_bias_wrapper)를 에피소드별 env에 장착하여 적용.
    - 학습 단계에는 편향을 적용하지 않고, 평가 시에만 적용(명시적 설계).
    """
    WT = []
    returns = []
    base_seed = int(getattr(tr.cfg, "seed", 0))

    if eval_seed_jitter:
        base_seed = base_seed + (int(time.time_ns()) & 0xFFFF)

    for ep in range(int(n_episodes)):
        env = env_factory()
        eval_seed = base_seed + ep
        obs = env.reset(seed=eval_seed)
        done = False
        ret_sum = 0.0
        info = {}

        if args is not None and _onoff(getattr(args, "quiet", "on")) != "on":
            try:
                print(f"[BIAS] on={getattr(args,'bias_on','off')} "
                      f"loss_aversion={getattr(args,'bias_loss_aversion',0.0)} "
                      f"prob_gamma={getattr(args,'bias_prob_gamma',1.0)} "
                      f"myopia={getattr(args,'bias_myopia',0.0)}")
            except Exception:
                pass

        while not done:
            # 1) 기본 행위자(트레이너) 결정
            base_act = _deterministic_policy_step(tr, obs, tr.device)  # dict{"q","w"}
            q0, w0 = float(base_act.get("q", 0.0)), float(base_act.get("w", 0.0))

            # 2) 행동편향 래퍼 적용
            q_b, w_b = q0, w0
            try:
                if args is not None:
                    def _mini_actor(_obs):
                        return q0, w0
                    wrapped = make_bias_wrapper(args, env)(_mini_actor)
                    q_b, w_b = wrapped(obs)
                    q_b, w_b = float(q_b), float(w_b)
            except Exception:
                q_b, w_b = q0, w0  # 안전 폴백

            # 3) env.step — 기존 dict 인터페이스 유지
            step_out = env.step({"q": q_b, "w": w_b})
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
        "episodes": int(n_episodes),
        # 디버그 값은 메타로만 유지하고 최종 extras에서는 제거할 것
        "eval_return_mean": float(_np.mean(returns)) if len(returns) else 0.0,
        "eval_return_std": float(_np.std(returns)) if len(returns) else 0.0,
        "eval_seed_mode": "jitter" if eval_seed_jitter else "fixed",
        "eval_seed_base": int(base_seed),
    }
    return out


def run_rl(args):
    """
    RL 경로:
      - 학습(train)에는 액션-레이어 편향을 적용하지 않음.
      - 평가(evaluate) 시에만 make_bias_wrapper(args, env)로 보정된 (q,w)를 사용.
      - ★공용 산출 형식★: eval_WT 기반 EW/ES95/Ruin 재계산 & 메타 주입은 rule/hjb와 동일.
      - 디버그 수치(eval_return_mean/std)는 최종 extras에서 제거(메타로만 보존).
    """
    t_all_0 = time.perf_counter()

    quiet_ctx = silence_stdio(also_stderr=True) if _onoff(getattr(args, "quiet", "on")) == "on" else contextlib.nullcontext()

    with quiet_ctx:
        # 0) CFG 구성 및 시드 단일화
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

        # 출력 디렉토리 및 태그
        ensure_dir(args.outputs)
        if getattr(args, "tag", None) is not None:
            setattr(cfg, "tag", args.tag)

        # 1) 데이터 배선 + 연금 오버레이
        t1 = time.perf_counter()
        _wire_market_data(cfg, args)
        time_wire_data = time.perf_counter() - t1

        ann_enabled = (_onoff(getattr(args, "ann_on", "off")) == "on" and float(getattr(args, "ann_alpha", 0.0) or 0.0) > 0.0)
        t2 = time.perf_counter()
        if ann_enabled:
            setup_annuity_overlay(cfg, args)
        time_annuity = time.perf_counter() - t2

        # 2) Env factory
        env_factory = _build_env_factory_from_args(args, cfg)

        # 3) RLTrainer 로드 & 설정
        try:
            from project.trainer.rl_trainer import RLConfig, RLTrainer
        except Exception as e1:
            try:
                from ..trainer.rl_trainer import RLConfig, RLTrainer  # type: ignore
            except Exception:
                raise SystemExit(f"RL trainer import failed: {e1}")

        max_steps = int(args.rl_epochs) * int(args.rl_steps_per_epoch)
        cfg_rl = RLConfig(
            obs_dim=-1,
            hidden_dims=[128, 128],
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

        # 4) 학습
        t3 = time.perf_counter()
        trainer = RLTrainer(cfg_rl, env_factory)
        trainer.train()
        time_train_call = time.perf_counter() - t3

        # 5) 평가 (WT 수집 + ES/EW 산출) — ★ RL도 공용 지표 스펙으로 산출
        t4 = time.perf_counter()
        n_eval = int(getattr(args, "rl_n_paths_eval", 0) or 0)
        if n_eval <= 0:
            n_eval = 500  # 기본 500으로 통일
        extras_dict = _evaluate_collect_WT(
            trainer,
            env_factory,
            n_eval,
            eval_seed_jitter=(str(getattr(args, "eval_seed_jitter", "off")).lower() == "on"),
            args=args,
        )
        # 디버그 값은 최종 extras에서 제거 (메타로만 남김)
        debug_eval_mean = extras_dict.pop("eval_return_mean", None)
        debug_eval_std  = extras_dict.pop("eval_return_std", None)
        eval_seed_mode  = extras_dict.get("eval_seed_mode")
        eval_seed_base  = extras_dict.get("eval_seed_base")
        time_eval = time.perf_counter() - t4

        # ---- metrics from eval_WT (항상 재계산) ----
        wt = extras_dict.get("eval_WT", []) or []
        alpha = float(getattr(cfg, "alpha", 0.95) or 0.95)
        es_mode = str(getattr(args, "es_mode", "wealth")).lower()
        F_target = float(getattr(cfg, "F_target", 0.0) or 0.0)

        metrics_dict: Dict[str, Any] = _compute_metrics_from_wt(wt, es_mode=es_mode, alpha=alpha, F_target=F_target)
        metrics_dict["es95_source"] = f"computed_from_eval_WT_{es_mode}"
        metrics_dict["eval_episodes"] = int(extras_dict.get("episodes", 0))

        # 시장 통계 일부도 metrics에
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
            if _a_mix is not None:
                metrics_dict.setdefault("alpha_mix_used", tuple(map(float, _a_mix)))
            if _hfx is not None:
                metrics_dict.setdefault("h_FX_used", float(_hfx))
            fx_cost_ann = getattr(cfg, "fx_hedge_cost_annual", None)
            if fx_cost_ann is not None:
                metrics_dict.setdefault("fx_hedge_cost_annual", float(fx_cost_ann))
        except Exception:
            pass

        # 6) 결과 패키징
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

        # 시장/평가 시드 메타 + 편향 메타
        _inject_market_meta(cfg, args, out)
        try:
            out.setdefault("meta", {})
            out["meta"]["eval_seed_mode"] = eval_seed_mode
            out["meta"]["eval_seed_base"] = eval_seed_base
            # 디버그 수치(평균/표준편차)는 메타로만 보존
            if debug_eval_mean is not None:
                out["meta"]["eval_return_mean"] = float(debug_eval_mean)
            if debug_eval_std is not None:
                out["meta"]["eval_return_std"] = float(debug_eval_std)
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

        # 파일 기록
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
            # 액션-레이어 편향 메타도 중앙 로그에 얹기
            **_bias_meta_from_args(args),
        }

        # (A) 중앙 로그 파일 갱신
        try:
            write_metrics_csv(metrics_csv, args, out, meta=meta)
        except Exception:
            pass

        # (B) 기존 자동 CSV 경로 유지
        try:
            setattr(cfg, "method", "rl")
            setattr(cfg, "es_mode", es_mode)
            save_metrics_autocsv(out.get("metrics", {}), cfg, outputs=args.outputs)
        except Exception:
            pass

        # (C) outputs/<tag>/metrics.json & metrics.csv 강제 생성
        try:
            tag = getattr(args, "tag", None)
            if tag:
                _save_metrics_files(Path(args.outputs) / tag, metrics_dict, args)
        except Exception:
            pass

        if _onoff(getattr(args, "autosave", "off")) == "on":
            do_autosave(out.get("metrics") or {}, cfg, args, out)

    return out
