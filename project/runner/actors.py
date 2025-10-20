# project/runner/actors.py
from __future__ import annotations
from typing import Any, Tuple, Callable
import numpy as _np

from ..config import SimConfig
from ..env.retirement_env import RetirementEnv  # type: ignore
from ..hjb import HJBSolver
from .helpers import arrhash, monthly_from_cfg, get_life_table_from_env
from .logging_filters import mute_logs, mute_kgr_year_logs_if

# K-GR rule
from ..policy.kgr_rule import (
    KGRLiteConfig, kgr_lite_init, kgr_lite_update_yearly, kgr_lite_policy_step,
)

# 행동편향(action-layer) 래퍼: 있으면 사용, 없으면 no-op
try:
    from ..policy.behavioral_bias import make_bias_wrapper  # type: ignore
except Exception:
    def make_bias_wrapper(_args, _env):  # type: ignore
        def _noop(actor: Callable[[Any], Tuple[float, float]]) -> Callable[[Any], Tuple[float, float]]:
            return actor
        return _noop


# --------------------------
# Adapters & Safety
# --------------------------
def _as_nd_state(raw: Any, *, T_hint: int | None = None, W_hint: float | None = None) -> _np.ndarray:
    """정책 입력을 항상 ndarray([t_norm, W])로 정규화."""
    if isinstance(raw, dict):
        T = int(T_hint or 1)
        t_idx = float(raw.get("t", 0.0))
        t_norm = t_idx / float(max(1, T - 1))
        W_now = float(raw.get("W", W_hint if W_hint is not None else 0.0))
        return _np.asarray([t_norm, W_now], dtype=float)
    if isinstance(raw, _np.ndarray):
        arr = _np.asarray(raw, dtype=float).ravel()
        if arr.size >= 2:
            return arr[:2]
        W_now = float(W_hint if W_hint is not None else 0.0)
        return _np.asarray([float(arr[0]) if arr.size else 0.0, W_now], dtype=float)
    return _np.asarray([0.0, float(W_hint if W_hint is not None else 0.0)], dtype=float)


def _clip_action(q: float, w: float, cfg: SimConfig) -> Tuple[float, float]:
    """최종 액션을 q_floor(월), w_max에 맞춰 클립."""
    q_floor_m = float(getattr(cfg, "q_floor", 0.0) or 0.0)
    w_max = float(getattr(cfg, "w_max", 1.0) or 1.0)
    q = float(_np.clip(q, q_floor_m, 1.0))
    w = float(_np.clip(w, 0.0, w_max))
    return q, w


def _w_default(cfg: SimConfig) -> float:
    """w_fixed가 주어지면 그 값을, 아니면 w_max를 사용하되 경계 안전."""
    try:
        w_max = float(getattr(cfg, "w_max", 1.0) or 1.0)
    except Exception:
        w_max = 1.0
    w_fix = getattr(cfg, "w_fixed", None)
    if w_fix is None:
        return float(_np.clip(w_max, 0.0, 1.0))
    try:
        return float(_np.clip(float(w_fix), 0.0, w_max))
    except Exception:
        return float(_np.clip(w_max, 0.0, 1.0))


# ---------- RULE ----------
def rule_actor_4pct(cfg: SimConfig, _env: RetirementEnv) -> Callable[[Any], Tuple[float, float]]:
    spm = int(getattr(cfg, "steps_per_year", 12) or 12)
    q_m = 1.0 - (1.0 - 0.04) ** (1.0 / spm)
    w_def = _w_default(cfg)
    def actor(_obs):
        return _clip_action(q_m, w_def, cfg)
    return actor


def rule_actor_cpb(cfg: SimConfig, _env: RetirementEnv) -> Callable[[Any], Tuple[float, float]]:
    _g_m, p_m = monthly_from_cfg(cfg)
    w_def = _w_default(cfg)
    def actor(_obs):
        return _clip_action(float(p_m), w_def, cfg)
    return actor


def rule_actor_vpw(cfg: SimConfig, env: RetirementEnv) -> Callable[[Any], Tuple[float, float]]:
    def _get_g_m(_cfg):
        try:
            if hasattr(_cfg, "monthly") and callable(_cfg.monthly):
                m = _cfg.monthly()
                if isinstance(m, dict) and "g_m" in m:
                    gm = float(m["g_m"])
                    if _np.isfinite(gm):
                        return gm
        except Exception:
            pass
        g_ann = float(getattr(_cfg, "g_real_annual", 0.0) or 0.0)
        spm = int(getattr(_cfg, "steps_per_year", 12) or 12)
        return (1.0 + g_ann) ** (1.0 / spm) - 1.0

    w_def = _w_default(cfg)

    def actor(_obs):
        t = int(getattr(env, "t", 0))
        T = int(getattr(env, "T", 1) or 1)
        Nm = max(T - t, 1)
        g_m = _get_g_m(cfg)
        if abs(g_m) < 1e-12:
            q_m = 1.0 / Nm
        else:
            a = (1.0 - (1.0 + g_m) ** (-Nm)) / g_m
            q_m = 1.0 / max(a, 1e-12)
        return _clip_action(q_m, w_def, cfg)
    return actor


def rule_actor_kgr(cfg: SimConfig, env: RetirementEnv, *, quiet: bool) -> Callable[[Any], Tuple[float, float]]:
    steps_per_year = int(getattr(cfg, "steps_per_year", 12) or 12)
    q_floor_ann = float(getattr(cfg, "q_floor_annual", 0.02) or 0.02)  # cfg.q_floor는 월, kgr는 연 기준 파라미터 사용
    fee_annual = float(getattr(cfg, "phi_adval", getattr(cfg, "fee_annual", 0.004)) or 0.004)
    w_fixed = _w_default(cfg)

    life_table_df = get_life_table_from_env(env)
    r_f_real_annual = float(getattr(env, "r_f_real_annual", 0.02) or 0.02)
    W0 = float(getattr(env, "W0", 1.0))
    age0 = float(getattr(cfg, "age0", 65) or 65)

    kgr_cfg = KGRLiteConfig(
        FR_high=1.30, FR_low=0.85, delta_up=0.07, delta_dn=-0.07,
        kappa_safety=0.002, w_fixed=w_fixed, q_floor=q_floor_ann,
        phi_adval_annual=fee_annual, steps_per_year=steps_per_year,
    )
    kgr_state = None
    _kgr_once_logged = False

    if life_table_df is not None:
        with mute_logs(patterns=("[kgr:year]", "[kgr:init]"), enabled=(not quiet)):
            kgr_state = kgr_lite_init(
                W0=W0, age0=age0, life_table=life_table_df,
                r_f_real_annual=r_f_real_annual, cfg=kgr_cfg,
            )

    def actor(obs):
        nonlocal kgr_state, _kgr_once_logged
        if isinstance(obs, dict):
            o = dict(obs)
        else:
            o = {
                "W_t": float(getattr(env, "W", W0)),
                "age_years": float(getattr(env, "age_years", age0)),
                "cpi_yoy": float(getattr(env, "cpi_yoy", 0.0)),
                "is_new_year": bool(getattr(env, "is_new_year", False)),
            }
        o.setdefault("life_table", life_table_df)
        o.setdefault("r_f_real_annual", r_f_real_annual)

        if kgr_state is None:
            with mute_logs(patterns=("[kgr:year]", "[kgr:init]"), enabled=(not quiet)):
                kgr_state = kgr_lite_init(
                    W0=float(o.get("W_t", W0)),
                    age0=float(o.get("age_years", age0)),
                    life_table=o.get("life_table", None),
                    r_f_real_annual=float(o.get("r_f_real_annual", r_f_real_annual)),
                    cfg=kgr_cfg,
                )

        if bool(o.get("is_new_year", False)):
            no_lt = (o.get("life_table", None) is None)
            if not _kgr_once_logged and (not quiet):
                print("[kgr:info] life_table 없음 → CPI-only 조정(연 1회) (1회만)" if no_lt
                      else "[kgr:info] life_table 기반 FR 가드레일 적용 (1회만)")
                _kgr_once_logged = True

            with mute_kgr_year_logs_if(no_life_table=no_lt if (not quiet) else False):
                kgr_lite_update_yearly(
                    W_t=float(o.get("W_t", W0)),
                    age_years=float(o.get("age_years", age0)),
                    CPI_yoy=float(o.get("cpi_yoy", 0.0)),
                    life_table=o.get("life_table", None),
                    r_f_real_annual=float(o.get("r_f_real_annual", r_f_real_annual)),
                    state=kgr_state, cfg=kgr_cfg,
                )

        with mute_kgr_year_logs_if(no_life_table=(o.get("life_table", None) is None) if (not quiet) else False):
            out = kgr_lite_policy_step(o, kgr_state, kgr_cfg)

        q_annual = float(out.get("q_t", q_floor_ann))
        q_m = 1.0 - (1.0 - q_annual) ** (1.0 / steps_per_year)
        return _clip_action(q_m, float(getattr(kgr_cfg, "w_fixed", w_fixed)), cfg)
    return actor


def build_rule_actor(cfg: SimConfig, args, env: RetirementEnv):
    if cfg.baseline == "4pct":
        return rule_actor_4pct(cfg, env)
    if cfg.baseline == "cpb":
        return rule_actor_cpb(cfg, env)
    if cfg.baseline == "vpw":
        return rule_actor_vpw(cfg, env)
    if cfg.baseline == "kgr":
        return rule_actor_kgr(cfg, env, quiet=(getattr(args, "quiet", "on") == "on"))
    raise SystemExit("--baseline required for method=rule (4pct|cpb|vpw|kgr)")


# ---------- HJB ----------
def build_hjb_actor(cfg: SimConfig, args, env: RetirementEnv):
    """HJB 정책을 테이블 보간 actor로 래핑."""
    # HJBSolver.solve() 보호 호출
    try:
        sol = HJBSolver(cfg).solve(seed=(cfg.seeds[0] if getattr(cfg, "seeds", None) else None))
    except Exception as e:
        # 완전 실패 시 상수 정책 폴백
        if str(getattr(args, "quiet", "on")).lower() != "on":
            print(f"[hjb] solver failed: {type(e).__name__} → using constant fallback")
        spm = int(getattr(cfg, "steps_per_year", 12) or 12)
        const_q = 1.0 - (1.0 - 0.04) ** (1.0 / spm)
        const_w = _w_default(cfg)
        def actor_const(_obs):
            return _clip_action(const_q, const_w, cfg)
        return actor_const

    Pi_w = sol.get("Pi_w", None)
    Pi_q = sol.get("Pi_q", None)

    # 해시 로그(quiet=off일 때만)
    if str(getattr(args, "quiet", "on")).lower() != "on":
        try:
            print("policy_hash_q=", arrhash(Pi_q))
            print("policy_hash_w=", arrhash(Pi_w))
        except Exception:
            pass

    # W grid 확보
    if "W_grid" in sol and sol["W_grid"] is not None:
        Wg = _np.asarray(sol["W_grid"], dtype=float).ravel()
    else:
        Wg = _np.linspace(float(getattr(cfg, "hjb_W_min", 0.0)),
                          float(getattr(cfg, "hjb_W_max", 1.5)),
                          int(getattr(cfg, "hjb_W_grid", 65) or 65))

    # 비정상/빈 테이블 → 상수 정책 폴백
    def _fallback_actor():
        spm = int(getattr(cfg, "steps_per_year", 12) or 12)
        const_q = 1.0 - (1.0 - 0.04) ** (1.0 / spm)
        const_w = _w_default(cfg)
        def actor(_obs):
            return _clip_action(const_q, const_w, cfg)
        return actor

    if (Pi_w is None) or (getattr(Pi_w, "size", 0) == 0) or (Pi_q is None) or (getattr(Pi_q, "size", 0) == 0):
        return _fallback_actor()

    # 정책 테이블 정화(+경계 값 채움)
    def _clean_pi(arr, lo, hi, fill):
        a = _np.asarray(arr, dtype=float)
        a = _np.nan_to_num(a, nan=fill, posinf=hi, neginf=lo)
        return _np.clip(a, lo, hi)

    q_floor_m = float(getattr(cfg, "q_floor", 0.0) or 0.0)
    spm = int(getattr(cfg, "steps_per_year", 12) or 12)
    q_fill = max(q_floor_m, 0.04 / spm)
    w_fill = min(0.6, float(getattr(cfg, "w_max", 1.0)))

    w_max = float(getattr(cfg, "w_max", 1.0) or 1.0)
    Pi_w = _clean_pi(Pi_w, 0.0, w_max, w_fill)
    Pi_q = _clean_pi(Pi_q, q_floor_m, 1.0, q_fill)

    # grid 정합성
    if Wg.size < 2 or not _np.isfinite(Wg).all():
        return _fallback_actor()

    T_pol = int(Pi_w.shape[0])

    # 단일 칼럼/행이면 보간 의미 없음 → 상수 정책
    if Pi_w.ndim != 2 or Pi_q.ndim != 2 or Pi_w.shape[1] < 2 or Pi_q.shape[1] < 2:
        return _fallback_actor()

    def _interp_row(row: _np.ndarray, xg: _np.ndarray, x: float, lo: float, hi: float, fill: float) -> float:
        """단일 시점 행(row)에서 W에 대해 선형 보간."""
        try:
            if not _np.isfinite(x):
                return fill
            return float(_np.clip(_np.interp(x, xg, row, left=row[0], right=row[-1]), lo, hi))
        except Exception:
            return float(_np.clip(fill, lo, hi))

    def actor(obs):
        s = _as_nd_state(obs, T_hint=T_pol, W_hint=float(getattr(env, "W", 1.0)))
        t_norm = float(_np.clip(s[0], 0.0, 1.0))
        W_now = float(s[1])

        t_idx = int(_np.clip(round(t_norm * (T_pol - 1)), 0, T_pol - 1))
        # 선형 보간(각 시점 행에서 W축 보간)
        q = _interp_row(Pi_q[t_idx, :], Wg, W_now, q_floor_m, 1.0, q_fill)
        w = _interp_row(Pi_w[t_idx, :], Wg, W_now, 0.0, w_max, w_fill)

        return _clip_action(q, w, cfg)

    return actor


# ---------- RL ----------
def build_rl_actor(cfg: SimConfig, _args):
    # 이 경로는 run_rl()에서 trainer를 통해 사용하는 편이 권장됨.
    try:
        from ..trainer.rl_a2c import PolicyNet, make_actor_from_policy  # noqa: F401
        raise SystemExit("RL은 runner.run_rl 경로를 사용하세요 (method=rl).")
    except Exception as e:
        raise SystemExit(f"RL route not available here: {e}")


# ---------- entry ----------
def build_actor(cfg: SimConfig, args):
    env = RetirementEnv(cfg)  # 일부 룰 정책이 참조
    if args.method == "rule":
        base_actor = build_rule_actor(cfg, args, env)
    elif args.method == "hjb":
        base_actor = build_hjb_actor(cfg, args, env)
    elif args.method == "rl":
        base_actor = build_rl_actor(cfg, args)
    else:
        raise SystemExit("Unknown method")

    # (선택) 행동편향(action-layer) 래퍼 적용
    # behavioral_bias.py가 존재하고 --bias_on on 등일 때만 효과가 있음
    wrapper = make_bias_wrapper(args, env)
    return wrapper(base_actor)
