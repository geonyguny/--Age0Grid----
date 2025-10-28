# project/policy/behavioral_bias.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple, Callable, Optional, Dict
import numpy as np


# ────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────
@dataclass
class BiasConfig:
    """
    액션-레이어 행동편향 파라미터.
    - on:            편향 적용 여부
    - loss_aversion: λ≥0; 최근 손실 신호(recent_ret<0) 있을 때 위험자산 비중 w 축소 강도
    - prob_gamma:    γ∈(0,1]일수록 테일 과대평가 → w 보수화(0.75*(1-γ) 축소)
    - myopia:        0~1; 클수록 소비 q 상향(현재소비 편향)
    - w_floor:       위험자산 최소 비중 하한(0~1)
    - w_cap_shock:   최근 변동성(recent_vol) 기반 추가 축소 강도(0~1)
    """
    on: bool = False
    loss_aversion: float = 0.0
    prob_gamma: float = 1.0
    myopia: float = 0.0
    w_floor: float = 0.0
    w_cap_shock: float = 0.0

    def __post_init__(self):
        # 안전 범위로 정규화
        self.loss_aversion = _clip01f(self.loss_aversion)  # 강도계수는 0~1 범위로 제한
        self.prob_gamma = float(np.clip(self.prob_gamma, 1e-6, 1.0))
        self.myopia = _clip01f(self.myopia)
        self.w_floor = _clip01f(self.w_floor)
        self.w_cap_shock = _clip01f(self.w_cap_shock)


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────
def _clip01f(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _as_bool_on(v: Any, default: bool = False) -> bool:
    s = str(v).strip().lower()
    if s in ("on", "true", "1", "y", "yes"):
        return True
    if s in ("off", "false", "0", "n", "no"):
        return False
    return bool(default)


def _verbose_from_args(args: Any) -> bool:
    # args.quiet가 'on'이면 출력 억제
    q = getattr(args, "quiet", "on")
    return (str(q).strip().lower() not in ("on", "true", "1", "y", "yes"))


# ────────────────────────────────────────────────────────────────────
# Core bias application
# ────────────────────────────────────────────────────────────────────
def apply_bias(q: float, w: float, state: Dict[str, float], cfg: BiasConfig) -> Tuple[float, float]:
    """
    (q,w) → (q_b,w_b) 편향 보정.
    state: {"recent_ret": float, "recent_vol": float}
    """
    if not cfg.on:
        return float(_clip01f(q)), float(max(cfg.w_floor, _clip01f(w)))

    # 입력 정규화
    q_b = _clip01f(_safe_float(q))
    w_b = _clip01f(_safe_float(w))

    recent_ret = _safe_float(state.get("recent_ret", 0.0))
    recent_vol = _safe_float(state.get("recent_vol", 0.0))
    recent_vol = max(0.0, recent_vol)  # 음수는 무시

    # 1) 손실회피 λ: 최근 수익 < 0 이면 강도 비례 축소
    if cfg.loss_aversion > 0.0 and recent_ret < 0.0:
        # |r|을 1로 클램프 → 과도한 축소 방지
        shrink = cfg.loss_aversion * min(1.0, abs(recent_ret))
        w_b *= max(0.0, 1.0 - shrink)

    # 2) 확률왜곡 γ: γ<1 → 위험 비중 보수화 (선형 축소)
    if cfg.prob_gamma < 1.0:
        k = 1.0 - cfg.prob_gamma  # 0~1
        w_b *= (1.0 - 0.25 * k)   # 최대 25% 축소

    # 3) 근시 myopia: 소비성향 q 상향
    if cfg.myopia > 0.0:
        q_b *= (1.0 + 0.1 * cfg.myopia)

    # 4) 변동성 쇼크 캡: 최근 변동성의 크기에 비례해 w 축소
    if cfg.w_cap_shock > 0.0 and recent_vol > 0.0:
        # vol을 [0,1] 범위로 가정(사전 표준화되어 있지 않다면 env에서 스케일 관리)
        v = min(1.0, recent_vol)
        w_b *= max(0.0, 1.0 - cfg.w_cap_shock * v)

    # 5) 최종 클립 및 바닥
    w_b = max(cfg.w_floor, _clip01f(w_b))
    q_b = _clip01f(q_b)

    return q_b, w_b


# ────────────────────────────────────────────────────────────────────
# Env signal extraction
# ────────────────────────────────────────────────────────────────────
def _extract_env_signal(env: Any, vol_window: int = 12) -> Dict[str, float]:
    """
    env에서 최근 수익/변동성 신호를 추출.
    - 우선순위: env.recent_ret / env.recent_vol → path_risky와 t를 이용해 계산
    - 실패 시 0으로 폴백
    """
    # 1) 직접 속성 우선
    try:
        rr = getattr(env, "recent_ret", None)
        rv = getattr(env, "recent_vol", None)
        if rr is not None and rv is not None:
            return {"recent_ret": _safe_float(rr), "recent_vol": max(0.0, _safe_float(rv))}
    except Exception:
        pass

    recent_ret = 0.0
    recent_vol = 0.0
    try:
        t = int(getattr(env, "t", 0))
        pr = getattr(env, "path_risky", None)
        if pr is not None and isinstance(pr, (list, tuple, np.ndarray)) and t > 0:
            pr_arr = np.asarray(pr, dtype=float)
            if 0 < t <= pr_arr.size:
                recent_ret = float(pr_arr[t - 1])
                # 최근 vol_window 길이로 표준편차
                start = max(0, t - int(max(2, vol_window)))
                w = pr_arr[start:t]
                if w.size >= 2:
                    w = w[np.isfinite(w)]
                    if w.size >= 2:
                        recent_vol = float(np.std(w))
    except Exception:
        pass

    return {"recent_ret": recent_ret, "recent_vol": max(0.0, recent_vol)}


# ────────────────────────────────────────────────────────────────────
# Public wrapper factory
# ────────────────────────────────────────────────────────────────────
def make_bias_wrapper(
    args: Any,
    env: Optional[Any] = None,
) -> Callable[[Callable[[Any], Tuple[float, float]]], Callable[[Any], Tuple[float, float]]]:
    """
    사용법:
        wrapped = make_bias_wrapper(args, env)(actor)
        q_b, w_b = wrapped(obs)

    - args에서 bias_* 파라미터를 파싱하고, OFF면 원본 actor를 그대로 반환.
    - ON일 때만 한 번 래핑하여 단일 적용(이중 적용 방지).
    - 로그는 args.quiet='off'일 때만 간결하게 1~2스텝 알림.
    """
    # 파라미터 파싱(안전)
    try:
        cfg = BiasConfig(
            on=_as_bool_on(getattr(args, "bias_on", "off"), default=False),
            loss_aversion=_safe_float(getattr(args, "bias_loss_aversion", 0.0), 0.0),
            prob_gamma=_safe_float(getattr(args, "bias_prob_gamma", 1.0), 1.0),
            myopia=_safe_float(getattr(args, "bias_myopia", 0.0), 0.0),
            w_floor=_safe_float(getattr(args, "bias_w_floor", 0.0), 0.0),
            w_cap_shock=_safe_float(getattr(args, "bias_w_cap_shock", 0.0), 0.0),
        )
    except Exception:
        cfg = BiasConfig(on=False)

    verbose = _verbose_from_args(args)

    def _identity(actor: Callable[[Any], Tuple[float, float]]) -> Callable[[Any], Tuple[float, float]]:
        return actor

    if not cfg.on:
        return _identity

    # 상태 출력은 최초 1~2 스텝만 (quiet=off일 때)
    _logged_once = {"header": False}

    def _wrapper(actor: Callable[[Any], Tuple[float, float]]) -> Callable[[Any], Tuple[float, float]]:
        if not cfg.on:
            return actor  # 방어적

        def _acted(obs: Any) -> Tuple[float, float]:
            q, w = actor(obs)
            state = _extract_env_signal(env) if env is not None else {"recent_ret": 0.0, "recent_vol": 0.0}
            q_b, w_b = apply_bias(q, w, state, cfg)

            if verbose:
                try:
                    t = int(getattr(env, "t", -1))
                except Exception:
                    t = -1
                if not _logged_once["header"]:
                    print(f"[BIAS] on=True λ={cfg.loss_aversion} γ={cfg.prob_gamma} myopia={cfg.myopia} "
                          f"w_floor={cfg.w_floor} w_cap_shock={cfg.w_cap_shock}")
                    _logged_once["header"] = True
                if 0 <= t < 2:
                    print(f"[BIAS-APPLY] t={t} q:{_safe_float(q):.4f}->{q_b:.4f} "
                          f"w:{_safe_float(w):.4f}->{w_b:.4f} state={state}")
            return q_b, w_b

        return _acted

    return _wrapper
