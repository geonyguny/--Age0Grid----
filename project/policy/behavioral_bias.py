# project/policy/behavioral_bias.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple, Callable, Optional
import numpy as np

@dataclass
class BiasConfig:
    on: bool = False
    loss_aversion: float = 0.0   # λ≥0; 최근 손실 신호가 있을 때 w 축소 강도
    prob_gamma: float = 1.0      # γ∈(0,1]이면 테일 과대평가 → w 보수화
    myopia: float = 0.0          # 0~1; 클수록 q 상향(현재소비 편향)
    w_floor: float = 0.0         # risky 최소 비중 하한
    w_cap_shock: float = 0.0     # 단기 변동성 신호 기반 추가 축소 강도

def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

def apply_bias(q: float, w: float, state: Any, cfg: BiasConfig) -> Tuple[float, float]:
    if not cfg.on:
        return float(q), float(w)

    q_b, w_b = float(q), float(w)

    # 1) 손실회피: 최근 수익 신호가 음수면 w 축소
    r = 0.0
    if isinstance(state, dict):
        r = float(state.get("recent_ret", 0.0))
    if cfg.loss_aversion > 0 and r < 0:
        w_b *= (1.0 - cfg.loss_aversion * min(1.0, abs(r)))

    # 2) 확률왜곡: γ<1 → w 보수화
    if cfg.prob_gamma < 1.0:
        k = 1.0 - cfg.prob_gamma  # 0~1
        w_b *= (1.0 - 0.25 * k)

    # 3) 근시: q 상향
    if cfg.myopia > 0:
        q_b *= (1.0 + 0.1 * cfg.myopia)

    # 4) 변동성 기반 추가 캡
    vol = 0.0
    if isinstance(state, dict):
        vol = float(state.get("recent_vol", 0.0))
    if cfg.w_cap_shock > 0 and vol > 0:
        w_b *= max(0.0, 1.0 - cfg.w_cap_shock * min(vol, 1.0))

    # 5) 클립/바닥
    w_b = max(cfg.w_floor, _clip01(w_b))
    q_b = _clip01(q_b)
    return q_b, w_b

def _extract_env_signal(env) -> dict:
    """
    env가 주어지면 최근 수익/변동성 신호를 만들어서 반환.
    없거나 접근 불가면 0으로 둠.
    """
    recent_ret = 0.0
    recent_vol = 0.0
    try:
        t = int(getattr(env, "t", 0))
        # path_risky가 있고 t>0이면 바로 이전 수익을 사용
        pr = getattr(env, "path_risky", None)
        if pr is not None and isinstance(pr, (list, tuple, np.ndarray)) and t > 0:
            pr_arr = np.asarray(pr, dtype=float)
            if 0 < t <= pr_arr.size:
                recent_ret = float(pr_arr[t-1])
                # 최근 12개월 표준편차 근사(있으면)
                w = pr_arr[max(0, t-12):t]
                if w.size >= 2 and np.isfinite(w).all():
                    recent_vol = float(np.std(w))
    except Exception:
        pass
    return {"recent_ret": recent_ret, "recent_vol": recent_vol}

def make_bias_wrapper(args, env: Optional[Any] = None) -> Callable[[Callable[[Any], Tuple[float, float]]], Callable[[Any], Tuple[float, float]]]:
    """
    actors.build_actor에서 wrapper = make_bias_wrapper(args, env) 식으로 호출.
    반환값은 actor(q,w)를 후처리하는 래퍼.
    """
    try:
        on = str(getattr(args, "bias_on", "off")).lower() == "on"
        cfg = BiasConfig(
            on=on,
            loss_aversion=float(getattr(args, "bias_loss_aversion", 0.0) or 0.0),
            prob_gamma=float(getattr(args, "bias_prob_gamma", 1.0) or 1.0),
            myopia=float(getattr(args, "bias_myopia", 0.0) or 0.0),
            w_floor=float(getattr(args, "bias_w_floor", 0.0) or 0.0),
            w_cap_shock=float(getattr(args, "bias_w_cap_shock", 0.0) or 0.0),
        )
    except Exception:
        # 파라미터 파싱 실패 시 no-op
        cfg = BiasConfig(on=False)

 # ▼ 추가: 켜졌을 때만 1회 알림
    if cfg.on:
        print(f"[BIAS] on={cfg.on} loss_aversion={cfg.loss_aversion} prob_gamma={cfg.prob_gamma} myopia={cfg.myopia}")

    def wrapper(actor):
        if not cfg.on:
            return actor

        def acted(obs):
            q, w = actor(obs)
            state = _extract_env_signal(env) if env is not None else {}
            q_b, w_b = apply_bias(q, w, state, cfg)

            # ▼ 추가: 초기 2스텝 정도만 q/w 변환 확인(시끄러우면 나중에 삭제)
            try:
                t = int(getattr(env, "t", 0))
            except Exception:
                t = -1
            if 0 <= t < 2:
                print(f"[BIAS-APPLY] t={t} q:{q:.4f}->{q_b:.4f} w:{w:.4f}->{w_b:.4f} state={state}")

            return q_b, w_b
        return acted

    return wrapper
