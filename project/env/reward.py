# project/env/reward.py
"""
Reward / utility helpers for retirement / IRP environments.

- CRRA 효용 u(c)를 기반으로:
  * step_utility_reward: 한 시점 소비(consumption)에 대한 효용을 계산 (RL용 per-step reward).
  * terminal_loss_utility: CVaR 계열 목적함수를 위한 terminal utility-based loss.

구현 목표
---------
1) RL의 episode return을 "효용 합(Σ u(c_t))"으로 정의할 수 있도록
   step_utility_reward(consumption, cfg)을 제공한다.
2) terminal_loss_utility는 utility 단위(cvar_unit='utility') 또는
   화폐단위(cvar_unit!='utility') 손실을 CRRA 효용에 맞게 변환한다.
3) 행동편향 관련 스케일링(u_scale, loss_aversion κ)은 behavioral_bias 네임스페이스를
   우선 사용하고, 없는 경우 flat config 키를 폴백으로 사용한다.
"""

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    _HAS_TORCH = False


# ─────────────────────────────────────────────────────────
# Backend helpers (NumPy / Torch 공통 처리)
# ─────────────────────────────────────────────────────────
def _is_torch(x) -> bool:
    return _HAS_TORCH and hasattr(x, "device")


def _xp_and_ops(x):
    """
    입력 x의 타입에 맞춰 (xp, relu, as_tensor) 제공.
    - xp: np 또는 torch
    - relu: 동일 backend의 ReLU
    - as_tensor: 입력을 x와 동일 backend/dtype/device로 변환
    """
    if _is_torch(x):
        xp = torch
        relu = torch.relu

        def as_tensor(v, dtype=None, device=None):
            return torch.as_tensor(
                v,
                dtype=dtype or getattr(x, "dtype", torch.float32),
                device=device or getattr(x, "device", None),
            )
    else:
        xp = np

        def relu(z):
            return np.maximum(z, 0.0)

        def as_tensor(v, dtype=None, device=None):  # noqa: ARG001
            if np.isscalar(v):
                return float(v)
            arr = np.asarray(v)
            if dtype is not None:
                return arr.astype(dtype)
            return arr

    return xp, relu, as_tensor


def _clip_pos(xp, x, eps: float = 1e-12):
    """
    numpy / torch 공용의 "양수 클리핑" 구현.
    - 최소 eps 이상으로 올려서 log/거듭제곱에서의 수치불안을 방지.
    """
    if xp is np:
        return xp.clip(x, eps, None)
    # torch
    return x.clamp(min=eps)


# ─────────────────────────────────────────────────────────
# Bias / scale config helpers
# ─────────────────────────────────────────────────────────
def _get_bias_flags(cfg):
    """
    behavioral_bias 네임스페이스를 우선 사용하고,
    없으면 기존 flat 키(bias_on, bias_loss_aversion, prob_gamma)를 폴백으로 사용.
    u_scale은 cfg.u_scale을 사용(없으면 1.0).

    반환:
        bias_on: bool, 손실회피 등 행동편향 on/off
        kappa:  float, loss_aversion(손실 가중 κ)
        prob_gamma: float, 확률 왜곡 계수(여기서는 사용 X, 호환성 유지용)
        u_scale: float, 효용 스케일 계수 (보상 크기 조정용)
    """
    bias_ns = getattr(cfg, "behavioral_bias", None)

    def _get(ns, name, flat_name, default):
        if ns is not None and hasattr(ns, name):
            return getattr(ns, name)
        return getattr(cfg, flat_name, default)

    bias_on_raw = _get(bias_ns, "bias_on", "bias_on", "off")
    bias_on = str(bias_on_raw).strip().lower() == "on"

    kappa_raw = _get(bias_ns, "loss_aversion", "bias_loss_aversion", 0.0)
    try:
        kappa = float(kappa_raw or 0.0)
    except Exception:
        kappa = 0.0

    pg_raw = _get(bias_ns, "prob_gamma", "prob_gamma", 1.0)
    try:
        prob_gamma = float(pg_raw or 1.0)
    except Exception:
        prob_gamma = 1.0

    try:
        u_scale = float(getattr(cfg, "u_scale", 1.0) or 1.0)
    except Exception:
        u_scale = 1.0

    return bias_on, kappa, prob_gamma, u_scale


# ─────────────────────────────────────────────────────────
# CRRA utility & marginal utility
# ─────────────────────────────────────────────────────────
def _u_crra(x, gamma: float, xp):
    """
    CRRA 효용 u(x; γ) = (x^{1-γ} - 1)/(1-γ),  γ ≠ 1
                      = log x               ,  γ = 1
    - x는 양수로 클리핑되어 들어온다고 가정.
    """
    x = _clip_pos(xp, x)
    if abs(gamma - 1.0) < 1e-12:
        # log utility
        return xp.log(x) if xp is not np else np.log(x)
    return (x ** (1.0 - gamma) - 1.0) / (1.0 - gamma)


def _u_prime_crra(c, gamma: float, xp):
    """
    CRRA 효용의 한계효용 u'(c).
    - γ ≠ 1: c^{-γ}
    - γ = 1: 1 / c
    """
    c = _clip_pos(xp, c)
    if abs(gamma - 1.0) < 1e-12:
        return 1.0 / c
    return c ** (-gamma)


# ─────────────────────────────────────────────────────────
# Per-step utility reward (RL episode return = Σ u(c_t))
# ─────────────────────────────────────────────────────────
def step_utility_reward(consumption, cfg, allow_negative: bool = False):
    """
    한 기간 소비(consumption)에 대한 CRRA 효용을 계산하여 반환한다.
    RL이 maximize하는 episode return을 "소비 효용 합"으로 만들고자 할 때
    환경에서 per-step reward로 이 함수를 호출하면 된다.

    인자
    ----
    consumption: float 또는 배열 (np.ndarray, torch.Tensor)
        한 시점의 소비 금액 C_t.
    cfg:
        환경 / 시뮬레이션 설정 객체 (crra_gamma, u_scale 등 포함).
    allow_negative: bool, default False
        True  이면 음수 소비도 허용하되, |c|로 효용을 계산한다.
        False 이면 consumption < 0 은 0으로 클립한다. (무소비로 간주)

    반환
    ----
    동일 backend(np / torch)의 효용 값 u(C_t) * u_scale.
    """
    xp, _relu, as_tensor = _xp_and_ops(consumption)

    # gamma, u_scale 설정
    gamma = float(getattr(cfg, "crra_gamma", 3.0) or 3.0)
    _, _kappa, _prob_gamma, u_scale = _get_bias_flags(cfg)

    # 소비값 전처리
    if xp is np:
        c = np.asarray(consumption, dtype=float)
    else:
        c = as_tensor(consumption)

    if not allow_negative:
        # 음수 소비는 0으로 클립: "소비 없음"으로 해석
        if xp is np:
            c = np.maximum(c, 0.0)
        else:
            c = c.clamp(min=0.0)
    else:
        # 음수 허용 시 절대값 기준 효용 (선호 구조 단순화)
        if xp is np:
            c = np.abs(c)
        else:
            c = c.abs()

    u_c = _u_crra(c, gamma, xp)

    # 최종 스케일링
    if xp is np:
        return u_scale * u_c
    return as_tensor(u_scale, dtype=c.dtype, device=c.device) * u_c


# ─────────────────────────────────────────────────────────
# Terminal utility loss (CVaR 용 손실) + κ(손실회피) 반영
# ─────────────────────────────────────────────────────────
def terminal_loss_utility(W_T, F_target, cfg):
    """
    L_u (terminal utility loss)을 계산한다.

    목적
    ----
    CVaR 계열 목적함수에서 "terminal wealth" 기준 손실을
    효용 스케일로 변환하여 사용하기 위한 함수.

    모드
    ----
    • cfg.cvar_unit == "utility":
        L_u = [ u(F_target) - u(W_T) ]_+
          - W_T가 F_target 보다 효용 관점에서 얼마나 부족한지의 양의 부분.
    • 그 외(통화단위 → 효용 환산):
        L_u = u'(c̄) * [F_target - W_T]_+
          - 통화단위 손실을 기준 한계효용(u'(c̄))으로 선형 근사하여 효용 스케일로 변환.

    손실회피(κ) 가중
    ----------------
    • behavioral_bias.bias_on == "on" 이고 behavioral_bias.loss_aversion = κ > 0 이면
        L_u ← κ * L_u    (κ=1.0 이면 baseline, κ>1 이면 손실 가중 강화)
    • κ ≤ 0 또는 bias_on != "on" 이면 가중 없음.

    스케일
    ------
    • cfg.u_scale(기본 1.0)을 최종 곱하여, episode reward와의 스케일을 통일한다.

    반환
    ----
    입력 W_T와 동일 backend(np.ndarray / torch.Tensor)의 L_u.
    """
    xp, relu, as_tensor = _xp_and_ops(W_T)

    # 파라미터 준비
    gamma = float(getattr(cfg, "crra_gamma", 3.0) or 3.0)
    bias_on, kappa, _prob_gamma, u_scale = _get_bias_flags(cfg)

    # 목표 F (backend 일치)
    if xp is np:
        F = float(F_target)
    else:
        F = as_tensor(F_target)

    # 기본 유닛 모드 (utility vs currency)
    unit_mode = str(getattr(cfg, "cvar_unit", "utility")).lower()

    if unit_mode == "utility":
        # 유틸리티 레벨 차의 양의 부분
        bF = _u_crra(F,   gamma, xp)
        bW = _u_crra(W_T, gamma, xp)
        L_u = relu(bF - bW)
    else:
        # 통화단위 손실을 한계효용으로 환산
        L_cur = relu(F - W_T)
        uprime_mode = str(getattr(cfg, "uprime_cbar_mode", "annuity")).lower()

        if uprime_mode == "annuity":
            # 월 정액 비율로 c̄ = cstar_m * F (기본 4%/년 = 0.04/12/월)
            cstar_m = float(
                getattr(cfg, "cstar_m", 0.04 / 12.0) or (0.04 / 12.0)
            )
            c_bar = cstar_m * F
        else:
            # 고정 상수값 사용
            c_bar_val = getattr(cfg, "uprime_cbar_value", 1.0)
            c_bar = as_tensor(c_bar_val)

        upr = _u_prime_crra(c_bar, gamma, xp)
        L_u = upr * L_cur

    # κ(손실회피) 가중치 적용: bias_on==on && κ>0 일 때만 적용
    if bias_on and (kappa > 0.0):
        if xp is np:
            L_u = kappa * L_u
        else:
            L_u = as_tensor(kappa, dtype=W_T.dtype, device=W_T.device) * L_u

    # 최종 효용 스케일 적용 (기본 1.0)
    if xp is np:
        return u_scale * L_u
    return as_tensor(u_scale, dtype=W_T.dtype, device=W_T.device) * L_u
