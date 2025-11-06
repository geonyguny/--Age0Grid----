# project/env/reward.py
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False


# ─────────────────────────────────────────────────────────
# Backend helpers (NumPy / Torch 공통 처리)
# ─────────────────────────────────────────────────────────
def _is_torch(x):
    return _HAS_TORCH and hasattr(x, "device")

def _xp_and_ops(x):
    """입력 x의 타입에 맞춰 (xp, relu, as_tensor) 제공."""
    if _is_torch(x):
        xp = torch
        relu = torch.relu
        def as_tensor(v, dtype=None, device=None):
            return torch.as_tensor(v, dtype=dtype or x.dtype, device=device or x.device)
    else:
        xp = np
        relu = lambda z: np.maximum(z, 0.0)
        def as_tensor(v, dtype=None, device=None):  # noqa: ARG001
            return float(v) if np.isscalar(v) else np.asarray(v)
    return xp, relu, as_tensor

def _clip_pos(xp, x, eps=1e-12):
    # numpy / torch 공용 clip(min) 구현
    return xp.clip(x, eps, None) if xp is np else (x.clamp(min=eps))


# ─────────────────────────────────────────────────────────
# Bias config helpers (behavioral_bias 우선, flat 키 폴백)
# ─────────────────────────────────────────────────────────
def _get_bias_flags(cfg):
    """
    behavioral_bias 네임스페이스를 우선 사용하고,
    없으면 기존 flat 키(bias_on, bias_loss_aversion, prob_gamma)를 폴백으로 사용.
    u_scale은 cfg.u_scale을 사용(없으면 1.0).
    """
    # behavioral_bias 우선
    bias_ns = getattr(cfg, "behavioral_bias", None)

    def _get(ns, name, flat_name, default):
        if ns is not None and hasattr(ns, name):
            return getattr(ns, name)
        return getattr(cfg, flat_name, default)

    bias_on_raw = _get(bias_ns, "bias_on", "bias_on", "off")
    bias_on = str(bias_on_raw).lower() == "on"

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
# CRRA utility and marginal utility
# ─────────────────────────────────────────────────────────
def _u_crra(x, gamma, xp):
    x = _clip_pos(xp, x)
    if abs(gamma - 1.0) < 1e-12:
        return xp.log(x) if xp is np else torch.log(x)
    return (x ** (1.0 - gamma) - 1.0) / (1.0 - gamma)

def _u_prime_crra(c, gamma, xp):
    c = _clip_pos(xp, c)
    if abs(gamma - 1.0) < 1e-12:
        # du/dc = 1/c  (log 유틸리티)
        return 1.0 / c if xp is np else (1.0 / c)
    return c ** (-gamma)


# ─────────────────────────────────────────────────────────
# Terminal utility loss (CVaR 용 손실) + κ(손실회피) 반영
# ─────────────────────────────────────────────────────────
def terminal_loss_utility(W_T, F_target, cfg):
    """
    L_u (terminal utility loss)을 계산한다.

    모드:
      • cfg.cvar_unit == "utility":
          L_u = [ u(F_target) - u(W_T) ]_+
      • 그 외(통화단위 → 효용 환산):
          L_u = u'(c̄) * [F_target - W_T]_+

    손실회피(κ) 가중:
      • behavioral_bias.bias_on == "on" 이고 behavioral_bias.loss_aversion = κ > 0 이면
          L_u ← κ * L_u    (κ=1.0 이면 baseline, 1.5/2.0이면 손실 가중 강화)
      • κ ≤ 0 또는 bias_on != "on" 이면 가중 없음

    스케일:
      • cfg.u_scale(기본 1.0)을 최종 곱.

    반환 타입은 입력 W_T의 타입을 따른다 (np.ndarray 또는 torch.Tensor).
    """
    xp, relu, as_tensor = _xp_and_ops(W_T)

    # 파라미터 준비
    gamma = float(getattr(cfg, "crra_gamma", 3.0) or 3.0)
    # bias 플래그/스케일 (behavioral_bias 우선)
    bias_on, kappa, _prob_gamma, u_scale = _get_bias_flags(cfg)

    # 목표 F
    if xp is np:
        F = float(F_target)
    else:
        F = as_tensor(F_target)

    # L_u 기본 계산
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
            cstar_m = float(getattr(cfg, "cstar_m", 0.04 / 12.0) or (0.04 / 12.0))
            c_bar = cstar_m * F
        else:
            # 고정 상수값 사용
            c_bar_val = getattr(cfg, "uprime_cbar_value", 1.0)
            c_bar = as_tensor(c_bar_val, dtype=None, device=None)

        upr = _u_prime_crra(c_bar, gamma, xp)
        L_u = upr * L_cur

    # κ(손실회피) 가중치 적용: bias_on==on && κ>0 일 때만 적용
    if bias_on and (kappa > 0.0):
        if xp is np:
            L_u = kappa * L_u
        else:
            L_u = (as_tensor(kappa, dtype=W_T.dtype, device=W_T.device)) * L_u

    # 보상 스케일 적용 (기본 1.0)
    if xp is np:
        return u_scale * L_u
    return (as_tensor(u_scale, dtype=W_T.dtype, device=W_T.device)) * L_u
