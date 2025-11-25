# project/trainer/rl_a2c.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time, csv, math, os, random, contextlib
from pathlib import Path
from typing import Any, Dict, Callable, Optional, Tuple, List

import numpy as np

# Keep import (compat)
from project.metrics.es import es95_wealth

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta

# env import: env/__init__.py 가 RetirementEnv를 re-export 하지 않는 경우엔
# `from project.env.retirement_env import RetirementEnv` 로 바꿔도 됨.
from project.env import RetirementEnv

# (옵션) 효용기반 종단손실 함수가 있으면 사용, 없으면 무시
try:
    from project.env.reward import terminal_loss_utility  # noqa: F401
    _HAS_UTILITY_LOSS = True
except Exception:
    _HAS_UTILITY_LOSS = False

try:
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
except Exception:
    pass


# ─────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────
def _to_f(x, default: float = 0.0) -> float:
    """안전 부동소수 변환."""
    try:
        if x is None:
            return float(default)
        xf = float(x)
        if not np.isfinite(xf):
            return float(default)
        return xf
    except Exception:
        return float(default)

def _sanitize_tensor(x: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32)
    mask = ~torch.isfinite(x)
    if mask.any():
        x = x.clone()
        x[mask] = float(fill)
    return x


# ─────────────────────────────────────────────────────────
# Gym-like shim for our Env
# ─────────────────────────────────────────────────────────
class _GymShim:
    """
    RetirementEnv를 관측 벡터 기반의 경량 Gym 인터페이스로 감싸는 어댑터.
    관측: [t_norm, W/W0, age_norm, pad] (pad는 0)
    """
    def __init__(self, base_env: RetirementEnv):
        self.base = base_env

    def _obs_vec(self, s):
        if isinstance(s, dict):
            t  = _to_f(s.get("t", s.get("age", 0.0)), 0.0)
            W  = _to_f(s.get("W", s.get("W_t", 0.0)), 0.0)
            T  = max(getattr(self.base, "T", 1), 1)
            W0 = _to_f(getattr(self.base, "W0", 1.0), 1.0) or 1.0
            age = _to_f(s.get("age", getattr(self.base, "age_years", 55.0)), 55.0)
            age_norm = age / 120.0
            return np.array([t/float(T-1 if T > 1 else 1), W/max(W0, 1e-12), age_norm, 0.0], dtype=np.float32)
        arr = np.asarray(s, dtype=np.float32).ravel()
        if arr.size >= 2:
            if arr.size < 4:
                pad = np.zeros((4-arr.size,), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=0)
        else:
            arr = np.array([0.0, 1.0, 65.0/120.0, 0.0], dtype=np.float32)
        return arr.astype(np.float32, copy=False)

    def reset(self, seed=None):
        try:
            out = self.base.reset(seed=seed)
        except TypeError:
            out = self.base.reset()
        s = out[0] if (isinstance(out, tuple) and len(out) >= 1) else out
        return self._obs_vec(s), {}

    def step(self, action):
        q = float(action[0]); w = float(action[1])
        out = self.base.step(q, w)
        if not isinstance(out, tuple):
            raise RuntimeError("RetirementEnv.step must return a tuple")
        if len(out) == 5:
            s_next, r, done, trunc, info = out
        elif len(out) == 4:
            s_next, r, done, info = out; trunc = False
        else:
            raise RuntimeError(f"Unexpected step() return length: {len(out)}")

        r = _to_f(r, 0.0)
        obs = self._obs_vec(s_next)
        # W_T가 info에 없다면 best-effort로 보완
        if isinstance(info, dict) and "W_T" not in info:
            try:
                wt = _to_f(getattr(self.base, "W", None) or (s_next.get("W") if isinstance(s_next, dict) else None), 0.0)
                info["W_T"] = wt
            except Exception:
                pass
        return obs, float(r), bool(done), bool(trunc), info

    def close(self):
        if hasattr(self.base, "close"):
            with contextlib.suppress(Exception):
                self.base.close()


# ─────────────────────────────────────────────────────────
# Utility (CRRA)
# ─────────────────────────────────────────────────────────
def u_crra(c: float, gamma: float = 3.0, eps: float = 1e-8) -> float:
    c = max(float(c), 0.0) + eps
    if abs(gamma - 1.0) < 1e-9:
        return math.log(c)
    return (c ** (1.0 - gamma) - 1.0) / (1.0 - gamma)


# ─────────────────────────────────────────────────────────
# Policy network
# ─────────────────────────────────────────────────────────
class BetaHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc_a = nn.Linear(in_dim, 1)
        self.fc_b = nn.Linear(in_dim, 1)
        nn.init.xavier_uniform_(self.fc_a.weight); nn.init.zeros_(self.fc_a.bias)
        nn.init.xavier_uniform_(self.fc_b.weight); nn.init.zeros_(self.fc_b.bias)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = F.softplus(self.fc_a(h)) + 1.001
        beta  = F.softplus(self.fc_b(h)) + 1.001
        return alpha, beta


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.head_q = BetaHead(hidden)
        self.head_w = BetaHead(hidden)
        self.v_head  = nn.Linear(hidden, 1)
        nn.init.xavier_uniform_(self.v_head.weight); nn.init.zeros_(self.v_head.bias)

    def forward(self, obs: torch.Tensor):
        h = self.backbone(obs)
        a_q, b_q = self.head_q(h)
        a_w, b_w = self.head_w(h)
        v = self.v_head(h).squeeze(-1)
        return (a_q, b_q), (a_w, b_w), v

    def act(self, obs: torch.Tensor, cfg: Any, eval_mode: bool = False):
        (a_q, b_q), (a_w, b_w), v = self.forward(obs)
        dist_q = Beta(a_q, b_q); dist_w = Beta(a_w, b_w)
        if eval_mode:
            raw_q = (a_q / (a_q + b_q)).clamp(1e-4, 1-1e-4)
            raw_w = (a_w / (a_w + b_w)).clamp(1e-4, 1-1e-4)
        else:
            raw_q = dist_q.rsample().clamp(1e-4, 1-1e-4)
            raw_w = dist_w.rsample().clamp(1e-4, 1-1e-4)

        q_floor = float(getattr(cfg, "q_floor", 0.0) or 0.0)
        w_max   = float(getattr(cfg, "w_max", 1.0) or 1.0)
        q_cap   = float(getattr(cfg, "rl_q_cap", 0.0) or 0.0)

        q = q_floor + (1.0 - q_floor) * raw_q
        w = w_max * raw_w

        if q_cap > 0.0:
            q = torch.minimum(q, torch.as_tensor(q_cap, dtype=q.dtype, device=q.device))

        logp = dist_q.log_prob(raw_q) + dist_w.log_prob(raw_w)
        ent  = dist_q.entropy().squeeze(-1) + dist_w.entropy().squeeze(-1)
        return q.squeeze(-1), w.squeeze(-1), logp.squeeze(-1), ent, v, (dist_q, dist_w)


# ─────────────────────────────────────────────────────────
# GAE (safe)
# ─────────────────────────────────────────────────────────
def compute_gae(rews: torch.Tensor,
                vals: torch.Tensor,
                dones: torch.Tensor,
                gamma: float,
                lam: float):
    if rews.ndim > 1: rews = rews.squeeze(-1)
    if vals.ndim > 1: vals = vals.squeeze(-1)
    if dones.ndim > 1: dones = dones.squeeze(-1)

    T = int(rews.shape[0])
    device = rews.device
    dtype  = rews.dtype

    vals_ext = torch.cat([vals, torch.zeros(1, device=device, dtype=dtype)], dim=0)

    if dones.dtype == torch.bool:
        dones_f = dones.float()
    else:
        dones_f = dones.to(dtype)

    adv = torch.zeros(T, device=device, dtype=dtype)
    lastgaelam = torch.zeros((), device=device, dtype=dtype)

    for t in reversed(range(T)):
        nonterminal = 1.0 - dones_f[t]
        delta = rews[t] + gamma * vals_ext[t + 1] * nonterminal - vals_ext[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam

    ret = adv + vals
    return adv, ret


# ─────────────────────────────────────────────────────────
# Dual CVaR (terminal / stage-wise)
# ─────────────────────────────────────────────────────────
class DualCVaR:
    def __init__(self, alpha: float = 0.95, eta_init: float = 0.0, tau: float = 0.3):
        self.alpha = alpha; self.eta = eta_init; self.tau = tau
    def update_eta_with_batch(self, L_batch: List[float] | np.ndarray):
        if L_batch is None or len(L_batch) == 0:
            return
        q_alpha = np.quantile(np.asarray(L_batch, dtype=float), self.alpha)
        self.eta = (1.0 - self.tau) * self.eta + self.tau * float(q_alpha)
    def terminal_penalty(self, L: float) -> float:
        return self.eta + max(L - self.eta, 0.0) / max(1e-12, (1.0 - self.alpha))


class DualCVaRStage:
    def __init__(self, alpha: float = 0.95, eta_init: float = 0.0, tau: float = 0.3):
        self.alpha = alpha; self.eta = eta_init; self.tau = tau
    def update_eta_with_batch(self, Ls: List[float] | np.ndarray):
        if Ls is None or len(Ls) == 0:
            return
        q_alpha = np.quantile(np.asarray(Ls, dtype=float), self.alpha)
        self.eta = (1.0 - self.tau) * self.eta + self.tau * float(q_alpha)
    def penalty(self, L: float) -> float:
        return self.eta + max(L - self.eta, 0.0) / max(1e-12, (1.0 - self.alpha))


# ─────────────────────────────────────────────────────────
# Teacher policy (warm start)
# ─────────────────────────────────────────────────────────
def teacher_action(cfg):
    # 4% rule monthly
    spm = int(getattr(cfg, "steps_per_year", 12) or 12)
    q4_m = 1.0 - (1.0 - 0.04) ** (1.0 / spm)
    q_cap = float(getattr(cfg, "rl_q_cap", 0.0) or 0.0)
    q_floor = float(getattr(cfg, "q_floor", 0.0) or 0.0)
    w_teacher = min(0.60, float(getattr(cfg, "w_max", 1.0) or 1.0))
    q_teacher = q4_m
    if q_cap > 0.0: q_teacher = min(q_teacher, q_cap)
    q_teacher = max(q_teacher, q_floor)
    return float(q_teacher), float(w_teacher)


# ─────────────────────────────────────────────────────────
# Helper: c_star (reference consumption) for shaping / loss aversion
# ─────────────────────────────────────────────────────────
def _ref_consumption_cstar(cfg, env_shim: _GymShim, obs_vec: np.ndarray, mode: str = "annuity") -> float:
    """
    기준소비 c* 계산:
      - annuity : c* = p_m * (W/W0)
      - fixed   : c* = cstar_m * (W/W0)
      - vpw     : 간단 근사(VPW) 기반
    """
    try:
        W_over_W0 = float(obs_vec[1])
    except Exception:
        W_over_W0 = 1.0

    mode = str(mode or "annuity").lower()
    if mode == "fixed":
        c_star_m = float(getattr(cfg, "cstar_m", 0.04/12) or 0.04/12)
        return c_star_m * W_over_W0

    if mode == "vpw":
        try:
            Nm = int(getattr(env_shim.base, "T", 1)) - int(getattr(env_shim.base, "t", 0))
        except Exception:
            Nm = 1
        g_m = 0.0
        try:
            monthly = getattr(cfg, "monthly", None)
            if isinstance(monthly, dict):
                g_m = float(monthly.get("g_m", 0.0) or 0.0)
        except Exception:
            g_m = 0.0
        a = (1.0 - (1.0+g_m)**(-Nm))/g_m if g_m > 0 else max(Nm, 1)
        q_m = min(1.0, 1.0 / a)
        return q_m * W_over_W0

    # annuity (default)
    try:
        monthly = getattr(cfg, "monthly", None)
        if isinstance(monthly, dict):
            p_m = float(monthly.get("p_m", 0.04/12) or 0.04/12)
        else:
            p_m = float(getattr(cfg, "cstar_m", 0.04/12) or 0.04/12)
    except Exception:
        p_m = 0.04/12
    return p_m * W_over_W0


# ─────────────────────────────────────────────────────────
# F_target 추정 (CLI 미지정 시)
# ─────────────────────────────────────────────────────────
def _infer_F_target(cfg) -> float:
    """F_target을 안전하게 추정. 엔진 스케일(초기 W0=1) 가정.
    우선순위: CLI값 > annuity-PV 근사(c* * years)
    """
    F_cli = getattr(cfg, "F_target", None)
    try:
        if F_cli is not None and float(F_cli) > 0.0:
            return float(F_cli)
    except Exception:
        pass

    try:
        cstar_m = float(getattr(cfg, "cstar_m", 0.04) or 0.04)
    except Exception:
        cstar_m = 0.04

    if cstar_m <= 0.2:  # 연간 4~10% 등으로 간주
        years = int(getattr(cfg, "horizon_years", 35) or 35)
        return max(1e-6, float(cstar_m) * float(years))
    else:
        return max(1e-6, float(cstar_m))


# ─────────────────────────────────────────────────────────
# Rollout
# ─────────────────────────────────────────────────────────
def rollout(env, policy, cfg, steps, gamma, lam, device,
            cvar_hook, lw_scale, survive_bonus, teacher_eps,
            stage_cvar: Optional[DualCVaRStage] = None, cstar_mode: str = "annuity"):
    """
    1 에포크 roll-out + GAE 계산.
    ★ 손실회피 진단치: la_sf_mean (c* 대비 소비 부족률의 평균)을 함께 반환.
    """
    obs_list, act_q, act_w, logp_list, ent_list, val_list, rew_list, done_list = ([] for _ in range(8))
    stage_L_list: List[float] = []  # for stage-wise CVaR eta update

    # ★ 손실회피 진단치(평균 부족률)
    la_sf_sum = 0.0
    la_sf_cnt = 0

    obs, _ = env.reset(seed=None)

    q_cap = float(getattr(cfg, "rl_q_cap", 0.0) or 0.0)
    q_floor = float(getattr(cfg, "q_floor", 0.0) or 0.0)
    w_max   = float(getattr(cfg, "w_max", 1.0) or 1.0)
    gamma_crra = float(getattr(cfg, "crra_gamma", 3.0) or 3.0)
    u_scale    = float(getattr(cfg, "u_scale", 0.0) or 0.0)

    # κ(손실회피) 강도
    kappa = float(getattr(cfg, "bias_loss_aversion", 0.0) or 0.0)
    bias_on = str(getattr(cfg, "bias_on", "off")).lower() == "on"
    la_active = bias_on and (kappa > 0.0)

    for _ in range(steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        q, w, logp, ent, v, (dist_q, dist_w) = policy.act(obs_t, cfg, eval_mode=False)

        # teacher forcing
        if random.random() < max(0.0, min(1.0, float(teacher_eps))):
            tq, tw = teacher_action(cfg)
            if q_cap > 0.0: tq = min(tq, q_cap)
            tq = max(tq, q_floor); tw = min(max(tw, 0.0), w_max)
            denom_q = max(1e-8, 1.0 - q_floor)
            raw_q_t = np.clip((tq - q_floor) / denom_q, 1e-4, 1-1e-4)
            raw_w_t = np.clip(tw / max(w_max, 1e-8), 1e-4, 1-1e-4)
            raw_q_t = torch.tensor([raw_q_t], dtype=torch.float32, device=device)
            raw_w_t = torch.tensor([raw_w_t], dtype=torch.float32, device=device)
            logp = (dist_q.log_prob(raw_q_t) + dist_w.log_prob(raw_w_t)).squeeze(-1)
            q = torch.tensor([tq], dtype=torch.float32, device=device)
            w = torch.tensor([tw], dtype=torch.float32, device=device)

        if q_cap > 0.0:
            q = torch.clamp(q, max=q_cap)

        obs2, c_t, done, trunc, info = env.step(np.array([float(q.item()), float(w.item())], dtype=np.float32))

        # reward shaping ───────────────────────────────────
        rew = 0.0

        # (1) 기본 효용(선택): CRRA
        if u_scale != 0.0:
            rew += u_scale * u_crra(_to_f(c_t, 0.0), gamma=gamma_crra)

        # (2) Stage-wise CVaR (소비 단기부족) + κ 반영
        if stage_cvar is not None:
            stage_on = (str(getattr(cfg, "cvar_stage", "off")).lower() == "on") or bool(getattr(cfg, "cvar_stage_on", False))
            if stage_on:
                c_star = _ref_consumption_cstar(cfg, env, obs, mode=cstar_mode)
                L_t = max(c_star - _to_f(c_t, 0.0), 0.0)
                stage_L_list.append(L_t)
                lam_s = float(getattr(cfg, "lambda_stage", 0.0) or 0.0)
                if lam_s > 0.0:
                    lam_eff = lam_s * (max(1.0, kappa) if la_active else 1.0)
                    rew -= lam_eff * stage_cvar.penalty(L_t)

        # (3) ★ 손실회피 κ: c_t < c* 일 때 부족률 페널티 + 진단치 집계
        #     (la_active가 아니더라도 진단치는 집계하여 la_sf_mean이 항상 정의되게)
        c_star_la = _ref_consumption_cstar(cfg, env, obs, mode=cstar_mode)
        if c_star_la > 0.0:
            shortfall = max(c_star_la - _to_f(c_t, 0.0), 0.0)
            shortfall_ratio = shortfall / max(c_star_la, 1e-12)
            if la_active:
                rew -= kappa * shortfall_ratio
            la_sf_sum += float(shortfall_ratio)
            la_sf_cnt += 1

        # (4) 생존 보너스
        if not (done or trunc) and survive_bonus != 0.0:
            rew += float(survive_bonus)

        # (5) 에피소드 종료 시 종단 CVaR 및 W_T shaping
        if (done or trunc):
            if cvar_hook is not None:
                rew += float(cvar_hook(info))
            lw = float(getattr(cfg, "lw_scale", 0.0) or 0.0)
            if lw != 0.0:
                rew += lw * _to_f(info.get("W_T", 0.0), 0.0)
        # ─────────────────────────────────────────────────

        # keep graph
        obs_list.append(obs_t.squeeze(0))
        act_q.append(q.detach()); act_w.append(w.detach())
        logp_list.append(logp)
        ent_list.append(ent)
        val_list.append(v)
        rew_list.append(torch.tensor(_to_f(rew, 0.0), dtype=torch.float32, device=device))
        done_list.append(bool(done or trunc))
        obs = obs2
        if done or trunc:
            obs, _ = env.reset(seed=None)

    # stack
    obs_t = torch.stack(obs_list)
    logp  = torch.stack(logp_list)
    ent   = torch.stack(ent_list)
    val   = torch.stack(val_list)
    rews  = torch.stack(rew_list)
    dones = torch.tensor(done_list, device=rews.device, dtype=torch.float32)

    # final sanitization before GAE
    rews = _sanitize_tensor(rews, 0.0).float()
    val  = _sanitize_tensor(val,  0.0).float()

    adv, ret = compute_gae(rews, val, dones, gamma, lam)

    # ★ 진단치: 에포크 내 평균 부족률
    la_sf_mean = (la_sf_sum / max(la_sf_cnt, 1)) if la_sf_cnt > 0 else 0.0

    return {
        "obs": obs_t,
        "q": torch.stack(act_q), "w": torch.stack(act_w),
        "logp": logp, "ent": ent, "val": val,
        "adv": adv.detach(), "ret": ret.detach(),
        "stage_L": np.asarray(stage_L_list, dtype=np.float64),
        # ★ 손실회피 평균 부족률(진단)
        "la_sf_mean": float(la_sf_mean),
    }


# ─────────────────────────────────────────────────────────
# Mean-policy evaluation (returns W_T samples)
# ─────────────────────────────────────────────────────────
def evaluate_mean_policy(make_env_fn, policy: PolicyNet, cfg, n_paths=300, device="cpu") -> Dict[str, Any]:
    Ws: List[float] = []
    with torch.no_grad():
        for _ in range(n_paths):
            env = make_env_fn()
            obs, _ = env.reset(seed=None)
            done = False; trunc = False
            while not (done or trunc):
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                q, w, *_ = policy.act(obs_t, cfg, eval_mode=True)
                obs, r, done, trunc, info = env.step(np.array([float(q.item()), float(w.item())], dtype=np.float32))
            W_T = info.get("W_T", None)
            if W_T is None and hasattr(env, "W"):
                W_T = float(env.W)
            Ws.append(float(_to_f(W_T, 0.0)))
            env.close()
    Ws = np.asarray(Ws, dtype=np.float64)
    EW = float(np.mean(Ws)) if Ws.size > 0 else 0.0

    # ES95 계산 단위 선택: wealth vs utility(있을 때)
    F_t = _infer_F_target(cfg)
    unit = str(getattr(cfg, "cvar_unit", "wealth")).lower()
    if unit == "utility" and _HAS_UTILITY_LOSS:
        L_u = terminal_loss_utility(Ws, F_t, cfg)  # κ, u_scale 내부 반영 가정
        L_u = np.asarray(L_u, dtype=float)
        k = max(1, int(0.05 * max(0, L_u.size)))
        if k > 0 and L_u.size >= k:
            tail_idx = np.argsort(L_u)[-k:]
            ES95 = float(np.mean(L_u[tail_idx]))
        else:
            ES95 = float(np.mean(L_u)) if L_u.size > 0 else 0.0
    else:
        L = np.maximum(F_t - Ws, 0.0)
        k = max(1, int(0.05 * max(0, L.size)))
        if k > 0 and L.size >= k:
            tail_idx = np.argsort(L)[-k:]
            ES95 = float(np.mean(L[tail_idx]))
        else:
            ES95 = float(np.mean(L)) if L.size > 0 else 0.0

    Ruin = float(np.mean(Ws <= 0.0)) if Ws.size > 0 else 0.0
    return {"EW": EW, "ES95": ES95, "Ruin": Ruin, "mean_WT": EW, "eval_WT": Ws.tolist()}


# ─────────────────────────────────────────────────────────
# XAI helpers (best-effort; ignore failures)
# ─────────────────────────────────────────────────────────
def make_policy_heatmaps(policy: PolicyNet, cfg, outputs, device="cpu"):
    import matplotlib.pyplot as plt
    out = Path(outputs) / "xai"; out.mkdir(parents=True, exist_ok=True)
    W_grid = np.linspace(0.1, 2.0, 81, dtype=np.float32)
    Tn = 20
    Q = np.zeros((Tn, len(W_grid)), dtype=np.float32)
    Wp = np.zeros_like(Q)
    with torch.no_grad():
        policy.eval()
        for ti in range(Tn):
            t_norm = ti / (Tn-1)
            for j, wv in enumerate(W_grid):
                obs = np.array([t_norm, wv, 65.0/120.0, 0.0], dtype=np.float32)
                q_m, w_m, *_ = policy.act(torch.tensor(obs).unsqueeze(0).to(device), cfg, eval_mode=True)
                Q[ti, j] = float(q_m.item()); Wp[ti, j] = float(w_m.item())
    for name, A in [("Pi_q_heatmap.png", Q), ("Pi_w_heatmap.png", Wp)]:
        plt.figure()
        plt.imshow(A, aspect="auto", origin="lower",
                   extent=[W_grid.min(), W_grid.max(), 0.0, 1.0])
        plt.colorbar()
        plt.xlabel("Wealth (W/W0)"); plt.ylabel("t_norm")
        plt.title(name.replace(".png",""))
        plt.tight_layout()
        plt.savefig(out / name); plt.close()

def collect_occupancy(make_env_fn, policy: PolicyNet, cfg, outputs, n_paths=300, device="cpu"):
    import matplotlib.pyplot as plt
    out = Path(outputs) / "xai"; out.mkdir(parents=True, exist_ok=True)
    hist = np.zeros((50, 50), dtype=np.int32)
    with torch.no_grad():
        for _ in range(n_paths):
            env = make_env_fn()
            obs, _ = env.reset(seed=None)
            done = False; trunc = False
            while not (done or trunc):
                t_norm = float(obs[0]); Wn = float(obs[1])
                i = min(49, max(0, int(t_norm*50)))
                j = min(49, max(0, int((Wn/2.0)*50)))  # W in [0,2]
                hist[i, j] += 1
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                q, w, *_ = policy.act(obs_t, cfg, eval_mode=True)
                obs, r, done, trunc, info = env.step(np.array([float(q.item()), float(w.item())], dtype=np.float32))
            env.close()
    plt.figure()
    plt.imshow(hist, aspect="auto", origin="lower")
    plt.colorbar(); plt.title("Occupancy (t_norm vs W/W0)")
    plt.tight_layout(); plt.savefig(out / "occupancy.png"); plt.close()

def replay_tail_paths(make_env_fn, policy: PolicyNet, cfg, outputs, k=8, device="cpu"):
    import matplotlib.pyplot as plt
    out = Path(outputs) / "xai"; out.mkdir(parents=True, exist_ok=True)
    paths: List[Tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    with torch.no_grad():
        for _ in range(200):
            env = make_env_fn()
            obs, _ = env.reset(seed=None)
            Wts, qs, ws = [], [], []
            done = False; trunc = False
            while not (done or trunc):
                Wts.append(float(obs[1]))
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                q, w, *_ = policy.act(obs_t, cfg, eval_mode=True)
                qs.append(float(q.item())); ws.append(float(w.item()))
                obs, r, done, trunc, info = env.step(np.array([float(q.item()), float(w.item())], dtype=np.float32))
            WT = _to_f(info.get("W_T", getattr(env, "W", 0.0)), 0.0)
            paths.append((WT, np.array(Wts), np.array(qs), np.array(ws)))
            env.close()
    paths.sort(key=lambda x: x[0])
    sel = paths[:max(1, int(k))]
    for idx, (WT, Wts, qs, ws) in enumerate(sel):
        t = np.arange(len(Wts))
        plt.figure(); plt.plot(t, Wts); plt.title(f"tail path {idx} WT={WT:.3f}"); plt.tight_layout()
        plt.savefig(out / f"tail_{idx}_W.png"); plt.close()
        plt.figure(); plt.plot(t, qs); plt.title(f"q(t) tail {idx}"); plt.tight_layout()
        plt.savefig(out / f"tail_{idx}_q.png"); plt.close()
        plt.figure(); plt.plot(t, ws); plt.title(f"w(t) tail {idx}"); plt.tight_layout()
        plt.savefig(out / f"tail_{idx}_w.png"); plt.close()


# ─────────────────────────────────────────────────────────
# Trainer-local CSV logger
# ─────────────────────────────────────────────────────────
def append_metrics_csv(outputs_dir: str, fields: Dict[str, Any]):
    out_logs = Path(outputs_dir) / "_logs"; out_logs.mkdir(parents=True, exist_ok=True)
    dest = out_logs / "metrics.csv"; write_header = (not dest.exists())
    # 헤더 고정성 보장을 위해 fieldnames를 명시적으로 정렬
    safe = {k: v for k, v in fields.items()
            if isinstance(v, (int, float, str, bool)) or v is None}
    fieldnames = list(safe.keys())
    with dest.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(safe)


# ─────────────────────────────────────────────────────────
# Actor adapter (policy -> (q,w))
# ─────────────────────────────────────────────────────────
def make_actor_from_policy(policy: PolicyNet, cfg: Any, device: str = "cpu") -> Callable[[Dict[str, Any]], Tuple[float, float]]:
    T_ref = int(getattr(cfg, "T", 0) or getattr(cfg, "horizon_years", 35) * getattr(cfg, "steps_per_year", 12))
    T_ref = max(T_ref, 1)
    W0 = float(getattr(cfg, "W0", 1.0) or 1.0)

    try:
        in_dim = int(policy.backbone[0].in_features)
    except Exception:
        in_dim = 4

    policy = policy.to(device)
    policy.eval()

    def _build_obs(state: Dict[str, Any]) -> np.ndarray:
        try:
            if "t_norm" in state:
                t_norm = float(state["t_norm"])
            else:
                t_val = _to_f(state.get("t", state.get("age", 0.0)), 0.0)
                t_norm = max(0.0, min(1.0, t_val / float(T_ref)))
            W = _to_f(state.get("W", state.get("W_t", 0.0)), 0.0)
            age = _to_f(state.get("age", 55.0), 55.0)
            obs4 = np.array([t_norm, W / max(W0, 1e-12), age / 120.0, 0.0], dtype=np.float32)
        except Exception:
            obs4 = np.array([0.0, 1.0, 65.0/120.0, 0.0], dtype=np.float32)

        if in_dim <= obs4.shape[0]:
            return obs4[:in_dim]
        pad = np.zeros((in_dim - obs4.shape[0],), dtype=np.float32)
        return np.concatenate([obs4, pad], axis=0)

    @torch.no_grad()
    def actor(state: Dict[str, Any]) -> Tuple[float, float]:
        obs = _build_obs(state)
        q_t, w_t, *_ = policy.act(torch.tensor(obs).unsqueeze(0).to(device), cfg, eval_mode=True)
        return float(q_t.item()), float(w_t.item())

    return actor


# ─────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────
def train_rl(cfg, seed_list, outputs, n_paths_eval=300, rl_epochs=60, steps_per_epoch=2048,
             lr=3e-4, gamma=None, gae_lambda=0.95, entropy_coef=0.01, value_coef=0.5,
             max_grad_norm=0.5, device=None) -> Dict[str, Any]:

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # robust gamma: cfg.beta가 None/비정상이어도 안전 고정
    beta_cfg = getattr(cfg, "beta", 0.996)
    try:
        gamma = float(gamma if gamma is not None else (beta_cfg if beta_cfg is not None else 0.996))
    except Exception:
        gamma = 0.996

    def _make_env_local():
        return _GymShim(RetirementEnv(cfg))

    # CVaR dual (terminal)
    cvar = DualCVaR(alpha=float(getattr(cfg, "alpha", 0.95) or 0.95), eta_init=0.0, tau=0.3)
    lambda_term = float(getattr(cfg, "lambda_term", 0.0) or 0.0)
    F_target    = _infer_F_target(cfg)

    # κ 설정 (종단 CVaR에도 반영)
    kappa = float(getattr(cfg, "bias_loss_aversion", 0.0) or 0.0)
    bias_on = str(getattr(cfg, "bias_on", "off")).lower() == "on"
    la_active = bias_on and (kappa > 0.0)

    def cvar_hook(info: Dict[str, Any]) -> float:
        if lambda_term == 0.0:
            return 0.0
        W_T = info.get("W_T", None)
        if W_T is None:
            return 0.0
        unit = str(getattr(cfg, "cvar_unit", "wealth")).lower()
        if unit == "utility" and _HAS_UTILITY_LOSS:
            L_val = float(np.asarray(terminal_loss_utility(np.array([float(W_T)], dtype=float), F_target, cfg), dtype=float)[0])
            lambda_eff = float(lambda_term)  # 효용 모드에선 내부에서 κ/u_scale 반영 가정
        else:
            L_val = max(F_target - float(W_T), 0.0)
            # ★ 핵심: wealth 단위 CVaR에는 κ를 가중치로 반영
            lambda_eff = float(lambda_term) * (max(1.0, kappa) if la_active else 1.0)
        return - lambda_eff * cvar.terminal_penalty(L_val)

    # Stage-wise CVaR (consumption)
    stage_flag = (str(getattr(cfg, "cvar_stage", "off")).lower() == "on") or bool(getattr(cfg, "cvar_stage_on", False))
    stage_cvar = DualCVaRStage(alpha=float(getattr(cfg, "alpha_stage", 0.95) or 0.95), eta_init=0.0, tau=0.3) if stage_flag else None
    cstar_mode = str(getattr(cfg, "cstar_mode", "annuity") or "annuity").lower()

    # shaping params
    lw_scale       = float(getattr(cfg, "lw_scale", 0.0) or 0.0)
    survive_bonus  = float(getattr(cfg, "survive_bonus", 0.0) or 0.0)
    teacher_eps0   = float(getattr(cfg, "teacher_eps0", 0.0) or 0.0)
    teacher_decay  = float(getattr(cfg, "teacher_decay", 1.0) or 1.0)

    # build once to get obs_dim
    env0 = _make_env_local(); obs0, _ = env0.reset(seed=None)
    obs_dim = int(np.asarray(obs0, dtype=np.float32).shape[0]); env0.close()

    policy = PolicyNet(obs_dim).to(device)
    optim_all = optim.Adam(policy.parameters(), lr=lr)

    train_t0 = time.perf_counter()
    best_epoch = None
    last_la_sf_mean = 0.0  # 마지막 에포크 진단치 (CSV/반환용)

    for epoch in range(int(rl_epochs)):
        eps = teacher_eps0 * (teacher_decay ** epoch)
        batch = rollout(
            _make_env_local(), policy, cfg,
            int(steps_per_epoch), float(gamma), float(gae_lambda),
            device, cvar_hook, lw_scale, survive_bonus, eps,
            stage_cvar=stage_cvar, cstar_mode=cstar_mode
        )
        # 최신 진단치 보관
        last_la_sf_mean = float(batch.get("la_sf_mean", 0.0) or 0.0)

        # normalize adv
        adv = (batch["adv"] - batch["adv"].mean()) / (batch["adv"].std() + 1e-8)
        pi_loss = -(batch["logp"] * adv).mean()
        v_loss  = 0.5 * ((batch["val"] - batch["ret"])**2).mean()
        ent_b   = batch["ent"].mean()
        loss = pi_loss + float(value_coef) * v_loss - float(entropy_coef) * ent_b

        optim_all.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), float(max_grad_norm))
        optim_all.step()

        # update eta using small mean-policy rollouts (terminal CVaR)
        Ws_tmp: List[float] = []
        for _ in range(8):
            env = _make_env_local()
            obs, _ = env.reset(seed=None)
            done = False; trunc = False
            with torch.no_grad():
                while not (done or trunc):
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    q, w, *_ = policy.act(obs_t, cfg, eval_mode=True)
                    obs, r, done, trunc, info = env.step(np.array([float(q.item()), float(w.item())], dtype=np.float32))
            W_T = info.get("W_T", None)
            if W_T is None and hasattr(env, "W"):
                W_T = float(env.W)
            Ws_tmp.append(float(_to_f(W_T, 0.0))); env.close()
        if lambda_term > 0.0 and len(Ws_tmp) > 0:
            unit = str(getattr(cfg, "cvar_unit", "wealth")).lower()
            if unit == "utility" and _HAS_UTILITY_LOSS:
                Ls_u = terminal_loss_utility(np.asarray(Ws_tmp, dtype=float), F_target, cfg)
                Ls_u = np.asarray(Ls_u, dtype=float)
                cvar.update_eta_with_batch(Ls_u)
            else:
                Ls = np.maximum(F_target - np.asarray(Ws_tmp, dtype=np.float64), 0.0)
                cvar.update_eta_with_batch(Ls)

        # stage-wise CVaR eta update
        if stage_cvar is not None:
            stage_cvar.update_eta_with_batch(batch.get("stage_L", []))

    train_t1 = time.perf_counter()

    # evaluation (mean-policy)
    eval_t0 = time.perf_counter()
    metrics_mp = evaluate_mean_policy(_make_env_local, policy, cfg, n_paths=int(n_paths_eval), device=device)
    eval_t1 = time.perf_counter()

    # ---- save checkpoint with minimal hints for later evaluation ----
    ckpt_path = None
    try:
        tag = getattr(cfg, "tag", "rl_run") or "rl_run"
        out_dir = os.path.join(outputs, tag)
        os.makedirs(out_dir, exist_ok=True)
        ckpt_path = os.path.join(out_dir, "policy.pt")

        # arch inference
        try:
            first = policy.backbone[0]
            obs_dim_save = int(first.in_features)
            hidden_save = int(first.out_features)
        except Exception:
            obs_dim_save = int(locals().get("obs_dim", 4))
            hidden_save = 128

        # collect minimal env hints
        T_hint, W0_hint = 0, 1.0
        steps_per_year_hint = int(getattr(cfg, "steps_per_year", 12) or 12)
        horizon_years_hint  = int(getattr(cfg, "horizon_years", 35) or 35)

        _tmp_env = _make_env_local()
        try:
            _tmp_env.reset(seed=None)
            T_hint  = int(getattr(_tmp_env.base, "T", getattr(cfg, "T", 0)) or 0)
            W0_hint = float(getattr(_tmp_env.base, "W0", getattr(cfg, "W0", 1.0)) or 1.0)
        finally:
            with contextlib.suppress(Exception):
                _tmp_env.close()

        ckpt = {
            "state_dict": policy.state_dict(),
            "obs_dim": obs_dim_save,
            "arch": {"obs_dim": obs_dim_save, "hidden": hidden_save},
            "cfg_hints": {
                "q_floor": float(getattr(cfg, "q_floor", 0.0) or 0.0),
                "w_max":   float(getattr(cfg, "w_max", 1.0) or 1.0),
                "rl_q_cap":float(getattr(cfg, "rl_q_cap", 0.0) or 0.0),
                "T": int(T_hint),
                "W0": float(W0_hint),
                "steps_per_year": steps_per_year_hint,
                "horizon_years":  horizon_years_hint,
            },
        }
        torch.save(ckpt, ckpt_path)
    except Exception:
        ckpt_path = None

    # XAI (optional)
    if bool(getattr(cfg, "xai_on", True)):
        with contextlib.suppress(Exception):
            make_policy_heatmaps(policy, cfg, outputs, device=device)
            collect_occupancy(_make_env_local, policy, cfg, outputs, n_paths=200, device=device)
            replay_tail_paths(_make_env_local, policy, cfg, outputs, k=6, device=device)

    # trainer-local CSV (best-effort)
    ts = time.strftime("%y%m%dT%H%M%S")

    # CSV 필드 구성(필요한 진단/파라미터를 넓게 남긴다)
    cstar_mode_csv = str(getattr(cfg, "cstar_mode", "annuity") or "annuity")
    try:
        cstar_m_csv = float(getattr(cfg, "cstar_m", 0.04/12) or 0.04/12)
    except Exception:
        cstar_m_csv = 0.04/12
    rl_q_cap_csv = float(getattr(cfg, "rl_q_cap", 0.0) or 0.0)

    fields_csv = dict(
        ts=ts, asset=getattr(cfg, "asset", "US"), method="rl", baseline="",
        es_mode=str(getattr(cfg, "cvar_unit", "wealth")).lower(),  # wealth / utility
        F_target=F_target, w_max=getattr(cfg, "w_max", 1.0),
        hedge_on=getattr(cfg, "hedge_on", False), hedge_mode=getattr(cfg, "hedge_mode", ""),
        hedge_sigma_k=getattr(cfg, "hedge_sigma_k", 0.0), lambda_term=lambda_term,
        fee_annual=getattr(cfg, "phi_adval", getattr(cfg, "fee_annual", 0.0)),
        floor_on=getattr(cfg, "floor_on", False),
        f_min_real=getattr(cfg, "f_min_real", 0.0),
        EW=metrics_mp.get("EW"), ES95=metrics_mp.get("ES95"),
        Ruin=metrics_mp.get("Ruin"), mean_WT=metrics_mp.get("mean_WT"),
        seeds=" ".join(map(str, seed_list)) if seed_list else "",
        n_paths_eval=int(n_paths_eval), outputs=str(outputs),
        mortality_on=getattr(cfg, "mortality_on", False),
        market_mode=getattr(cfg, "market_mode", "iid"),
        cvar_stage_on=stage_flag,
        bias_on=str(getattr(cfg, "bias_on", "off")),
        bias_loss_aversion=getattr(cfg, "bias_loss_aversion", 0.0),
        # ★ 핵심 진단치
        la_sf_mean=float(last_la_sf_mean),
        # ★ 참고 파라미터 추가 기록
        cstar_mode=cstar_mode_csv,
        cstar_m=cstar_m_csv,
        rl_q_cap=rl_q_cap_csv,
    )
    append_metrics_csv(outputs, fields_csv)

    actor = make_actor_from_policy(policy, cfg, device=device)

    return {
        "EW": metrics_mp.get("EW"),
        "ES95": metrics_mp.get("ES95"),
        "Ruin": metrics_mp.get("Ruin"),
        "mean_WT": metrics_mp.get("mean_WT"),
        "best_epoch": best_epoch,
        "train_time_s": float(train_t1 - train_t0),
        "eval_time_s": float(eval_t1 - eval_t0),
        "actor": actor,
        "ckpt_path": ckpt_path,
        "eval_WT": metrics_mp.get("eval_WT"),
        # ★ 반환에도 포함 (runner/cli가 metrics로 승격할 수 있게)
        "la_sf_mean": float(last_la_sf_mean),
    }
