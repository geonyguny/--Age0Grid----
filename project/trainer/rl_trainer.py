# project/trainer/rl_trainer.py
# -*- coding: utf-8 -*-
"""
A2C + GAE trainer for IRP decumulation with two Beta-headed actions (q_t, w_t) ∈ [0,1]^2.
(중략: 상단 주석 동일)
"""
from __future__ import annotations
import os
import math
import time
import json
import csv
import random
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Any, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Beta
except Exception as e:  # pragma: no cover
    raise ImportError("PyTorch is required for RL training. Please install torch.")

# ─────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────
@dataclass
class RLConfig:
    # Model / Env
    obs_dim: int
    hidden_dims: List[int] = None
    gamma: float = 0.996
    lam: float = 0.95  # for GAE
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Optim
    lr: float = 3e-4
    weight_decay: float = 0.0

    # Training loop
    max_steps: int = 200_000
    rollout_len: int = 512
    batch_size: int = 2048  # effective batch across rollouts
    seed: int = 42

    # Logging / IO
    log_dir: str = r".\\outputs\\_logs"
    tag: str = "rl_a2c_gae"
    ckpt_every: int = 10_000
    save_every: int = 10_000
    verbose: bool = True

    # Misc
    device: str = "auto"  # "cpu" | "cuda" | "auto"
    entropy_clip: float = 0.0  # if >0, clamp entropy bonus minimum
    value_clip: float = 0.0     # if >0, clip target - value for stability

    # Hooks (future)
    teacher_kl_coef: float = 0.0  # if >0, add KL(π||π_teacher)

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 128]

# ─────────────────────────────────────────────────────────
# Networks
# ─────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int, act=nn.Tanh):
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i+1]), act()]
        layers += [nn.Linear(dims[-1], out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class BetaActor(nn.Module):
    """Outputs α, β for each action head (q, w). Ensures positivity via softplus + 1."""
    def __init__(self, obs_dim: int, hidden: List[int]):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden, out_dim=4)  # [a_q, b_q, a_w, b_w]
        self.softplus = nn.Softplus()
    def forward(self, x) -> Tuple[Beta, Beta, torch.Tensor]:
        raw = self.backbone(x)
        a_q, b_q, a_w, b_w = torch.chunk(raw, 4, dim=-1)
        a_q = self.softplus(a_q) + 1.0
        b_q = self.softplus(b_q) + 1.0
        a_w = self.softplus(a_w) + 1.0
        b_w = self.softplus(b_w) + 1.0
        dist_q = Beta(a_q, b_q)
        dist_w = Beta(a_w, b_w)
        return dist_q, dist_w, raw

class ValueCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: List[int]):
        super().__init__()
        self.v = MLP(obs_dim, hidden, out_dim=1)
    def forward(self, x):
        return self.v(x).squeeze(-1)

# ─────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

class RolloutBuffer:
    def __init__(self, obs_dim: int, capacity: int):
        self.capacity = capacity
        self.reset(obs_dim)
    def reset(self, obs_dim: int):
        self.obs = []
        self.actions_q = []
        self.actions_w = []
        self.logp_sum = []   # logp(q)+logp(w)
        self.rews = []
        self.dones = []
        self.vals = []
        self.infos = []
        self.last_obs = None  # ← bootstrap value용 마지막 관측치
    def add(self, obs, a_q, a_w, logp_sum, rew, done, val, info):
        self.obs.append(obs)
        self.actions_q.append(a_q)
        self.actions_w.append(a_w)
        self.logp_sum.append(logp_sum)
        self.rews.append(rew)
        self.dones.append(done)
        self.vals.append(val)
        self.infos.append(info)
    def to_tensors(self, device: torch.device):
        obs = torch.as_tensor(np.asarray(self.obs), dtype=torch.float32, device=device)
        a_q = torch.as_tensor(np.asarray(self.actions_q), dtype=torch.float32, device=device)
        a_w = torch.as_tensor(np.asarray(self.actions_w), dtype=torch.float32, device=device)
        logp = torch.as_tensor(np.asarray(self.logp_sum), dtype=torch.float32, device=device)
        rew = torch.as_tensor(np.asarray(self.rews), dtype=torch.float32, device=device)
        done = torch.as_tensor(np.asarray(self.dones), dtype=torch.float32, device=device)
        val = torch.as_tensor(np.asarray(self.vals), dtype=torch.float32, device=device)
        return obs, a_q, a_w, logp, rew, done, val

# ─────────────────────────────────────────────────────────
# GAE (expects val_pad length T+1)
# ─────────────────────────────────────────────────────────
@torch.no_grad()
def compute_gae(rew: torch.Tensor, val_pad: torch.Tensor, done: torch.Tensor,
                gamma: float, lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    rew: [T]
    val_pad: [T+1]  # v_0..v_T (부트스트랩)
    done: [T]       # 1.0 if terminal at t, else 0.0
    """
    T = rew.shape[0]
    adv = torch.zeros(T, device=rew.device)
    gae = torch.zeros((), device=rew.device)

    for t in reversed(range(T)):
        nonterminal = 1.0 - done[t]
        delta = rew[t] + gamma * val_pad[t+1] * nonterminal - val_pad[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae

    ret = adv + val_pad[:-1]   # [T]
    return adv, ret

# ─────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────
class RLTrainer:
    def __init__(self, cfg: RLConfig, env_factory: Callable[[], Any]):
        self.cfg = cfg
        set_global_seed(cfg.seed)
        self.device = to_device(cfg.device)
        self.env = env_factory()
        # infer obs_dim if needed
        if cfg.obs_dim <= 0:
            o = self.env.reset(seed=cfg.seed)
            cfg.obs_dim = int(np.asarray(o, dtype=np.float32).shape[-1])
        # nets
        self.actor = BetaActor(cfg.obs_dim, cfg.hidden_dims).to(self.device)
        self.critic = ValueCritic(cfg.obs_dim, cfg.hidden_dims).to(self.device)
        self.opt = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        # IO
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(cfg.log_dir, f"rl_{cfg.tag}_{ts}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.log_path = os.path.join(self.run_dir, "train_log.csv")
        self.ckpt_path = os.path.join(self.run_dir, "ckpt.pt")
        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step","loss_pi","loss_v","ent","adv_mean","ret_mean","rew_mean","ew_proxy"])

    def _act(self, obs_t: torch.Tensor, deterministic: bool=False) -> Tuple[np.ndarray, float, float, Dict[str,float]]:
        dist_q, dist_w, raw = self.actor(obs_t)
        if deterministic:
            a_q = (dist_q.concentration1 / (dist_q.concentration1 + dist_q.concentration0)).squeeze(-1)
            a_w = (dist_w.concentration1 / (dist_w.concentration1 + dist_w.concentration0)).squeeze(-1)
            logp_sum = torch.zeros_like(a_q)
        else:
            a_q = dist_q.rsample().squeeze(-1)
            a_w = dist_w.rsample().squeeze(-1)
            logp_sum = dist_q.log_prob(a_q) + dist_w.log_prob(a_w)
        ent = dist_q.entropy().mean() + dist_w.entropy().mean()
        return a_q.detach().cpu().numpy(), a_w.detach().cpu().numpy(), logp_sum.detach().cpu().numpy(), {"entropy": float(ent.item())}

    def _value(self, obs_t: torch.Tensor) -> torch.Tensor:
        return self.critic(obs_t)

    def _rollout(self, start_obs: np.ndarray, steps: int) -> Tuple[RolloutBuffer, np.ndarray]:
        buf = RolloutBuffer(self.cfg.obs_dim, steps)
        obs = start_obs
        for _ in range(steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            a_q, a_w, logp_arr, ent_info = self._act(obs_t, deterministic=False)
            act = {"q": float(a_q[0]), "w": float(a_w[0])}
            next_obs, rew, done, info = self.env.step(act)
            v = self._value(obs_t).item()
            # 확실한 스칼라 변환(DeprecationWarning 회피)
            logp_scalar = float(torch.as_tensor(logp_arr).squeeze().item())
            buf.add(obs, act["q"], act["w"], logp_scalar, float(rew), float(done), float(v), info)
            obs = self.env.reset(seed=None) if done else next_obs
            buf.last_obs = obs  # ← 다음 상태 저장(bootstrap V에 사용)
        return buf, obs

    def _update(self, buf: RolloutBuffer, step0: int) -> Tuple[float,float,float,float,float,float]:
        obs, a_q, a_w, logp, rew, done, val = buf.to_tensors(self.device)

        # 마지막 상태의 bootstrap value 계산
        assert buf.last_obs is not None, "buf.last_obs must be set during rollout"
        last_obs_t = torch.as_tensor(buf.last_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            last_v = self.critic(last_obs_t).squeeze(0)  # scalar tensor

        val_pad = torch.cat([val, last_v.reshape(1)], dim=0)  # [T+1]
        adv, ret = compute_gae(rew, val_pad, done, self.cfg.gamma, self.cfg.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # One big batch A2C update
        self.opt.zero_grad()
        dist_q, dist_w, raw = self.actor(obs)
        logp_new = dist_q.log_prob(a_q) + dist_w.log_prob(a_w)
        entropy = (dist_q.entropy() + dist_w.entropy()).mean()
        ratio = torch.exp(logp_new - logp)
        loss_pi = -(ratio * adv).mean() - self.cfg.ent_coef * entropy
        v_pred = self.critic(obs)
        v_target = ret
        if self.cfg.value_clip > 0:
            v_pred_clipped = val + (v_pred - val).clamp(-self.cfg.value_clip, self.cfg.value_clip)
            loss_v = 0.5 * torch.max((v_pred - v_target)**2, (v_pred_clipped - v_target)**2).mean()
        else:
            loss_v = 0.5 * (v_pred - v_target).pow(2).mean()
        loss = loss_pi + self.cfg.vf_coef * loss_v
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), self.cfg.max_grad_norm)
        self.opt.step()

        rew_mean = float(rew.mean().item())
        adv_mean = float(adv.mean().item())
        ret_mean = float(ret.mean().item())
        ent_val = float(entropy.item())
        return float(loss_pi.item()), float(loss_v.item()), ent_val, adv_mean, ret_mean, rew_mean

    def _log(self, step, loss_pi, loss_v, ent, adv_mean, ret_mean, rew_mean):
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([step, loss_pi, loss_v, ent, adv_mean, ret_mean, rew_mean, rew_mean])

    def _save(self):
        torch.save({
            "cfg": asdict(self.cfg),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "opt": self.opt.state_dict(),
        }, self.ckpt_path)

    def train(self):
        cfg = self.cfg
        obs = self.env.reset(seed=cfg.seed)
        step = 0
        if cfg.verbose:
            print(f"[RL] start training tag={cfg.tag} device={self.device} log_dir={self.run_dir}")
        while step < cfg.max_steps:
            buf, obs = self._rollout(obs, steps=cfg.rollout_len)
            loss_pi, loss_v, ent, adv_mean, ret_mean, rew_mean = self._update(buf, step)
            step += cfg.rollout_len
            self._log(step, loss_pi, loss_v, ent, adv_mean, ret_mean, rew_mean)
            if cfg.verbose and step % (cfg.rollout_len * 5) == 0:
                print(f"[RL] step={step} loss_pi={loss_pi:.4f} loss_v={loss_v:.4f} ent={ent:.3f} rew_mean={rew_mean:.6f}")
            if (cfg.ckpt_every > 0) and (step % cfg.ckpt_every == 0):
                self._save()
        if cfg.save_every >= 0:
            self._save()
        if cfg.verbose:
            print(f"[RL] done. logs={self.log_path}")

    @torch.no_grad()
    def evaluate_mean_policy(self, n_episodes: int = 32) -> Dict[str, float]:
        cfg = self.cfg
        device = self.device
        self.actor.eval(); self.critic.eval()
        ep_returns = []
        ep_lengths = []
        for _ in range(n_episodes):
            obs = self.env.reset(seed=None)
            done = False
            ret_sum = 0.0
            t = 0
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                a_q, a_w, _, _ = self._act(obs_t, deterministic=True)
                next_obs, rew, done, info = self.env.step({"q": float(a_q[0]), "w": float(a_w[0])})
                ret_sum += float(rew)
                t += 1
                obs = next_obs
            ep_returns.append(ret_sum)
            ep_lengths.append(t)
        return {
            "eval_return_mean": float(np.mean(ep_returns)),
            "eval_return_std": float(np.std(ep_returns)),
            "eval_len_mean": float(np.mean(ep_lengths)),
            "episodes": int(n_episodes),
        }

    def load(self, path: str | None = None):
        p = path or self.ckpt_path
        state = torch.load(p, map_location=self.device)
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.opt.load_state_dict(state["opt"])
        if "cfg" in state:
            s = state["cfg"]
            for k,v in s.items():
                setattr(self.cfg, k, v)
        return self

# ─────────────────────────────────────────────────────────
# CLI Entrypoint (optional): python -m project.trainer.rl_trainer --help
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--obs_dim", type=int, default=-1)
    parser.add_argument("--max_steps", type=int, default=50_000)
    parser.add_argument("--rollout_len", type=int, default=512)
    parser.add_argument("--tag", type=str, default="rl_a2c_gae")
    parser.add_argument("--log_dir", type=str, default=r".\\outputs\\_logs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Dummy env for smoke
    class _ToyEnv:
        def __init__(self, obs_dim=8):
            self.obs_dim = obs_dim
            self.t = 0
            self.T = 128
            self.rng = np.random.default_rng(0)
        def reset(self, seed=None):
            if seed is not None:
                self.rng = np.random.default_rng(seed)
            self.t = 0
            return self.rng.normal(size=self.obs_dim).astype(np.float32)
        def step(self, act: Dict[str,float]):
            q, w = act["q"], act["w"]
            r = -((q-0.4)**2 + (w-0.6)**2) + 0.1
            self.t += 1
            done = (self.t >= self.T)
            obs = self.rng.normal(size=self.obs_dim).astype(np.float32)
            info = {"toy": 1}
            return obs, float(r), bool(done), info

    def env_factory():
        return _ToyEnv(obs_dim=8 if args.obs_dim <= 0 else args.obs_dim)

    cfg = RLConfig(obs_dim=args.obs_dim, max_steps=args.max_steps,
                   rollout_len=args.rollout_len, tag=args.tag,
                   log_dir=args.log_dir, seed=args.seed, device=args.device)
    tr = RLTrainer(cfg, env_factory)
    tr.train()
    print(tr.evaluate_mean_policy(n_episodes=8))
