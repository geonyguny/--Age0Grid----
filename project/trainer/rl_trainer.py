# -*- coding: utf-8 -*-
"""
A2C + GAE trainer for IRP decumulation with two Beta-headed actions (q_t, w_t) ∈ [0,1]^2.

- Actor: BetaActor → (q, w) 각각에 대한 Beta 분포 파라미터(α,β) 출력
- Critic: Value network (V(s))
- Rollout: IRPEnvAdapter/RetirementEnv 와 호환 (obs = np.float32[...])

주요 특징
---------
1) GAE(γ, λ) 기반 Advantage/Return 계산
2) 엔트로피 보너스(ent_coef), value_clip(optional) 지원
3) rollout마다 last_obs 기반 value bootstrap
4) CLI에서 toy env 로 smoke-test 가능
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Beta
except Exception as e:  # pragma: no cover
    raise ImportError("PyTorch is required for RL training. Please install torch.") from e


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
    log_dir: str = r".\outputs\_logs"
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
            layers += [nn.Linear(dims[i], dims[i + 1]), act()]
        layers += [nn.Linear(dims[-1], out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BetaActor(nn.Module):
    """
    두 개의 Beta 분포 (q, w) 헤드를 가지는 Actor.

    출력:
      - dist_q: Beta(α_q, β_q)
      - dist_w: Beta(α_w, β_w)
    """
    def __init__(self, obs_dim: int, hidden: List[int]):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden, out_dim=4)  # [a_q, b_q, a_w, b_w]
        self.softplus = nn.Softplus()

        # [FIX 2026-07] 초기화 문제: PyTorch 기본 초기화로는 마지막 레이어의 원시
        # 출력이 0 근방이라, softplus(x)+1 이후 a_q≈b_q≈1.69가 되어 초기 정책이
        # Beta(1.69,1.69)(평균 0.5, q_cap 대비 절반)에서 시작한다. 그런데 q는
        # 월간 소비율로서 실제 적정 범위(월 0.3~1.5%, 즉 q_cap의 15~75% 수준)와
        # 스케일 자체가 크게 어긋나 있어, 학습이 이 격차를 극복하는 데 오래 걸리거나
        # (critic이 부실한 상태와 겹쳐) 수렴에 실패하는 원인 중 하나로 보인다.
        # 마지막 레이어의 weight를 0으로, bias를 아래 계산된 값으로 초기화해
        # "학습 시작 시점의 정책"이 이미 합리적인 근방(q 평균≈q_cap의 15%,
        # w 평균≈0.35, 본 연구에서 확인된 근사 최적 위험자산비중과 유사)에서
        # 출발하도록 한다. 입력에 대한 의존성(가중치)은 그대로 학습되며,
        # 이는 "합리적 사전(prior)에서 시작"하는 표준적인 정책경사 초기화 기법이다.
        last = self.backbone.net[-1]
        if isinstance(last, nn.Linear):
            with torch.no_grad():
                last.weight.zero_()
                # inverse-softplus(target-1) 로 역산한 bias 값
                last.bias.copy_(torch.tensor([-0.4328, 7.4994, 0.6952, 2.8434]))

    def forward(self, x: torch.Tensor) -> Tuple[Beta, Beta, torch.Tensor]:
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
    def __init__(self, obs_dim: int, hidden: List[int], init_value: float = -1800.0):
        super().__init__()
        self.v = MLP(obs_dim, hidden, out_dim=1)
        # [FIX 2026-07] 크리틱 초기 출력이 0 근방인데, 실제 관측된 리턴(ret) 스케일은
        # 대략 -1500~-2000 수준이었다(로그 rew_mean/ret_mean 참조). 이 격차를 그래디언트
        # 하강만으로 좁히려면 학습 초반 상당 기간이 소요되고, loss_v가 학습 내내
        # 거의 줄지 않는 것처럼 보이는 현상의 원인 중 하나였다. 마지막 레이어를
        # 0으로, bias를 경험적 스케일 근방으로 초기화해 이 문제를 완화한다.
        last = self.v.net[-1]
        if isinstance(last, nn.Linear):
            with torch.no_grad():
                last.weight.zero_()
                last.bias.fill_(float(init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v(x).squeeze(-1)


# ─────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class RolloutBuffer:
    """
    단일 rollout_len 구간을 담는 버퍼.
    """
    def __init__(self, obs_dim: int, capacity: int):
        self.capacity = capacity
        self.reset(obs_dim)

    def reset(self, obs_dim: int) -> None:
        self.obs: List[np.ndarray] = []
        self.actions_q: List[float] = []
        self.actions_w: List[float] = []
        self.logp_sum: List[float] = []   # logp(q)+logp(w)
        self.rews: List[float] = []
        self.dones: List[float] = []
        self.vals: List[float] = []
        self.infos: List[Dict[str, Any]] = []
        self.last_obs: np.ndarray | None = None  # ← bootstrap value용 마지막 관측치

    def add(
        self,
        obs: np.ndarray,
        a_q: float,
        a_w: float,
        logp_sum: float,
        rew: float,
        done: float,
        val: float,
        info: Dict[str, Any],
    ) -> None:
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
def compute_gae(
    rew: torch.Tensor,
    val_pad: torch.Tensor,
    done: torch.Tensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        delta = rew[t] + gamma * val_pad[t + 1] * nonterminal - val_pad[t]
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

        # obs_dim auto infer
        if cfg.obs_dim <= 0:
            o = self.env.reset(seed=cfg.seed)
            cfg.obs_dim = int(np.asarray(o, dtype=np.float32).shape[-1])

        # nets
        self.actor = BetaActor(cfg.obs_dim, cfg.hidden_dims).to(self.device)
        self.critic = ValueCritic(cfg.obs_dim, cfg.hidden_dims).to(self.device)
        self.opt = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        # IO
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(cfg.log_dir, f"rl_{cfg.tag}_{ts}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.log_path = os.path.join(self.run_dir, "train_log.csv")
        self.ckpt_path = os.path.join(self.run_dir, "ckpt.pt")

        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                ["step", "loss_pi", "loss_v", "ent", "adv_mean", "ret_mean", "rew_mean", "ew_proxy"]
            )

    def _act(
        self,
        obs_t: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, float]]:
        dist_q, dist_w, raw = self.actor(obs_t)
        if deterministic:
            # Beta 평균 = α / (α + β)
            a_q_mean = dist_q.concentration1 / (dist_q.concentration1 + dist_q.concentration0)
            a_w_mean = dist_w.concentration1 / (dist_w.concentration1 + dist_w.concentration0)
            a_q = a_q_mean.squeeze(-1)
            a_w = a_w_mean.squeeze(-1)
            logp_sum = torch.zeros_like(a_q)
        else:
            a_q = dist_q.rsample().squeeze(-1)
            a_w = dist_w.rsample().squeeze(-1)
            logp_sum = dist_q.log_prob(a_q) + dist_w.log_prob(a_w)

        ent = dist_q.entropy().mean() + dist_w.entropy().mean()
        return (
            a_q.detach().cpu().numpy(),
            a_w.detach().cpu().numpy(),
            float(logp_sum.detach().cpu().numpy().squeeze().item()),
            {"entropy": float(ent.item())},
        )

    def _value(self, obs_t: torch.Tensor) -> torch.Tensor:
        return self.critic(obs_t)

    def _rollout(self, start_obs: np.ndarray, steps: int) -> Tuple[RolloutBuffer, np.ndarray]:
        buf = RolloutBuffer(self.cfg.obs_dim, steps)
        obs = start_obs
        for _ in range(steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            a_q, a_w, logp_scalar, ent_info = self._act(obs_t, deterministic=False)
            act = {"q": float(a_q[0]), "w": float(a_w[0])}
            next_obs, rew, done, info = self.env.step(act)
            v = self._value(obs_t).item()

            buf.add(
                obs=obs,
                a_q=act["q"],
                a_w=act["w"],
                logp_sum=float(logp_scalar),
                rew=float(rew),
                done=float(done),
                val=float(v),
                info=info,
            )

            obs = self.env.reset(seed=None) if done else next_obs
            buf.last_obs = obs  # ← 다음 상태 저장(bootstrap V에 사용)

        return buf, obs

    def _update(self, buf: RolloutBuffer, step0: int) -> Tuple[float, float, float, float, float, float]:
        obs, a_q, a_w, logp, rew, done, val = buf.to_tensors(self.device)

        # 마지막 상태의 bootstrap value 계산
        assert buf.last_obs is not None, "buf.last_obs must be set during rollout"
        last_obs_t = torch.as_tensor(
            buf.last_obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
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

        # [FIX 2026-07] entropy_clip: 학습 후반부에 엔트로피가 계속 줄어들며(탐색 감소)
        # 정책이 과도하게 확정적으로 굳어지다 붕괴(발산)하는 패턴이 반복 관측되었다.
        # 단순 clamp는 하한 아래에서 그래디언트가 0이 되어 "엔트로피를 다시 끌어올리는"
        # 효과가 없으므로, 하한 미달분(entropy_clip - entropy)에 비례한 페널티를 손실에
        # 더해 실제로 엔트로피를 하한 쪽으로 밀어올리는 그래디언트를 만든다.
        # (entropy_clip=0이면 기존과 완전히 동일하게 동작.)
        entropy_floor_penalty = torch.tensor(0.0, device=entropy.device)
        if self.cfg.entropy_clip > 0.0:
            entropy_floor_penalty = torch.clamp(
                float(self.cfg.entropy_clip) - entropy, min=0.0
            )

        loss_pi = (
            -(ratio * adv).mean()
            - self.cfg.ent_coef * entropy
            + self.cfg.ent_coef * entropy_floor_penalty
        )

        v_pred = self.critic(obs)
        v_target = ret
        if self.cfg.value_clip > 0:
            v_pred_clipped = val + (v_pred - val).clamp(
                -self.cfg.value_clip, self.cfg.value_clip
            )
            loss_v = 0.5 * torch.max(
                (v_pred - v_target) ** 2, (v_pred_clipped - v_target) ** 2
            ).mean()
        else:
            loss_v = 0.5 * (v_pred - v_target).pow(2).mean()

        loss = loss_pi + self.cfg.vf_coef * loss_v
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            self.cfg.max_grad_norm,
        )
        self.opt.step()

        rew_mean = float(rew.mean().item())
        adv_mean = float(adv.mean().item())
        ret_mean = float(ret.mean().item())
        ent_val = float(entropy.item())
        return float(loss_pi.item()), float(loss_v.item()), ent_val, adv_mean, ret_mean, rew_mean

    def _log(
        self,
        step: int,
        loss_pi: float,
        loss_v: float,
        ent: float,
        adv_mean: float,
        ret_mean: float,
        rew_mean: float,
    ) -> None:
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([step, loss_pi, loss_v, ent, adv_mean, ret_mean, rew_mean, rew_mean])

    def _save(self, path: Optional[str] = None) -> None:
        torch.save(
            {
                "cfg": asdict(self.cfg),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "opt": self.opt.state_dict(),
            },
            path or self.ckpt_path,
        )

    def train(self) -> None:
        cfg = self.cfg
        obs = self.env.reset(seed=cfg.seed)
        step = 0
        if cfg.verbose:
            print(f"[RL] start training tag={cfg.tag} device={self.device} log_dir={self.run_dir}")

        # [FIX 2026-07] 학습이 초반에 좋아졌다가 후반에 정책 붕괴(엔트로피 지속 하락 →
        # 발산)로 다시 나빠지는 현상이 반복적으로 관측되었다. entropy_coef/lr을
        # 조정해도 발산 자체를 완전히 막기보다 "발산 시점을 늦추는" 정도의 효과만
        # 있었으므로, 학습 중 주기적으로 evaluate_mean_policy()로 평가하고 가장
        # 좋았던 시점의 체크포인트를 별도로("best.pt") 저장해 두는 실용적 방식을
        # 도입한다. 최종 결과 보고 시 마지막 체크포인트가 아니라 best.pt를 사용하면
        # 후반부 불안정성의 영향을 받지 않는다.
        eval_every = max(cfg.rollout_len * 5, int(getattr(cfg, "eval_every", 0) or 0))
        best_path = os.path.join(self.run_dir, "best.pt")
        best_return = float("-inf")
        best_step = 0

        while step < cfg.max_steps:
            buf, obs = self._rollout(obs, steps=cfg.rollout_len)
            loss_pi, loss_v, ent, adv_mean, ret_mean, rew_mean = self._update(buf, step)
            step += cfg.rollout_len
            self._log(step, loss_pi, loss_v, ent, adv_mean, ret_mean, rew_mean)

            if cfg.verbose and step % (cfg.rollout_len * 5) == 0:
                print(
                    f"[RL] step={step} loss_pi={loss_pi:.4f} loss_v={loss_v:.4f} "
                    f"ent={ent:.3f} rew_mean={rew_mean:.6f}"
                )

            if (cfg.ckpt_every > 0) and (step % cfg.ckpt_every == 0):
                self._save()

            if eval_every > 0 and step % eval_every == 0:
                ev = self.evaluate_mean_policy(n_episodes=16)
                cur = ev.get("eval_return_mean", float("-inf"))
                if cur > best_return:
                    best_return = cur
                    best_step = step
                    self._save(best_path)
                    if cfg.verbose:
                        print(f"[RL][best] step={step} eval_return_mean={cur:.4f} -> best.pt 갱신")
                self.actor.train()
                self.critic.train()

        if cfg.save_every >= 0:
            self._save()
        if best_step == 0:
            # 평가가 한 번도 안 돌았다면(짧은 학습 등) 마지막 체크포인트를 best로도 저장
            self._save(best_path)
            best_step = step
        if cfg.verbose:
            print(f"[RL] done. logs={self.log_path}  best_step={best_step} best_return={best_return:.4f} best_ckpt={best_path}")

    @torch.no_grad()
    def evaluate_mean_policy(self, n_episodes: int = 32) -> Dict[str, float]:
        """
        학습된 정책의 deterministic mean action(q,w)을 사용해
        n_episodes 회 평가한 평균 리턴/길이/표준편차를 제공하고,
        한국형 guardrail 해석을 위한 q,w 행동 분포 요약치도 함께 반환합니다.
        """
        cfg = self.cfg
        device = self.device
        self.actor.eval()
        self.critic.eval()
        ep_returns: List[float] = []
        ep_lengths: List[int] = []
        all_q: List[float] = []
        all_w: List[float] = []

        for _ in range(n_episodes):
            obs = self.env.reset(seed=None)
            done = False
            ret_sum = 0.0
            t = 0
            while not done:
                obs_t = torch.as_tensor(
                    obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
                a_q, a_w, _, _ = self._act(obs_t, deterministic=True)
                q = float(a_q[0])
                w = float(a_w[0])

                next_obs, rew, done, info = self.env.step({"q": q, "w": w})
                ret_sum += float(rew)
                t += 1
                obs = next_obs

                # guardrail 밴드 추정을 위한 행동 기록
                all_q.append(q)
                all_w.append(w)

            ep_returns.append(ret_sum)
            ep_lengths.append(t)

        result: Dict[str, float] = {
            "eval_return_mean": float(np.mean(ep_returns)) if ep_returns else 0.0,
            "eval_return_std": float(np.std(ep_returns)) if ep_returns else 0.0,
            "eval_len_mean": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
            "episodes": int(n_episodes),
        }

        # guardrail 해석용 q,w 행동 분포 요약치 추가
        if all_q:
            q_arr = np.asarray(all_q, dtype=np.float32)
            w_arr = np.asarray(all_w, dtype=np.float32)

            def _band(arr: np.ndarray, p: float) -> float:
                try:
                    return float(np.quantile(arr, p))
                except Exception:
                    return float("nan")

            result.update(
                {
                    # 전체 분포 요약
                    "q_min": float(np.min(q_arr)),
                    "q_max": float(np.max(q_arr)),
                    "q_mean": float(np.mean(q_arr)),
                    "w_min": float(np.min(w_arr)),
                    # env에 설정된 w_max와 별개로, 실제 정책이 사용하는 상한
                    "w_max_eff": float(np.max(w_arr)),
                    "w_mean": float(np.mean(w_arr)),

                    # guardrail 밴드(대략적인 5–95% 범위)
                    "q_p5": _band(q_arr, 0.05),
                    "q_p25": _band(q_arr, 0.25),
                    "q_p50": _band(q_arr, 0.50),
                    "q_p75": _band(q_arr, 0.75),
                    "q_p95": _band(q_arr, 0.95),

                    "w_p5": _band(w_arr, 0.05),
                    "w_p25": _band(w_arr, 0.25),
                    "w_p50": _band(w_arr, 0.50),
                    "w_p75": _band(w_arr, 0.75),
                    "w_p95": _band(w_arr, 0.95),
                }
            )

        return result

    def load(self, path: str | None = None) -> "RLTrainer":
        p = path or self.ckpt_path
        state = torch.load(p, map_location=self.device)
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.opt.load_state_dict(state["opt"])
        if "cfg" in state:
            s = state["cfg"]
            for k, v in s.items():
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
    parser.add_argument("--log_dir", type=str, default=r".\outputs\_logs")
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

        def step(self, act: Dict[str, float]):
            q, w = act["q"], act["w"]
            r = -((q - 0.4) ** 2 + (w - 0.6) ** 2) + 0.1
            self.t += 1
            done = self.t >= self.T
            obs = self.rng.normal(size=self.obs_dim).astype(np.float32)
            info = {"toy": 1}
            return obs, float(r), bool(done), info

    def env_factory():
        return _ToyEnv(obs_dim=8 if args.obs_dim <= 0 else args.obs_dim)

    cfg = RLConfig(
        obs_dim=args.obs_dim,
        max_steps=args.max_steps,
        rollout_len=args.rollout_len,
        tag=args.tag,
        log_dir=args.log_dir,
        seed=args.seed,
        device=args.device,
    )
    tr = RLTrainer(cfg, env_factory)
    tr.train()
    print(tr.evaluate_mean_policy(n_episodes=8))
