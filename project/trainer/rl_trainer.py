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

    # [2026-07 신규] HJB 정책 모방학습(behavior cloning) 후 워밍업된 액터 체크포인트를
    # 불러와 PPO 학습의 초기값으로 사용한다(무작위 초기화 대비 훨씬 나은 출발점).
    warm_start_ckpt: str = ""

    # [2026-07 신규] 잔차 정책(Residual Policy) 옵션
    # ------------------------------------------------------------------
    # 무편향 RL을 처음부터 학습시키면 HJB(θ*=0)에서 너무 멀리 벗어난 지점에서 출발해,
    # 그 격차가 편향의 미세한 효과를 전부 삼켜버려 모든 편향에서 θ*≈1.0으로 뭉개졌다.
    # 잔차 정책은 HJB(모방) 정책을 "고정된 기준선"으로 두고, 그 위에 작고(bounded)
    # 상태의존적인 보정(residual)만 학습한다:
    #     action = clip(baseline(s) + tanh(net(s)) * residual_scale, 0, 1)
    # · residual_policy=True 이면 baseline_ckpt(없으면 warm_start_ckpt)의 액터를
    #   frozen baseline으로 로드하고, 그 위에 잔차 네트워크만 학습한다.
    # · 잔차 네트워크의 마지막 층은 0으로 초기화되어, 학습 시작 시점의 정책이 정확히
    #   baseline(=HJB 근사)과 일치한다("보정=0"이 공짜로 얻어짐 → 무편향 θ*≈0).
    # · residual_scale로 보정폭을 제한해, 편향이 있어도 baseline 근방에서만 벗어나도록
    #   구조적으로 강제한다(편향 유형별로 서로 다른 보정을 학습할 여지가 커짐).
    residual_policy: bool = False
    residual_scale: float = 0.15      # 잔차 보정의 최대 절대폭(raw [0,1] 액션 기준)
    residual_l2_coef: float = 0.0     # 잔차 크기에 대한 L2 정규화(0=미사용; 무편향 시 0 유지 유도)
    baseline_ckpt: str = ""           # frozen baseline 액터 체크포인트(없으면 warm_start_ckpt 사용)

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


class ResidualBetaActor(nn.Module):
    """
    잔차 정책(Residual Policy) 액터.

    고정된 baseline 액터(HJB 모방정책)를 감싸고, 그 위에 작고 상태의존적인 보정
    (residual)만 학습한다. 최종 행동의 평균은

        mean = clip(baseline_mean(s) + tanh(res_net(s)) * residual_scale, eps, 1-eps)

    로 정의되며, 이를 Beta(mean·conc, (1-mean)·conc)로 변환해 확률정책으로 사용한다.

    설계 포인트
    -----------
    1) baseline은 완전히 고정(requires_grad=False, eval 모드)한다. 학습되는 것은
       잔차 네트워크(res_net)와 head별 집중도(log_conc)뿐이다.
    2) res_net의 마지막 층은 weight=0, bias=0으로 초기화한다. 따라서 학습 시작
       시점의 tanh(0)=0 → 보정=0 → 정책이 정확히 baseline과 일치한다. 무편향
       상황에서는 어드밴티지가 baseline을 밀어낼 이유가 없으므로 보정이 0 근방에
       머물러 자연스럽게 HJB(θ*≈0)를 재현한다.
    3) residual_scale로 보정폭을 [-scale, +scale]로 제한해, 편향이 있어도 정책이
       baseline 근방에서만 벗어나도록 구조적으로 강제한다.

    반환 시그니처는 BetaActor와 동일하게 (dist_q, dist_w, residual)이며,
    세 번째 원소(residual)는 L2 정규화에 사용할 수 있도록 보정값 [N,2]를 담는다.
    """

    def __init__(
        self,
        baseline: "BetaActor",
        obs_dim: int,
        hidden: List[int],
        residual_scale: float = 0.15,
        init_conc: float = 12.0,
    ):
        super().__init__()
        # ── 고정 baseline (학습 제외) ──
        self.baseline = baseline
        for p in self.baseline.parameters():
            p.requires_grad_(False)
        self.baseline.eval()

        self.residual_scale = float(residual_scale)
        self.softplus = nn.Softplus()

        # ── 학습되는 잔차 네트워크: (delta_q, delta_w) 2개 출력 ──
        self.res_net = MLP(obs_dim, hidden, out_dim=2)
        last = self.res_net.net[-1]
        if isinstance(last, nn.Linear):
            with torch.no_grad():
                last.weight.zero_()
                last.bias.zero_()  # 시작 시 보정=0 → 정책 == baseline

        # ── head별 집중도(탐색폭). conc = softplus(log_conc)+2 ──
        # init_conc≈12 근방이 되도록 raw 파라미터를 역산 초기화.
        raw_init = float(init_conc - 2.0)
        self.log_conc_q = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))
        self.log_conc_w = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

        # 마지막 forward의 보정값(정규화용)
        self.last_residual: Optional[torch.Tensor] = None

    def _baseline_means(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            dq, dw, _ = self.baseline(x)
            mq = dq.concentration1 / (dq.concentration1 + dq.concentration0)
            mw = dw.concentration1 / (dw.concentration1 + dw.concentration0)
        return mq, mw

    def forward(self, x: torch.Tensor) -> Tuple[Beta, Beta, torch.Tensor]:
        mq, mw = self._baseline_means(x)  # (0,1), no grad

        delta = torch.tanh(self.res_net(x)) * self.residual_scale  # [N,2] in [-s, s]
        d_q, d_w = torch.chunk(delta, 2, dim=-1)
        d_q = d_q.reshape(mq.shape)
        d_w = d_w.reshape(mw.shape)

        eps = 1e-4
        mean_q = torch.clamp(mq + d_q, eps, 1.0 - eps)
        mean_w = torch.clamp(mw + d_w, eps, 1.0 - eps)

        conc_q = self.softplus(self.log_conc_q) + 2.0
        conc_w = self.softplus(self.log_conc_w) + 2.0

        a_q = mean_q * conc_q
        b_q = (1.0 - mean_q) * conc_q
        a_w = mean_w * conc_w
        b_w = (1.0 - mean_w) * conc_w

        self.last_residual = delta
        return Beta(a_q, b_q), Beta(a_w, b_w), delta


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
        self.critic = ValueCritic(cfg.obs_dim, cfg.hidden_dims).to(self.device)

        # [2026-07 신규] 잔차 정책 모드: HJB 모방 액터를 frozen baseline으로 두고
        # 그 위에 작은 보정만 학습하는 ResidualBetaActor를 사용한다.
        if getattr(cfg, "residual_policy", False):
            baseline = BetaActor(cfg.obs_dim, cfg.hidden_dims).to(self.device)
            # baseline 가중치 로드: baseline_ckpt 우선, 없으면 warm_start_ckpt 재활용
            base_ckpt = cfg.baseline_ckpt or cfg.warm_start_ckpt
            self.baseline_loaded = False
            self.baseline_error = ""
            if base_ckpt:
                try:
                    bstate = torch.load(base_ckpt, map_location=self.device)
                    baseline.load_state_dict(bstate["actor"])
                    # 크리틱도 함께 로드해 액터-크리틱 스케일을 맞춘다(BC critic 재활용).
                    if "critic" in bstate:
                        self.critic.load_state_dict(bstate["critic"])
                    self.baseline_loaded = True
                    if cfg.verbose:
                        print(f"[RL][residual] baseline 로드 완료(critic 포함={'critic' in bstate}): {base_ckpt}")
                except Exception as e:
                    self.baseline_error = repr(e)
                    if cfg.verbose:
                        print(f"[RL][residual][WARN] baseline 로드 실패({e!r}), bias-init baseline으로 진행")
            elif cfg.verbose:
                print("[RL][residual][WARN] baseline_ckpt/warm_start_ckpt 미지정 → bias-init BetaActor를 baseline으로 사용")

            self.actor = ResidualBetaActor(
                baseline=baseline,
                obs_dim=cfg.obs_dim,
                hidden=cfg.hidden_dims,
                residual_scale=float(getattr(cfg, "residual_scale", 0.15)),
            ).to(self.device)
        else:
            self.actor = BetaActor(cfg.obs_dim, cfg.hidden_dims).to(self.device)

        # [2026-07 신규] HJB 모방학습(BC) 워밍업 체크포인트가 지정되어 있으면 로드.
        # obs_dim/hidden_dims가 정확히 일치해야 하며, 일치하지 않으면 무시하고
        # 기존(무작위/bias-init) 초기화로 안전하게 폴백한다.
        # [FIX 2026-07] cli.py가 학습 중 모든 stdout을 io.StringIO로 가로채 버려서
        # print() 로그로는 성공/실패 여부를 확인할 수 없었다. 인스턴스 속성으로
        # 저장해 두어, run.py가 최종 반환 metrics에 이 값을 포함시킬 수 있게 한다.
        # [FIX 2026-07, 2차] 액터만 로드하고 크리틱은 무작위 초기화로 남겨두면,
        # PPO 시작 직후 크리틱의 잘못된 가치추정이 어드밴티지를 왜곡시켜 모처럼
        # 워밍업된 액터를 초반에 망가뜨리는 문제가 있었다. 체크포인트에 "critic"
        # 키가 있으면 그것도 같이 로드한다(pretrain_bc.py가 이제 크리틱도 저장).
        self.warm_start_loaded = False
        self.warm_start_error = ""
        # 잔차 모드에서는 baseline 로드가 위에서 이미 끝났고, self.actor(ResidualBetaActor)의
        # state_dict 구조가 BC 체크포인트("actor" 키)와 다르므로 이 경로를 건너뛴다.
        if cfg.warm_start_ckpt and not getattr(cfg, "residual_policy", False):
            try:
                state = torch.load(cfg.warm_start_ckpt, map_location=self.device)
                self.actor.load_state_dict(state["actor"])
                if "critic" in state:
                    self.critic.load_state_dict(state["critic"])
                self.warm_start_loaded = True
                if cfg.verbose:
                    print(f"[RL] warm_start_ckpt 로드 완료(critic 포함={'critic' in state}): {cfg.warm_start_ckpt}")
            except Exception as e:
                self.warm_start_error = repr(e)
                if cfg.verbose:
                    print(f"[RL][WARN] warm_start_ckpt 로드 실패({e!r}), 기본 초기화로 진행")

        # frozen baseline 파라미터(requires_grad=False)는 옵티마이저에서 제외한다.
        trainable_params = [
            p for p in (list(self.actor.parameters()) + list(self.critic.parameters()))
            if p.requires_grad
        ]
        self.opt = optim.Adam(
            trainable_params,
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

        # [2026-07 신규] 잔차 정규화: 잔차 정책 모드에서 보정 크기(delta)에 L2 페널티를
        # 부과하면, "굳이 보정할 이유가 없는" 무편향/무의미한 상태에서 보정이 0으로
        # 수렴하도록 추가로 유도한다(baseline=HJB로의 회귀). raw는 forward가 반환한
        # 잔차 delta [N,2]. residual_l2_coef=0이면 완전히 비활성.
        if getattr(self.cfg, "residual_policy", False) and self.cfg.residual_l2_coef > 0.0:
            loss_pi = loss_pi + float(self.cfg.residual_l2_coef) * (raw ** 2).mean()

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