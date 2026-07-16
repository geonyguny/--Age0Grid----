"""
HJB 정책 모방학습(Behavior Cloning) 사전학습 스크립트
========================================================
HJB로 이미 풀어놓은 (거의) 최적정책 (q*, w*)을 지도학습으로 RL 액터(BetaActor)에
먼저 모방시켜서, PPO 학습이 무작위 초기화가 아니라 훨씬 나은 출발점에서 시작하도록
한다. 그 다음 project.runner.cli의 --warm_start_ckpt 옵션으로 이 체크포인트를
불러와 기존 PPO 파이프라인으로 미세조정(fine-tuning)하면 된다.

사용법:
    python pretrain_bc.py --asset TDF --crra_gamma 3.0 --out bc_g3.pt
    python pretrain_bc.py --asset TDF --crra_gamma 1.0 --out bc_g1.pt

이후:
    python -m project.runner.cli --method rl ... --crra_gamma 3.0 --asset TDF
        --warm_start_ckpt bc_g3.pt --tag rl_bc_g3_base_s0 ...
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from project.config import SimConfig, ASSET_PRESETS
from project.hjb import HJBSolver
from project.trainer.rl_trainer import BetaActor, ValueCritic
from project.env.retirement_env import RetirementEnv


def estimate_returns_and_pretrain_critic(actor, cfg: SimConfig, q_cap: float, w_max: float,
                                          delta_annual: float = 0.9530, u_scale: float = 0.0001,
                                          n_starts: int = 150, n_mc: int = 8,
                                          hidden=(128, 128), epochs: int = 200, lr: float = 1e-3,
                                          seed: int = 0):
    """
    [2026-07 신규] 액터만 HJB를 모방하고 크리틱은 무작위 초기화 상태로 두면, PPO
    미세조정 시작 직후 크리틱이 이 정책의 실제 가치를 완전히 잘못 추정해 어드밴티지
    (실제수익-크리틱예측)가 크게 왜곡되고, 그 결과 모처럼 워밍업된 액터가 초반
    몇 스텝만에 크게 흔들리는 문제가 있었다. BC로 학습된 액터를 실제로 순방향
    몬테카를로 시뮬레이션해서 각 (나이,자산,소득) 시작점의 실현 수익을 추정하고,
    크리틱을 이 값에 미리 회귀시켜 액터와 크리틱이 서로 맞는 상태로 PPO에 넘긴다.
    """
    rng = np.random.default_rng(seed)
    T = int(round(float(cfg.horizon_years) * int(getattr(cfg, "steps_per_year", 12) or 12)))
    delta_m = float(delta_annual) ** (1.0 / 12.0)

    env = RetirementEnv(cfg)

    def deterministic_action(ob):
        with torch.no_grad():
            ob_t = torch.as_tensor(ob, dtype=torch.float32).unsqueeze(0)
            dist_q, dist_w, _ = actor(ob_t)
            aq = float((dist_q.concentration1 / (dist_q.concentration1 + dist_q.concentration0)).item())
            aw = float((dist_w.concentration1 / (dist_w.concentration1 + dist_w.concentration0)).item())
        return np.clip(aq, 0.0, 1.0) * q_cap, np.clip(aw, 0.0, 1.0) * w_max

    obs_list, ret_list = [], []
    print(f"[BC-critic] {n_starts}개 시작점 x {n_mc}회 몬테카를로 수익추정 중...")
    for si in range(n_starts):
        t0 = int(rng.integers(0, max(1, T - 1)))
        W0 = float(rng.uniform(0.05, 2.0))

        mc_returns = []
        obs0 = None
        for _ in range(n_mc):
            env.reset(seed=int(rng.integers(0, 1_000_000)))
            env.t = t0
            env.age_years = float(cfg.age0) + (t0 / max(1, int(getattr(cfg, "steps_per_year", 12) or 12)))
            env.W = W0
            ob = env._obs().astype(np.float32)
            if obs0 is None:
                obs0 = ob
            disc = 1.0
            ret = 0.0
            done = False
            while not done:
                q, w = deterministic_action(ob)
                ob, rew, done, info = env.step(q=q, w=w)
                ret += disc * float(info.get("u_eff", 0.0)) * u_scale
                disc *= delta_m
            mc_returns.append(ret)

        obs_list.append(obs0)
        ret_list.append(float(np.mean(mc_returns)))

    obs_arr = np.asarray(obs_list, dtype=np.float32)
    ret_arr = np.asarray(ret_list, dtype=np.float32)
    print(f"[BC-critic] 수익추정 완료. 평균={ret_arr.mean():.4f}, 표준편차={ret_arr.std():.4f}")

    critic = ValueCritic(obs_arr.shape[1], list(hidden), init_value=float(ret_arr.mean()))
    opt = optim.Adam(critic.parameters(), lr=lr)
    obs_t = torch.as_tensor(obs_arr, dtype=torch.float32)
    ret_t = torch.as_tensor(ret_arr, dtype=torch.float32)

    for ep in range(epochs):
        pred = critic(obs_t)
        loss = ((pred - ret_t) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (ep + 1) % 50 == 0 or ep == 0:
            print(f"[BC-critic] epoch={ep+1}/{epochs} loss={float(loss.item()):.6f}")

    return critic


def make_cfg(asset: str, crra_gamma: float, w_max: float, pension_rho: float,
             hjb_W_grid: int = 150, hjb_q_max_mult: float = 4.0) -> SimConfig:
    cfg = SimConfig()
    cfg.market_mode = "iid"
    cfg.horizon_years = 35
    cfg.age0 = 55
    cfg.crra_gamma = crra_gamma
    cfg.w_max = w_max
    cfg.pension_rho = pension_rho
    cfg.pension_claim_age = 65.0
    cfg.asset = asset
    if asset in ASSET_PRESETS:
        for k, v in ASSET_PRESETS[asset].items():
            setattr(cfg, k, v)
    cfg.hjb_W_max = 3.0
    cfg.hjb_W_grid = hjb_W_grid
    cfg.hjb_W_focus = 2.0
    cfg.hjb_W_focus_frac = 0.85
    cfg.hjb_q_max_mult = hjb_q_max_mult
    return cfg


def build_bc_dataset(cfg: SimConfig, Pi_q: np.ndarray, Pi_w: np.ndarray, W_grid: np.ndarray,
                      q_cap: float, w_max: float):
    """HJB 정책 그리드를 (관측값, 목표행동[0,1]) 지도학습 데이터셋으로 변환."""
    T, nW = Pi_q.shape
    spm = int(getattr(cfg, "steps_per_year", 12) or 12)

    pension_rho = float(getattr(cfg, "pension_rho", 0.0) or 0.0)
    pension_mult = float(getattr(cfg, "pension_income_mult", 3.692) or 3.692)
    claim_age = float(getattr(cfg, "pension_claim_age", 65.0) or 65.0)
    age0 = float(getattr(cfg, "age0", 55) or 55)
    claim_month = max(0, int(round((claim_age - age0) * spm)))
    pension_y_month = (pension_rho / (pension_mult * 12.0)) if pension_rho > 0 else 0.0

    obs_list, act_list = [], []
    for t in range(T):
        t_norm = t / max(1, T - 1)
        income = pension_y_month if t >= claim_month else 0.0
        for wi in range(nW):
            W = float(W_grid[wi])
            q_target = float(Pi_q[t, wi])
            w_target = float(Pi_w[t, wi])
            # RL 액션 공간은 [0,1] 원시값을 q_cap/w_max로 스케일링하는 구조이므로 역변환
            raw_q = np.clip(q_target / q_cap, 0.0, 1.0) if q_cap > 0 else 0.0
            raw_w = np.clip(w_target / w_max, 0.0, 1.0) if w_max > 0 else 0.0
            obs_list.append([t_norm, W, income])
            act_list.append([raw_q, raw_w])

    return np.asarray(obs_list, dtype=np.float32), np.asarray(act_list, dtype=np.float32)


def pretrain(obs: np.ndarray, act: np.ndarray, obs_dim: int = 3, hidden=(128, 128),
             epochs: int = 300, lr: float = 1e-3, batch_size: int = 512, seed: int = 0,
             target_concentration: float = 8.0, conc_reg_coef: float = 0.01):
    """
    [FIX 2026-07] 순수 MSE(평균만 맞춤) 손실은 Beta의 집중도(a,b)를 전혀 제약하지
    않는다 — 평균 a/(a+b)는 a,b를 같은 비율로 키워도 불변이므로, 학습이 진행될수록
    집중도가 한없이 커져(=거의 결정적/뾰족한 분포) 이후 PPO 미세조정 시 비율(ratio)
    계산이 극도로 민감해지고 학습이 급격히 불안정해지는 문제가 있었다. 집중도 합
    (a+b)이 target_concentration 근방에 머물도록 정규화항을 추가해, 평균은 HJB
    정책에 맞추되 적당한 탐색 여지(엔트로피)를 유지한 채로 PPO에 넘겨준다.
    """
    torch.manual_seed(seed)
    actor = BetaActor(obs_dim, list(hidden))
    opt = optim.Adam(actor.parameters(), lr=lr)

    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    act_t = torch.as_tensor(act, dtype=torch.float32)
    n = obs_t.shape[0]

    for ep in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            ob = obs_t[idx]
            ac = act_t[idx]

            dist_q, dist_w, _ = actor(ob)
            mean_q = dist_q.concentration1 / (dist_q.concentration1 + dist_q.concentration0)
            mean_w = dist_w.concentration1 / (dist_w.concentration1 + dist_w.concentration0)

            mse = ((mean_q - ac[:, 0]) ** 2).mean() + ((mean_w - ac[:, 1]) ** 2).mean()

            conc_q = dist_q.concentration1 + dist_q.concentration0
            conc_w = dist_w.concentration1 + dist_w.concentration0
            conc_penalty = (
                ((conc_q - target_concentration) ** 2).mean()
                + ((conc_w - target_concentration) ** 2).mean()
            )

            loss = mse + conc_reg_coef * conc_penalty
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * ob.shape[0]

        if (ep + 1) % 20 == 0 or ep == 0:
            with torch.no_grad():
                dq, dw, _ = actor(obs_t[:512])
                avg_conc = float((dq.concentration1 + dq.concentration0).mean().item())
            print(f"[BC] epoch={ep+1}/{epochs} loss={total_loss/n:.6f} avg_conc_q={avg_conc:.2f}")

    return actor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", type=str, default="TDF")
    ap.add_argument("--crra_gamma", type=float, default=3.0)
    ap.add_argument("--w_max", type=float, default=0.70)
    ap.add_argument("--pension_rho", type=float, default=0.30)
    ap.add_argument("--q_cap", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--critic_n_starts", type=int, default=150,
                    help="크리틱 사전학습용 몬테카를로 시작점 개수")
    ap.add_argument("--critic_n_mc", type=int, default=8,
                    help="시작점당 몬테카를로 반복횟수")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    print(f"[BC] HJB 정책 계산 중... asset={args.asset} gamma={args.crra_gamma}")
    cfg = make_cfg(args.asset, args.crra_gamma, args.w_max, args.pension_rho)
    solver = HJBSolver(cfg)
    res = solver.solve(seed=0)
    Pi_w, Pi_q, W_grid = res["Pi_w"], res["Pi_q"], res["W_grid"]
    print(f"[BC] HJB 정책 계산 완료. grid shape={Pi_q.shape}")

    print("[BC] 지도학습 데이터셋 구성 중...")
    obs, act = build_bc_dataset(cfg, Pi_q, Pi_w, W_grid, q_cap=args.q_cap, w_max=args.w_max)
    print(f"[BC] 데이터셋 크기: {obs.shape[0]}개 샘플")

    print("[BC] 모방학습 시작...")
    actor = pretrain(obs, act, epochs=args.epochs)

    print("[BC-critic] 크리틱 사전학습 시작 (액터와 스케일을 맞추기 위함)...")
    critic = estimate_returns_and_pretrain_critic(
        actor, cfg, q_cap=args.q_cap, w_max=args.w_max,
        n_starts=args.critic_n_starts, n_mc=args.critic_n_mc,
    )

    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, args.out)
    print(f"[BC] 완료. 저장: {args.out}")
    print(f"[BC] 다음 단계: --warm_start_ckpt {args.out} 옵션으로 PPO 미세조정 학습을 진행하세요.")


if __name__ == "__main__":
    main()
