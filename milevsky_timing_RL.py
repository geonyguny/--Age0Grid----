"""
Milevsky(2007) 대응 (RL/행동편향 버전): 이미 학습된 RL 정책(best.pt)을 그대로 재사용하여,
55->60/65세 사이 임의 시점에 종신연금을 매입하는 이벤트를 시뮬레이션 중간에 주입하고,
그 결과로 theta*가 달라지는지 확인한다.

HJB 버전과의 핵심 차이:
  - HJB는 각 (연금화시점,대표자산) 조합마다 "그 시점부터 남은 기간"을 새로 최적화(재최적화)한다.
  - RL은 55세부터 학습된 "고정된 정책 함수"를 그대로 쓰고, 다만 시뮬레이션 도중 특정 시점에
    "자산의 theta%를 연금으로 전환"하는 이벤트만 주입한다(정책 자체는 재학습하지 않음).
    이는 "이미 습관화된 행동편향을 가진 사람이, 어느 시점에 연금에 가입하면 어떻게 되는가"를
    보는 것이라 RL(행동편향 학습 결과)의 성격에 더 잘 맞는다.

사용법:
    python milevsky_timing_RL.py <best_pt_경로> [--quick]

    예: python milevsky_timing_RL.py "outputs\\_logs\\rl_rl5b_lossaversion_s0_..\\best.pt"

주의:
  - <best_pt_경로>는 이미 완료된 RL 학습 실행의 best.pt 파일 경로입니다.
  - 이 스크립트는 재학습을 하지 않으므로 HJB 버전보다 훨씬 빠릅니다(파일당 몇 분 내외 예상).
"""
import sys
import json
import copy
import numpy as np
import torch

from project.config import SimConfig
from project.env.retirement_env import RetirementEnv
from project.annuity.overlay import compute_ax_real
from project.runner.helpers import get_life_table_from_env
from project.trainer.rl_trainer import BetaActor


def load_actor(ckpt_path: str, obs_dim: int = 2, hidden=(128, 128)) -> BetaActor:
    state = torch.load(ckpt_path, map_location="cpu")
    actor = BetaActor(obs_dim, list(hidden))
    actor.load_state_dict(state["actor"])
    actor.eval()
    return actor


def deterministic_action_batch(actor: BetaActor, obs_batch: np.ndarray, q_cap: float, w_max: float):
    """여러 경로의 관측치를 한 번에(배치로) 통과시켜 결정적 행동을 뽑는다.
    [속도개선 2026-07] 경로 하나씩 순차로 forward하던 것을 배치 처리로 바꿔
    같은 계산량 대비 수십 배 이상 빨라진다(신경망 순전파는 배치화에 매우 유리)."""
    with torch.no_grad():
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32)
        dist_q, dist_w, _ = actor(obs_t)
        a_q = (dist_q.concentration1 / (dist_q.concentration1 + dist_q.concentration0)).numpy().reshape(-1)
        a_w = (dist_w.concentration1 / (dist_w.concentration1 + dist_w.concentration0)).numpy().reshape(-1)
    q = np.clip(a_q, 0.0, 1.0) * q_cap
    w = np.clip(a_w, 0.0, 1.0) * w_max
    return q, w


def simulate_with_midpoint_annuity_batch(actor, base_cfg, q_cap, w_max, age_ann, theta,
                                          life_df, r_f_annual, n_paths=200, seed0=0,
                                          n_months=420):
    """[속도개선] n_paths개 환경을 리스트로 두고, 매 스텝마다 전체 경로의 관측치를
    한꺼번에 모아 배치로 신경망을 통과시킨다(환경 자체는 개별이지만 신경망 호출만 배치화).
    """
    month_ann = int((age_ann - 55) * 12)
    envs = []
    obs_list = []
    for i in range(n_paths):
        cfg = copy.deepcopy(base_cfg)
        env = RetirementEnv(cfg)
        obs = env.reset(seed=seed0 + i)
        envs.append(env)
        obs_list.append(obs)

    annuitized = [theta <= 0.0] * n_paths
    u_sum = np.zeros(n_paths)
    delta_m = 0.9530 ** (1.0 / 12.0)
    disc = 1.0
    done_flags = [False] * n_paths
    a_factor_cache = None
    if theta > 0.0 and life_df is not None:
        a_factor_cache = float(compute_ax_real(age_ann, life_df, r_f_annual, S=12))

    for t in range(n_months):
        if all(done_flags):
            break
        # 연금화 이벤트(해당 시점 도달 시 1회) - 벡터화하기 어려운 개별 상태변경이라 루프 유지(가볍다)
        for i, env in enumerate(envs):
            if done_flags[i]:
                continue
            if (not annuitized[i]) and env.t >= month_ann:
                W_now = env.W
                P = theta * W_now
                y_ann_add = (P / a_factor_cache) if (a_factor_cache and a_factor_cache > 0) else 0.0
                env.W = max(0.0, W_now - P)
                env.y_ann = float(getattr(env, "y_ann", 0.0)) + y_ann_add
                annuitized[i] = True

        active_idx = [i for i in range(n_paths) if not done_flags[i]]
        if not active_idx:
            break
        obs_batch = np.stack([obs_list[i] for i in active_idx])
        q_batch, w_batch = deterministic_action_batch(actor, obs_batch, q_cap, w_max)

        for j, i in enumerate(active_idx):
            obs, rew, done, info = envs[i].step(q=float(q_batch[j]), w=float(w_batch[j]))
            obs_list[i] = obs
            u_sum[i] += disc * float(info.get("u_eff", 0.0))
            if done:
                done_flags[i] = True
        disc *= delta_m

    WT = np.array([env.W for env in envs])
    return {
        "theta": theta, "age_ann": age_ann,
        "EU_mean": float(np.mean(u_sum)),
        "EW_mean": float(np.mean(WT)),
        "n_paths": n_paths,
    }


def make_base_cfg(bias_kwargs=None):
    cfg = SimConfig()
    cfg.market_mode = "iid"
    cfg.horizon_years = 35
    cfg.crra_gamma = 3.0
    cfg.w_max = 0.70
    cfg.pension_rho = 0.30
    cfg.pension_claim_age = 65.0
    cfg.mortality = "on"
    cfg.mort_table = "project/data/kidi_qx.csv"
    cfg.sex = "M"
    cfg.age0 = 55
    if bias_kwargs:
        for k, v in bias_kwargs.items():
            setattr(cfg, k, v)
    return cfg


def main():
    if len(sys.argv) < 2:
        print("사용법: python milevsky_timing_RL.py <best_pt_경로> [--quick]")
        sys.exit(1)
    ckpt_path = sys.argv[1]
    quick = "--quick" in sys.argv

    base_cfg = make_base_cfg()
    actor = load_actor(ckpt_path)
    q_cap = 0.02
    w_max = 0.70

    probe = RetirementEnv(base_cfg)
    life_df = get_life_table_from_env(probe)
    r_f = float(getattr(base_cfg, "rf_annual", 0.02))

    ages = [65] if quick else list(range(55, 66))  # 55~65세, 1세 단위
    theta_candidates = [0.0, 0.3] if quick else [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # 0.6에서 안 꺾여 상한 확장
    n_paths = 60 if quick else 150  # 배치화 덕분에 경로수를 다시 늘려도 감당 가능

    all_results = []
    for age_ann in ages:
        row = {"age_ann": age_ann, "results": []}
        best_theta, best_eu = None, -1e300
        for theta in theta_candidates:
            r = simulate_with_midpoint_annuity_batch(
                actor, base_cfg, q_cap, w_max, age_ann, theta,
                life_df, r_f, n_paths=n_paths,
            )
            row["results"].append(r)
            print(f"  age={age_ann} theta={theta}: EU_mean={r['EU_mean']:.2f}  EW_mean={r['EW_mean']:.4f}")
            if r["EU_mean"] > best_eu:
                best_eu, best_theta = r["EU_mean"], theta
        row["theta_star"] = best_theta
        all_results.append(row)
        print(f"-> age={age_ann}: theta* = {best_theta}\n")

    with open("milevsky_timing_RL_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("결과 저장: milevsky_timing_RL_results.json")

    print("\n=== 요약 (RL/행동편향) ===")
    for row in all_results:
        print(f"  {row['age_ann']}세: theta* = {row['theta_star']}")


if __name__ == "__main__":
    main()