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
from project.trainer.rl_trainer import BetaActor, ResidualBetaActor


def load_actor(ckpt_path: str, obs_dim: int = 3, hidden=(128, 128)):
    """
    RL 체크포인트에서 액터를 복원한다.

    [2026-07] 잔차 정책(residual_policy=on)으로 학습한 best.pt는 액터가
    ResidualBetaActor이며 state_dict 키가 일반 BetaActor와 다르다
    (baseline.*, res_net.*, log_conc_*). 체크포인트 구조를 자동 감지해
    올바른 액터를 재구성한다. 두 액터 모두 forward가 (dist_q, dist_w, _)를
    반환하므로 이후 deterministic_action_batch는 그대로 동작한다.
    """
    state = torch.load(ckpt_path, map_location="cpu")
    actor_sd = state["actor"]
    is_residual = any(
        k.startswith("res_net.") or k.startswith("baseline.")
        for k in actor_sd.keys()
    )
    if is_residual:
        cfgd = state.get("cfg", {}) or {}
        rscale = float(cfgd.get("residual_scale", 0.15))
        baseline = BetaActor(obs_dim, list(hidden))
        actor = ResidualBetaActor(
            baseline, obs_dim, list(hidden), residual_scale=rscale
        )
        actor.load_state_dict(actor_sd)
    else:
        actor = BetaActor(obs_dim, list(hidden))
        actor.load_state_dict(actor_sd)
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
                                          n_months=420, u_scale=0.0001, ann_load=0.08):
    """[속도개선] n_paths개 환경을 리스트로 두고, 매 스텝마다 전체 경로의 관측치를
    한꺼번에 모아 배치로 신경망을 통과시킨다(환경 자체는 개별이지만 신경망 호출만 배치화).

    [FIX 2026-07] 기존엔 info["u_eff"](u_scale 적용 전 원본 CRRA 효용, 소비가 조금만
    낮아져도 -1e4~-1e6 단위로 폭발)를 그대로 누적하고 있었다. 그런데 RL 정책은
    실제로 base_reward = u_scale * u_eff (u_scale=0.0001)를 보상으로 학습했으므로,
    평가지표가 학습 목표와 스케일이 맞지 않아 정책 품질과 무관하게 원본 CRRA의
    극단치에 압도되는 문제가 있었다(θ=1.0만 시장노출이 없어 이 문제를 원천적으로
    피해가므로 항상 압도적으로 이기는 것처럼 보였음). u_scale을 곱해 학습 시와
    동일한 스케일로 평가한다.

    [FIX 2026-07, 2차] 연금 매입액을 부하율(사업비/이윤) 없이 순보험료(공정가격)
    그대로 계산하고 있었다. 현실의 보험사는 항상 부하를 얹어 판매하므로, 이를
    반영하지 않으면 연금화가 실제보다 유리하게(θ*가 인위적으로 높게) 나올 수
    있다. ann_load(기본 8%)만큼 매입비용을 더 지불하는 것으로 반영한다:
    실제 지급액 = θ*W / (1+ann_load) 만큼만 연금소득으로 환산(같은 P를 내고
    더 적은 보장소득을 받는 것과 동일).
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
                # [FIX] 부하율 반영: 같은 보험료(P)를 내고 (1+ann_load)로 나눈 만큼만 지급
                y_ann_add = (P / a_factor_cache / (1.0 + ann_load)) if (a_factor_cache and a_factor_cache > 0) else 0.0
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
            # [FIX 2026-07, 3차] info["u_eff"]는 현재편향(β) 할인이 적용되기 "전" 값이다
            # (retirement_env.py에서 disc_factor는 reward에만 곱해지고 u_eff에는 안 곱해짐).
            # 여기서 자체 표준할인(delta_m)만 곱하면, 실제 정책이 학습된 β 조정이
            # presentbias 조건 분석에서 완전히 누락된다. info["disc_factor"](β 반영,
            # beta=1.0인 다른 조건에서는 항상 1.0이라 무해함)를 추가로 곱해 보정한다.
            env_disc = float(info.get("disc_factor", 1.0))
            u_sum[i] += disc * env_disc * float(info.get("u_eff", 0.0)) * u_scale
            if done:
                done_flags[i] = True
        disc *= delta_m

    WT = np.array([env.W for env in envs])
    # [FIX 2026-07] 평균(mean) EU가 극소수의 파국적 경로(소비가 거의 0에 가까워
    # CRRA 효용이 천문학적으로 나빠지는 경로)에 크게 좌우되어, theta=1.0(완전
    # 연금화=시장위험 완전 차단)이 인위적으로 항상 이기는 현상이 나타났다.
    # 중위값(median)과 절사평균(상하 10% 제외)을 추가로 계산해 극단치 영향을
    # 줄인 지표로 theta*를 다시 판단할 수 있게 한다.
    u_sorted = np.sort(u_sum)
    trim = max(1, int(len(u_sorted) * 0.1))
    trimmed = u_sorted[trim:-trim] if len(u_sorted) > 2 * trim else u_sorted
    return {
        "theta": theta, "age_ann": age_ann,
        "EU_mean": float(np.mean(u_sum)),
        "EU_median": float(np.median(u_sum)),
        "EU_trimmed_mean": float(np.mean(trimmed)),
        "EW_mean": float(np.mean(WT)),
        "EW_median": float(np.median(WT)),
        "n_paths": n_paths,
    }


def make_base_cfg(bias_kwargs=None, crra_gamma=3.0, social_floor_on=False, social_floor_min=0.0, asset="KR"):
    from project.config import ASSET_PRESETS
    cfg = SimConfig()
    cfg.market_mode = "iid"
    cfg.horizon_years = 35
    cfg.crra_gamma = crra_gamma
    cfg.w_max = 0.70
    cfg.pension_rho = 0.30
    cfg.pension_claim_age = 65.0
    cfg.mortality = "on"
    cfg.mort_table = "project/data/kidi_qx.csv"
    cfg.sex = "M"
    cfg.age0 = 55
    cfg.social_floor_on = "on" if social_floor_on else "off"
    cfg.social_floor_min = social_floor_min
    # [FIX 2026-07] 자산 프리셋(mu_annual/sigma_annual)을 명시적으로 반영하지 않으면
    # SimConfig 기본값(mu=0.06, sigma=0.20 — KR도 TDF도 아닌 제3의 값)이 쓰여
    # 학습 때 사용한 자산가정과 분석 환경이 어긋나는 문제가 있었다.
    cfg.asset = asset
    if asset in ASSET_PRESETS:
        for k, v in ASSET_PRESETS[asset].items():
            setattr(cfg, k, v)
    if bias_kwargs:
        for k, v in bias_kwargs.items():
            setattr(cfg, k, v)
    return cfg


def main():
    if len(sys.argv) < 2:
        print("사용법: python milevsky_timing_RL.py <best_pt_경로> [--quick] [--gamma 1.0]")
        sys.exit(1)
    ckpt_path = sys.argv[1]
    quick = "--quick" in sys.argv
    # [FIX 2026-07] 정책이 학습된 gamma와 분석 환경의 gamma가 일치해야 효용
    # 계산이 의미를 가진다. --gamma로 지정 가능하게 하고, 기본값은 3.0(기존과 동일).
    gamma = 3.0
    if "--gamma" in sys.argv:
        idx = sys.argv.index("--gamma")
        if idx + 1 < len(sys.argv):
            gamma = float(sys.argv[idx + 1])
    social_floor_on = "--social_floor_on" in sys.argv
    social_floor_min = 0.0
    if "--social_floor_min" in sys.argv:
        idx = sys.argv.index("--social_floor_min")
        if idx + 1 < len(sys.argv):
            social_floor_min = float(sys.argv[idx + 1])
    asset = "KR"
    if "--asset" in sys.argv:
        idx = sys.argv.index("--asset")
        if idx + 1 < len(sys.argv):
            asset = sys.argv[idx + 1]

    base_cfg = make_base_cfg(crra_gamma=gamma, social_floor_on=social_floor_on, social_floor_min=social_floor_min, asset=asset)
    actor = load_actor(ckpt_path)
    q_cap = 0.02
    w_max = 0.70

    probe = RetirementEnv(base_cfg)
    life_df = get_life_table_from_env(probe)
    r_f = float(getattr(base_cfg, "rf_annual", 0.02))

    ages = [65] if quick else list(range(55, 66))  # 55~65세, 1세 단위
    theta_candidates = [0.0, 0.3] if quick else [round(0.1 * i, 1) for i in range(11)]  # 0.0~1.0, 0.1 단위(11개)로 세분화
    n_paths = 60 if quick else 150  # 배치화 덕분에 경로수를 다시 늘려도 감당 가능

    all_results = []
    for age_ann in ages:
        row = {"age_ann": age_ann, "results": []}
        best_theta, best_eu_median = None, -1e300
        for theta in theta_candidates:
            r = simulate_with_midpoint_annuity_batch(
                actor, base_cfg, q_cap, w_max, age_ann, theta,
                life_df, r_f, n_paths=n_paths,
            )
            row["results"].append(r)
            print(f"  age={age_ann} theta={theta}: "
                  f"EU_mean={r['EU_mean']:.2f}  EU_median={r['EU_median']:.2f}  "
                  f"EU_trimmed={r['EU_trimmed_mean']:.2f}  EW_median={r['EW_median']:.4f}")
            # [FIX 2026-07] 극단치에 취약한 mean 대신 median을 theta* 판단 기준으로 사용
            if r["EU_median"] > best_eu_median:
                best_eu_median, best_theta = r["EU_median"], theta
        row["theta_star"] = best_theta
        row["theta_star_criterion"] = "EU_median"
        all_results.append(row)
        print(f"-> age={age_ann}: theta* (median 기준) = {best_theta}\n")

    with open("milevsky_timing_RL_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("결과 저장: milevsky_timing_RL_results.json")

    print("\n=== 요약 (RL/행동편향, theta*는 EU_median 기준) ===")
    for row in all_results:
        print(f"  {row['age_ann']}세: theta* = {row['theta_star']}")


if __name__ == "__main__":
    main()