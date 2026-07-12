"""
Milevsky(2007) 대응: 연금화 시점(x0_ann) x 대표자산 분위수(25/50/75%) 민감도 분석
================================================================================
논문설계방 절충안 구현:
1) theta=0(무연금) 최적정책으로 55->60/65세 시점 자산분포를 순방향 시뮬레이션
2) 각 시점의 25/50/75 분위수를 "대표자산(certainty-equivalent 근사)"으로 채택
3) 각 (연금화시점 x 대표자산) 조합마다 theta 후보별 HJB를 풀고 EU 비교 -> theta*
4) 결과: "연금화 시점 x 자산분위수별 최적 theta*" 매트릭스

사용법:
    python milevsky_timing_analysis.py

주의: 조합 수가 많아 시간이 오래 걸립니다(30~60분 이상 예상).
      --quick 옵션으로 축소된 그리드(검증용)를 먼저 돌려보는 것을 권장합니다.
"""
import sys
import copy
import json
import numpy as np

from project.config import SimConfig
from project.hjb import HJBSolver
from project.annuity.overlay import compute_ax_real
from project.runner.helpers import get_life_table_from_env
from project.env.retirement_env import RetirementEnv
from project.evaluation import evaluate


def make_base_cfg():
    cfg = SimConfig()
    cfg.market_mode = "iid"
    cfg.crra_gamma = 3.0
    cfg.w_max = 0.70
    cfg.pension_rho = 0.30          # 필요시 조정
    cfg.pension_claim_age = 65.0
    cfg.mortality = "on"
    cfg.mort_table = "project/data/kidi_qx.csv"
    cfg.sex = "M"
    return cfg


def solve_theta0_policy(base_cfg, hjb_W_grid=150, hjb_q_max_mult=4.0):
    """1) theta=0 기준 최적정책(q*,w*)을 age0=55, 35년 전체에 대해 미리 풀어둔다."""
    cfg = copy.deepcopy(base_cfg)
    cfg.age0 = 55
    cfg.horizon_years = 35
    cfg.hjb_W_max = 3.0
    cfg.hjb_W_grid = hjb_W_grid
    cfg.hjb_W_focus = 2.0
    cfg.hjb_W_focus_frac = 0.85
    cfg.hjb_q_max_mult = hjb_q_max_mult
    solver = HJBSolver(cfg)
    res = solver.solve(seed=0)
    return res["Pi_w"], res["Pi_q"], res["W_grid"]


def simulate_wealth_at_ages(base_cfg, Pi_w, Pi_q, W_grid, ages, n_paths=500):
    """2) theta=0 정책으로 순방향 시뮬레이션, 후보 연금화 시점(ages)의 자산분포 수집."""
    def policy_actor(t, W):
        idx = int(np.argmin(np.abs(W_grid - W)))
        idx = min(idx, Pi_w.shape[1] - 1)
        tt = min(t, Pi_w.shape[0] - 1)
        return float(Pi_q[tt, idx]), float(Pi_w[tt, idx])

    cfg = copy.deepcopy(base_cfg)
    cfg.age0 = 55
    cfg.horizon_years = 35
    env = RetirementEnv(cfg)

    max_month = max(int((a - 55) * 12) for a in ages)
    target_months = {int((a - 55) * 12): a for a in ages}
    W_by_age = {a: [] for a in ages}

    for seed in range(n_paths):
        env.reset(seed=seed)
        if 0 in target_months:
            W_by_age[target_months[0]].append(env.W)
        for t in range(max_month):
            q, w = policy_actor(env.t, env.W)
            obs, rew, done, info = env.step(q=q, w=w)
            if env.t in target_months:
                W_by_age[target_months[env.t]].append(env.W)
            if done:
                break

    percentiles = {}
    for a in ages:
        arr = np.array(W_by_age[a])
        percentiles[a] = {
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "n": len(arr),
        }
    return percentiles


def solve_and_eval_theta(age_ann, W_rep, theta, base_cfg, life_df, r_f_annual,
                          n_paths=300, hjb_W_grid=100):
    """3) 특정 (연금화시점, 대표자산, theta)에서 HJB를 풀고 EU를 평가."""
    remaining_years = 35 - (age_ann - 55)
    cfg_t = copy.deepcopy(base_cfg)
    cfg_t.age0 = age_ann
    cfg_t.horizon_years = remaining_years

    if theta <= 0.0 or life_df is None:
        y_ann_month, W_after, a_factor = 0.0, W_rep, None
    else:
        a_factor = float(compute_ax_real(age_ann, life_df, r_f_annual, S=12))
        P = theta * W_rep
        y_ann_month = P / a_factor if a_factor > 0 else 0.0
        W_after = max(0.0, W_rep - P)

    cfg_t.y_ann = y_ann_month
    cfg_t.W0 = W_after
    cfg_t.hjb_W_max = max(2.0, W_after * 4.0)
    cfg_t.hjb_W_grid = hjb_W_grid
    cfg_t.hjb_W_focus = max(1.0, W_after * 2.5)
    cfg_t.hjb_W_focus_frac = 0.85
    cfg_t.hjb_q_max_mult = 4.0
    cfg_t.n_paths = n_paths
    cfg_t.seeds = [0]
    cfg_t.report_utility = "on"
    cfg_t.delta_annual = 0.9530
    cfg_t.alpha = 0.95

    solver = HJBSolver(cfg_t)
    res = solver.solve(seed=0)
    Pi_w, Pi_q, W_grid = res["Pi_w"], res["Pi_q"], res["W_grid"]

    def actor(obs):
        t_norm, W = obs[0], obs[1]
        T = Pi_w.shape[0]
        t = min(int(round(t_norm * T)), T - 1)
        idx = int(np.argmin(np.abs(W_grid - W)))
        idx = min(idx, Pi_w.shape[1] - 1)
        return float(Pi_q[t, idx]), float(Pi_w[t, idx])

    m = evaluate(cfg_t, actor, es_mode="wealth")
    return {
        "age_ann": age_ann, "W_rep": W_rep, "theta": theta,
        "a_factor": a_factor, "y_ann_month": y_ann_month, "W_after": W_after,
        "EU": m.get("EU"), "EW": m.get("EW"), "ES95": m.get("ES95"),
    }


def main(quick: bool = False):
    base_cfg = make_base_cfg()

    print("[1/3] theta=0 기준정책 계산 중...")
    Pi_w, Pi_q, W_grid = solve_theta0_policy(
        base_cfg, hjb_W_grid=(80 if quick else 150)
    )

    ages = [60, 65] if not quick else [65]
    print(f"[2/3] {ages}세 시점 자산분포 시뮬레이션 중...")
    pct = simulate_wealth_at_ages(
        base_cfg, Pi_w, Pi_q, W_grid, ages, n_paths=(100 if quick else 500)
    )
    for a in ages:
        print(f"  {a}세: {pct[a]}")

    probe = RetirementEnv(base_cfg)
    life_df = get_life_table_from_env(probe)
    r_f = float(getattr(base_cfg, "rf_annual", 0.02))

    theta_candidates = [0.0, 0.3] if quick else [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    age_wealth_combos = []
    # 55세(t=0) 기준은 W_rep=1.0 확정값(불확실성 없음)
    age_wealth_combos.append((55, 1.0, "deterministic"))
    for a in ages:
        for label in ["p25", "p50", "p75"]:
            age_wealth_combos.append((a, pct[a][label], label))

    print(f"[3/3] {len(age_wealth_combos)}개 (시점,자산) 조합 x {len(theta_candidates)}개 theta 후보 계산 중...")
    all_results = []
    for age_ann, W_rep, label in age_wealth_combos:
        row = {"age_ann": age_ann, "W_rep": W_rep, "percentile": label, "results": []}
        best_theta, best_eu = None, -1e300
        for theta in theta_candidates:
            r = solve_and_eval_theta(
                age_ann, W_rep, theta, base_cfg, life_df, r_f,
                n_paths=(100 if quick else 300),
                hjb_W_grid=(60 if quick else 100),
            )
            row["results"].append(r)
            if r["EU"] is not None and r["EU"] > best_eu:
                best_eu, best_theta = r["EU"], theta
            print(f"    age={age_ann} W={W_rep:.3f}({label}) theta={theta}: EU={r['EU']:.2f}")
        row["theta_star"] = best_theta
        all_results.append(row)
        print(f"  -> age={age_ann} {label}: theta* = {best_theta}")

    with open("milevsky_timing_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("\n결과 저장: milevsky_timing_results.json")

    print("\n=== 요약: 연금화시점 x 자산분위수별 theta* ===")
    for row in all_results:
        print(f"  {row['age_ann']}세 ({row['percentile']:>6s}, W={row['W_rep']:.3f}): theta* = {row['theta_star']}")


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    main(quick=quick)