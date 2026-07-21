# -*- coding: utf-8 -*-
"""
통합 θ* 평가기 — HJB 정책과 RL 정책을 '완전히 동일한' harness로 평가한다.
  · 동일 env(γ=3, 플로어 ON, 국민연금 ρ=0.30, 사망률 ON)
  · 중간시점(age_ann) 연금화(부하율 8%)
  · per-path 할인 CRRA 효용을 u_scale로 스케일 후 누적
  · 이상치에 강건한 MEDIAN으로 θ* 판정 (평균 EU의 파국경로 지배 회피)

사용:
  python unified_theta_eval.py --mode hjb
  python unified_theta_eval.py --mode rl --ckpt <best.pt>
"""
import argparse, copy
import numpy as np
from milevsky_timing_RL import (
    load_actor, deterministic_action_batch, make_base_cfg, get_life_table_from_env
)
from milevsky_timing_analysis import solve_theta0_policy
from project.annuity.overlay import compute_ax_real
from project.env.retirement_env import RetirementEnv

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["hjb", "rl"], required=True)
ap.add_argument("--ckpt", default="")
ap.add_argument("--gamma", type=float, default=3.0)
ap.add_argument("--f_min_real", type=float, default=0.072)
ap.add_argument("--floor", choices=["on", "off"], default="off")
ap.add_argument("--q_max_mult", type=float, default=1.5)
ap.add_argument("--n_paths", type=int, default=150)
ap.add_argument("--ages", default="55,60,65")
a = ap.parse_args()

q_cap, w_max, u_scale, ann_load = 0.02, 0.70, 0.0001, 0.08
delta_m = 0.9530 ** (1.0/12.0)
n_months = 420
thetas = [round(0.1*i, 1) for i in range(9)]  # 0.0~0.8
ages = [int(x) for x in a.ages.split(",")]

def make_cfg():
    cfg = make_base_cfg(crra_gamma=a.gamma, asset="KR")
    # ★ floor_on: HJB는 bool(), env는 str()=="on"로 판정 → 둘 다 맞도록
    #   ON = "on"(문자열), OFF = False(불리언)로 설정
    cfg.floor_on = "on" if a.floor == "on" else False
    cfg.f_min_real = a.f_min_real
    cfg.q_floor = 0.0
    return cfg

base_cfg = make_cfg()
probe = RetirementEnv(base_cfg)
life_df = get_life_table_from_env(probe)
r_f = float(getattr(base_cfg, "rf_annual", 0.02))

# ── 정책 actor(배치) 준비: obs_batch -> (q_arr, w_arr) 실제행동 ──
if a.mode == "rl":
    actor = load_actor(a.ckpt)
    def act_batch(obs_batch):
        return deterministic_action_batch(actor, obs_batch, q_cap, w_max)
else:
    # HJB θ=0 정책을 그리드로 풀어 lookup actor 구성 (실제 q,w 반환)
    Pi_w, Pi_q, W_grid = solve_theta0_policy(base_cfg, hjb_W_grid=150, hjb_q_max_mult=a.q_max_mult)
    T = Pi_w.shape[0]
    def act_batch(obs_batch):
        qs, ws = [], []
        for ob in obs_batch:
            t = min(int(round(ob[0]*T)), T-1)
            idx = int(np.argmin(np.abs(W_grid - ob[1])))
            idx = min(idx, Pi_w.shape[1]-1)
            qs.append(float(Pi_q[t, idx])); ws.append(float(Pi_w[t, idx]))
        return np.array(qs), np.array(ws)

def sim_theta(age_ann, theta):
    n = a.n_paths
    month_ann = int((age_ann-55)*12)
    envs=[]; obs_list=[]
    for i in range(n):
        env=RetirementEnv(copy.deepcopy(base_cfg)); obs=env.reset(seed=i); envs.append(env); obs_list.append(obs)
    annu=[theta<=0.0]*n; u_sum=np.zeros(n); done=[False]*n
    a_fac = float(compute_ax_real(age_ann, life_df, r_f, S=12)) if theta>0 and life_df is not None else 0.0
    disc=1.0
    for t in range(n_months):
        if all(done): break
        for i,env in enumerate(envs):
            if done[i]: continue
            if (not annu[i]) and env.t>=month_ann:
                W=env.W; P=theta*W
                yadd=(P/a_fac/(1.0+ann_load)) if a_fac>0 else 0.0
                env.W=max(0.0,W-P); env.y_ann=float(getattr(env,"y_ann",0.0))+yadd; annu[i]=True
        act=[i for i in range(n) if not done[i]]
        ob=np.stack([obs_list[i] for i in act])
        qb,wb=act_batch(ob)
        for j,i in enumerate(act):
            obs,rew,d,info=envs[i].step(q=float(qb[j]),w=float(wb[j]))
            obs_list[i]=obs
            u_sum[i]+=disc*float(info.get("u_eff",0.0))*u_scale
            if d: done[i]=True
        disc*=delta_m
    return u_sum   # per-path 배열 반환 (CRN: seed i가 θ 전반에 공통)

print(f"=== 통합 θ* 평가 (mode={a.mode} γ={a.gamma} floor={a.floor} f_min={a.f_min_real} n={a.n_paths}) ===")
print("[median] = θ별 median EU,  [paired] = θ=0 대비 경로별 차이의 평균(공통난수 상쇄, 저분산)")
for age in ages:
    arrs={th: sim_theta(age, th) for th in thetas}
    u0=arrs[0.0]
    med=[float(np.median(arrs[th])) for th in thetas]
    paired=[float(np.mean(arrs[th]-u0)) for th in thetas]   # θ=0 대비 쌍대차 평균
    ts_med=thetas[int(np.argmax(med))]
    ts_pair=thetas[int(np.argmax(paired))]
    print(f"\n-- age {age} --")
    print("   θ:     " + " ".join(f"{t:>7.1f}" for t in thetas))
    print("  median: " + " ".join(f"{v:>7.3f}" for v in med) + f"   θ*={ts_med}")
    print("  paired: " + " ".join(f"{v:>+7.4f}" for v in paired) + f"   θ*={ts_pair}")
