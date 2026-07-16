# -*- coding: utf-8 -*-
"""HJB θ=0 정책의 실제 전략 프로파일(인출률 q, 위험비중 w, 자산 W)을 연령대별로 특성화.
이게 '한국형 4%룰'의 핵심 — q가 4~6%/년 수준인지, 자산이 언제 고갈되는지 본다."""
import argparse, copy
import numpy as np
from milevsky_timing_RL import make_base_cfg, get_life_table_from_env
from milevsky_timing_analysis import solve_theta0_policy
from project.env.retirement_env import RetirementEnv

ap = argparse.ArgumentParser()
ap.add_argument("--gamma", type=float, default=3.0)
ap.add_argument("--f_min_real", type=float, default=0.072)
ap.add_argument("--floor", default="off")
ap.add_argument("--q_max_mult", type=float, default=1.5)
ap.add_argument("--n_paths", type=int, default=300)
ap.add_argument("--rl_ckpt", default="", help="주어지면 HJB 대신 이 RL 체크포인트 정책을 특성화")
a = ap.parse_args()

cfg = make_base_cfg(crra_gamma=a.gamma, asset="KR")
cfg.floor_on = "on" if a.floor == "on" else False
cfg.f_min_real = a.f_min_real
cfg.q_floor = 0.0

if a.rl_ckpt:
    from milevsky_timing_RL import load_actor, deterministic_action_batch
    rl_actor = load_actor(a.rl_ckpt)
    print(f"=== RL 전략 프로파일 (ckpt={a.rl_ckpt}, γ={a.gamma}) ===")
    def act(ob):
        q, w = deterministic_action_batch(rl_actor, np.array([ob]), 0.02, 0.70)
        return float(q[0]), float(w[0])
else:
    Pi_w, Pi_q, W_grid = solve_theta0_policy(cfg, hjb_W_grid=150, hjb_q_max_mult=a.q_max_mult)
    T = Pi_w.shape[0]
    print(f"=== HJB θ=0 전략 프로파일 (γ={a.gamma}, floor={a.floor}, q_max_mult={a.q_max_mult}) ===")
    print(f"Pi_q 범위(월 인출률): {Pi_q.min():.4f} ~ {Pi_q.max():.4f}  (연환산 최대 ~ {(1-(1-Pi_q.max())**12)*100:.1f}%)")
    def act(ob):
        t=min(int(round(ob[0]*T)),T-1); idx=int(np.argmin(np.abs(W_grid-ob[1]))); idx=min(idx,Pi_w.shape[1]-1)
        return float(Pi_q[t,idx]), float(Pi_w[t,idx])

n=a.n_paths
# 연령대별(55,60,65,70,75,80,85,90) q(연율), w, W, 소비 기록
buckets={55:[],60:[],65:[],70:[],75:[],80:[],85:[],90:[]}
rec={k:{"q":[],"w":[],"W":[],"c":[],"alive":0} for k in buckets}
for i in range(n):
    env=RetirementEnv(copy.deepcopy(cfg)); ob=env.reset(seed=i); done=False
    while not done:
        age=55+env.t/12.0
        q,w=act(ob)
        ob2,rew,done,info=env.step(q=q,w=w)
        ak=int(round(age/5.0)*5)
        if ak in rec:
            rec[ak]["q"].append(q*12*100)      # 월→연 근사 %
            rec[ak]["w"].append(w*100)
            rec[ak]["W"].append(info.get("W",0.0))
            rec[ak]["c"].append(info.get("consumption",0.0))
        ob=ob2

print(f"\n{'연령':>4} | {'인출률q(연%)':>12} | {'위험비중w(%)':>12} | {'평균자산W':>10} | {'생존자산>0비율':>12}")
for k in sorted(rec):
    if not rec[k]["q"]: continue
    q=np.mean(rec[k]["q"]); w=np.mean(rec[k]["w"]); W=np.mean(rec[k]["W"])
    alive=np.mean(np.array(rec[k]["W"])>1e-6)*100
    print(f"{k:>4} | {q:>12.1f} | {w:>12.1f} | {W:>10.3f} | {alive:>11.0f}%")
