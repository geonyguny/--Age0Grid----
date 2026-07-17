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
ap.add_argument("--w_max", type=float, default=0.70)
ap.add_argument("--pension_rho", type=float, default=0.30)
ap.add_argument("--asset", default="KR", help="위험자산 프리셋: KR/US/Gold/TDF")
ap.add_argument("--mu_annual", type=float, default=0.0, help=">0이면 위험자산 기대수익률 직접 지정(분산 포트폴리오용)")
ap.add_argument("--sigma_annual", type=float, default=0.0, help=">0이면 위험자산 변동성 직접 지정")
ap.add_argument("--bequest_kappa", type=float, default=0.0)
ap.add_argument("--survival", choices=["on","off"], default="off")
ap.add_argument("--w_grid_n", type=int, default=8)
ap.add_argument("--n_paths", type=int, default=300)
ap.add_argument("--rl_ckpt", default="", help="주어지면 HJB 대신 이 RL 체크포인트 정책을 특성화")
ap.add_argument("--labor_wan", type=float, default=0.0, help="계속고용 월 근로소득(만원). W0=1.0=1.37억 기준으로 정규화")
ap.add_argument("--labor_until", type=float, default=0.0, help="근로소득 종료 연령(예: 60=55~59세, 65=55~64세)")
a = ap.parse_args()
W0_WAN = 13700.0  # W0=1.0 = 1.37억원(만원 단위)

cfg = make_base_cfg(crra_gamma=a.gamma, asset=a.asset)
cfg.floor_on = "on" if a.floor == "on" else False
cfg.f_min_real = a.f_min_real
cfg.q_floor = 0.0
cfg.w_max = a.w_max
# ★ SimConfig 생성 시점에 hjb_w_grid가 그때의 w_max(0.70) 기준으로 이미 만들어지므로,
#   w_max를 바꾸면 격자도 같은 방식(0~w_max 8점 균등)으로 재생성해야 실제 반영된다.
cfg.hjb_w_grid = tuple(np.linspace(0.0, a.w_max, a.w_grid_n))
cfg.pension_rho = a.pension_rho
if a.mu_annual > 0.0: cfg.mu_annual = a.mu_annual
if a.sigma_annual > 0.0: cfg.sigma_annual = a.sigma_annual
if a.bequest_kappa > 0.0:
    cfg.bequest_kappa = a.bequest_kappa
    cfg.bequest_gamma = 1.0
if a.labor_wan > 0.0 and a.labor_until > 55.0:
    cfg.labor_income_m = a.labor_wan / W0_WAN   # 월소득(정규화 단위)
    cfg.labor_until_age = a.labor_until
    print(f"[labor] 계속고용: 월 {a.labor_wan:.0f}만원 (={cfg.labor_income_m:.5f}) × 55~{a.labor_until-1:.0f}세")
if a.survival == "on":
    _probe = RetirementEnv(copy.deepcopy(cfg))
    _ldf = get_life_table_from_env(_probe)
    _qx = {int(r['age']): float(r['qx']) for _, r in _ldf.iterrows()}
    cfg.hjb_survival_px = np.array([(1.0-min(max(_qx.get(int(55+m//12), _qx[max(_qx)]),0.0),0.999))**(1.0/12.0) for m in range(420)])
    print('[survival] HJB 해에 생존가중 적용')


if a.rl_ckpt:
    from milevsky_timing_RL import load_actor, deterministic_action_batch
    rl_actor = load_actor(a.rl_ckpt)
    print(f"=== RL 전략 프로파일 (ckpt={a.rl_ckpt}, γ={a.gamma}) ===")
    def act(ob):
        q, w = deterministic_action_batch(rl_actor, np.array([ob]), 0.02, a.w_max)
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
# 소비부족 위험: 최저생활비(f_min, 연 0.072 → 월 0.006) 미달 지표
F_MIN_M = 0.072 / 12.0
DELTA_M = 0.9530 ** (1.0/12.0); U_SCALE = 0.0001
short_bridge=[]; short_any=[]; short_months=[]; eu_paths=[]
for i in range(n):
    env=RetirementEnv(copy.deepcopy(cfg)); ob=env.reset(seed=i); done=False
    sb=False; sa=False; sm=0; disc=1.0; u_sum=0.0
    while not done:
        age=55+env.t/12.0
        q,w=act(ob)
        ob2,rew,done,info=env.step(q=q,w=w)
        u_sum+=disc*float(info.get("u_eff",0.0))*U_SCALE; disc*=DELTA_M
        c=info.get("consumption",0.0)
        if c < F_MIN_M:
            sa=True; sm+=1
            if age < 65.0: sb=True
        ak=int(round(age/5.0)*5)
        if ak in rec:
            rec[ak]["q"].append(q*12*100)      # 월→연 근사 %
            rec[ak]["w"].append(w*100)
            rec[ak]["W"].append(info.get("W",0.0))
            rec[ak]["c"].append(c)
        ob=ob2
    short_bridge.append(sb); short_any.append(sa); short_months.append(sm); eu_paths.append(u_sum)

print(f"\n{'연령':>4} | {'인출률q(연%)':>12} | {'위험비중w(%)':>12} | {'평균자산W':>10} | {'생존자산>0비율':>12}")
for k in sorted(rec):
    if not rec[k]["q"]: continue
    q=np.mean(rec[k]["q"]); w=np.mean(rec[k]["w"]); W=np.mean(rec[k]["W"])
    alive=np.mean(np.array(rec[k]["W"])>1e-6)*100
    print(f"{k:>4} | {q:>12.1f} | {w:>12.1f} | {W:>10.3f} | {alive:>11.0f}%")
print(f"\n[소비부족 위험] 최저생활비(연 {0.072}) 미달 경로 비율: "
      f"브릿지(55~64세) {np.mean(short_bridge)*100:.1f}%  |  전 생애 {np.mean(short_any)*100:.1f}%  |  "
      f"평균 미달 개월수 {np.mean(short_months):.1f}")
print(f"[EU] median {np.median(eu_paths):.3f}  mean {np.mean(eu_paths):.3f}  (동일 seed 공통난수 — 시나리오 간 비교 가능)")
