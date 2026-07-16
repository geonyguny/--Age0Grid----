# -*- coding: utf-8 -*-
"""규칙기반 인출전략 벤치마크 vs HJB 최적 — 동일 환경·동일 공통난수 비교.
전략: ①4%룰(정액) ②정률4% ③TDF 하락형 글라이드 ④TDF글라이드+HJB인출(배분효과 분리) ⑤HJB 최적
지표: median EU(θ=0), 브릿지 소비부족(경로비율·개월), W(90세)"""
import copy
import numpy as np
from milevsky_timing_RL import make_base_cfg, get_life_table_from_env
from milevsky_timing_analysis import solve_theta0_policy
from project.env.retirement_env import RetirementEnv

GAMMA = 3.0
N = 500
N_MONTHS = 420
U_SCALE = 1e-4
DELTA_M = 0.9530 ** (1.0/12.0)
F_MIN_M = 0.072/12.0

cfg = make_base_cfg(crra_gamma=GAMMA, asset="KR")
cfg.floor_on = False; cfg.f_min_real = 0.0; cfg.q_floor = 0.0

# HJB 최적 정책 (인출상한 연~12%)
Pi_w, Pi_q, W_grid = solve_theta0_policy(cfg, hjb_W_grid=150, hjb_q_max_mult=3.0)
T = Pi_w.shape[0]
def hjb_act(t_norm, W):
    t = min(int(round(t_norm*T)), T-1)
    i = min(int(np.argmin(np.abs(W_grid-W))), Pi_w.shape[1]-1)
    return float(Pi_q[t,i]), float(Pi_w[t,i])

def tdf_w(age):
    """시판 TDF 인출기 글라이드 근사: 55세 50% → 80세+ 20% 선형 하락"""
    return float(np.clip(0.50 - 0.30*(age-55)/25.0, 0.20, 0.50))

Q4_M = 1.0 - (1.0-0.04)**(1.0/12.0)   # 연 4% 정률의 월환산

def actor(kind, t_norm, W, age):
    if kind == "rule_4pct_fixed":      # 정액 4%룰: c=0.04*W0/년 실질, w=60%
        c_m = 0.04/12.0
        q = min(1.0, c_m/max(W,1e-9))
        return q, 0.60
    if kind == "rule_4pct_prop":       # 정률 4%: q=연4%, w=60%
        return Q4_M, 0.60
    if kind == "rule_tdf":             # 정률 4% + TDF 하락형 글라이드
        return Q4_M, tdf_w(age)
    if kind == "hjb_q_tdf_w":          # HJB 인출 + TDF 글라이드 (배분효과 분리)
        q,_ = hjb_act(t_norm, W); return q, tdf_w(age)
    if kind == "hjb":                  # HJB 최적
        return hjb_act(t_norm, W)
    raise ValueError(kind)

def evaluate(kind):
    u=np.zeros(N); sb=0; sm=[]; W90=[]
    for i in range(N):
        env=RetirementEnv(copy.deepcopy(cfg)); ob=env.reset(seed=i)
        disc=1.0; done=False; m_short=0; hit=False
        while not done:
            age=55+env.t/12.0
            q,w=actor(kind, ob[0], ob[1], age)
            ob,rew,done,info=env.step(q=q,w=w)
            c=info.get("consumption",0.0)
            if c<F_MIN_M and age<65.0: m_short+=1; hit=True
            u[i]+=disc*float(info.get("u_eff",0.0))*U_SCALE
            disc*=DELTA_M
        sb+=hit; sm.append(m_short); W90.append(env.W)
    us=np.sort(u); k=max(1,int(N*0.1)); trim=us[k:-k]
    return np.median(u), float(np.mean(trim)), float(np.mean(u)), sb/N*100, np.mean(sm), np.mean(W90)

print(f"{'전략':<22} | {'median':>9} | {'절사평균10%':>10} | {'평균':>11} | {'브릿지미달%':>9} | {'미달개월':>8} | {'W(90세)':>8}")
for kind,label in [("rule_4pct_fixed","4%룰(정액,w60)"),
                   ("rule_4pct_prop","정률4%(w60)"),
                   ("rule_tdf","정률4%+TDF하락형"),
                   ("hjb_q_tdf_w","HJB인출+TDF하락형"),
                   ("hjb","HJB 최적(상승형)")]:
    med,tr,mean,b,m,w90 = evaluate(kind)
    print(f"{label:<22} | {med:>9.2f} | {tr:>10.2f} | {mean:>11.2f} | {b:>8.1f}% | {m:>8.1f} | {w90:>8.3f}")
