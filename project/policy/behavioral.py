# project/policy/behavioral.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict
import math

def _f(x, d=0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return float(d)
    except Exception:
        return float(d)

@dataclass
class BehavioralSpec:
    """
    유틸리티/보상 계층에 적용되는 간단 행동편향 스펙.
    - lambda_loss: 손실 가중(λ>=0); u<ref 구간에서 패널티 강화
    - beta       : 현재편향(0<β<=1); t>=1 보상에 β 곱해 과소평가
    - habit_phi  : 습관효용(φ>=0); u_t ← u_t - φ*u_{t-1}
    - regret_rho : 후회민감도(ρ>=0); 논문 식(38) — 소비(c)가 기준소비(c*, 예: 4%룰
                   인출액)에 미달할 때 추가 불편(disutility) = -ρ*max(c*-c, 0) 부과.
                   [2026-07 최초 구현(시장수익률 기반 위험자산 추격매수)은 논문 본문의
                   Bell(1982)/Loomes-Sugden(1982) 후회이론(소비 미달 기준)과 맞지 않아
                   소비 관점으로 재구현함.
    - ambiguity_rho: 모호성회피 강도(ρamb>=0). 논문 식(36)/(44) — 강건제어(robust
                   control) 관점에서, 실현된 위험자산 충격이 모형가정(μ)과 크게
                   벗어날수록(=모형이 틀렸다는 게 사후적으로 드러날수록), 그리고
                   그 충격에 대한 노출(위험자산비중 w)이 클수록 추가 불편을 부과한다.
                   [2026-07] 기존 액션레이어(BiasConfig.ambiguity, w를 상시 구조적으로
                   축소)에서 가치함수 레벨 항으로 이전 — 논문 식(44)가 모호성회피를
                   손실회피·습관형성·후회와 동일하게 uB(ct)에 포함되는 항으로 명시.
    """
    on: bool = False
    lambda_loss: float = 0.0   # >=0
    beta: float = 1.0          # (0,1]
    habit_phi: float = 0.0     # >=0
    regret_rho: float = 0.0    # >=0
    ambiguity_rho: float = 0.0  # >=0

def parse_behavioral_from_args(args) -> BehavioralSpec:
    on = str(getattr(args, "bh_on", "off") or "off").lower() == "on"
    lam = max(0.0, _f(getattr(args, "bh_lambda_loss", getattr(args, "la_k", 0.0)), 0.0))
    beta = _f(getattr(args, "bh_beta", getattr(args, "beta", 1.0)), 1.0)
    if not (0.0 < beta <= 1.0):
        beta = 1.0
    habit_phi = max(0.0, _f(getattr(args, "bh_habit_phi", getattr(args, "habit_phi", 0.0)), 0.0))
    regret_rho = max(0.0, _f(getattr(args, "bh_regret_rho", 0.0), 0.0))
    ambiguity_rho = max(0.0, _f(getattr(args, "theta_ambiguity", 0.0), 0.0))
    return BehavioralSpec(on=on, lambda_loss=lam, beta=beta, habit_phi=habit_phi,
                           regret_rho=regret_rho, ambiguity_rho=ambiguity_rho)

def distort_utility(u: float, *, u_ref: float = 0.0, spec: BehavioralSpec) -> float:
    """
    논문 식(45): uB(ct) ← u(ct) - κ·[u(c_ref) - u(ct)]⁺  (κ=lambda_loss)
    현재 효용이 기준소비의 효용(u_ref)보다 낮을 때만 그 미달분에 비례해
    추가 불편(disutility)을 부과한다(Kahneman and Tversky 1979).

    [FIX 2026-07] 기존엔 u_ref가 0.0으로 고정되어 있어(항상 "소비=W0일 때의
    효용"이 기준), 논문이 말하는 "전기소비 또는 필수소비 수준"이라는 동적
    기준과 달랐다. 이제 u_ref는 호출부에서 실제 기준소비(c_ref, 보통
    전기소비 c_{t-1})의 CRRA 효용으로 매 시점 계산되어 전달된다.
    """
    if not spec.on or spec.lambda_loss <= 1e-16:
        return float(u)
    v = float(u)
    shortfall = max(float(u_ref) - v, 0.0)
    return v - float(spec.lambda_loss) * shortfall

def habit_utility(u: float, c_t: float, c_prev: float, *, spec: BehavioralSpec) -> float:
    """
    논문 식(33): U = Σ βt[u(ct) - θ(ct - c_{t-1})²]
    소비의 급격한 변화(제곱차분)에 벌점을 부과하여 습관형성/소비평탄화를 반영한다.

    [FIX 2026-07] 식(33)을 "소비 단위" 그대로 제곱하면(예: (0.01-0.008)²≈4e-6)
    CRRA 효용의 자연스러운 스케일(-1e4~-1e6)에 비해 지나치게 작아 사실상 어떤
    θ를 줘도 효과가 나타나지 않는다(후회항에서 발견한 것과 동일한 문제).
    "소비 단위 절대 제곱차분"이 아니라 "직전소비 대비 상대적(비율) 변화율의
    제곱"으로 바꾸고, 이를 현재 시점 효용 u의 크기에 비례시켜 두 항의 스케일이
    자연스럽게 맞도록 한다.
    """
    if not spec.on or spec.habit_phi <= 1e-16:
        return float(u)
    c_prev = float(c_prev)
    if c_prev <= 1e-12:
        return float(u)
    rel_change_sq = ((float(c_t) - c_prev) / c_prev) ** 2
    return float(u) - float(spec.habit_phi) * rel_change_sq * abs(float(u))

def ambiguity_utility(u: float, w: float, shock_dev: float, sigma: float, *, spec: BehavioralSpec) -> float:
    """
    논문 식(36)/(44): UR = min_P E_P[u(ct)] - δamb·D(P‖P0) (Hansen-Sargent 2001 강건제어)
    실현된 위험자산 충격이 모형가정(μ)에서 벗어난 정도(shock_dev = r_실현 - μ)가 클수록,
    그리고 그 충격에 노출된 정도(위험자산비중 w)가 클수록 "모형이 틀렸음이 사후적으로
    드러난 비용"을 부과한다: Φamb(t) = (w·z)², z = shock_dev/σ(표준화 충격).
    [2026-07] 기존 액션레이어(w를 상시 구조적으로 축소)에서 가치함수 레벨 항으로 이전.
    [FIX 2026-07] shock_dev를 수익률 단위 그대로 쓰면(월 표준편차가 이미 작은 값,
    예 0.06 근처) 제곱하면 더욱 작아져(예 (0.3*0.06)²≈3e-4) CRRA 효용 스케일에
    비해 무시할 수준이 된다(습관형성·후회에서 발견한 것과 동일한 문제). 표준편차로
    표준화한 충격(z-score, O(1) 스케일)을 쓰고, 이를 |u|에 비례시킨다.
    """
    if not spec.on or spec.ambiguity_rho <= 1e-16:
        return float(u)
    sigma = float(sigma)
    if sigma <= 1e-12:
        return float(u)
    z = float(shock_dev) / sigma
    phi_amb = (float(w) * z) ** 2
    return float(u) - float(spec.ambiguity_rho) * phi_amb * abs(float(u))


def regret_utility(u: float, c: float, c_ref: float, *, spec: BehavioralSpec) -> float:
    """
    논문 식(38): u_t ← u_t - ρ * max(c* - c, 0)
    소비(c)가 기준소비(c_ref, 예: 4%룰 인출액)에 미달할 때 추가 불편을 부과한다.
    (Bell 1982, Loomes and Sugden 1982 후회이론)

    [FIX 2026-07] 원래 식(38) 그대로 "소비 단위"의 절대적 미달분에 선형 페널티를
    매기면, CRRA 효용(절대소비가 작을수록 거듭제곱으로 발산)의 자연스러운 스케일
    (본 모델 파라미터 하에서 -1e4~-1e6 단위)에 비해 미달분(소비 단위, 보통
    0.001~0.02 수준)이 지나치게 작아 사실상 어떤 ρ를 줘도 효과가 전혀 나타나지
    않는 문제가 있었다. "소비 단위 절대량"이 아니라 "기준소비 대비 상대적
    미달비율"로 바꾸고, 이를 현재 시점 효용 u의 크기에 비례시켜 두 항의 스케일이
    자연스럽게 맞도록 한다(ρ가 "효용의 몇 %를 후회 페널티로 깎을지"를 직접
    통제하게 되어 해석도 더 명확해짐).
    """
    if not spec.on or spec.regret_rho <= 1e-16:
        return float(u)
    c_ref = float(c_ref)
    if c_ref <= 1e-12:
        return float(u)
    shortfall_frac = max(c_ref - float(c), 0.0) / c_ref  # 0~1 사이 상대적 미달비율
    return float(u) - float(spec.regret_rho) * shortfall_frac * abs(float(u))

def describe(spec: BehavioralSpec) -> Dict[str, float | int | str | bool]:
    d = asdict(spec)
    d["bh_on"] = d.pop("on")
    return d
