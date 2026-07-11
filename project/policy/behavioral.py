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
                   [2026-07] 최초 구현(시장수익률 기반 위험자산 추격매수)은 논문 본문의
                   Bell(1982)/Loomes-Sugden(1982) 후회이론(소비 미달 기준)과 맞지 않아
                   소비 관점으로 재구현함.
    """
    on: bool = False
    lambda_loss: float = 0.0   # >=0
    beta: float = 1.0          # (0,1]
    habit_phi: float = 0.0     # >=0
    regret_rho: float = 0.0    # >=0

def parse_behavioral_from_args(args) -> BehavioralSpec:
    on = str(getattr(args, "bh_on", "off") or "off").lower() == "on"
    lam = max(0.0, _f(getattr(args, "bh_lambda_loss", getattr(args, "la_k", 0.0)), 0.0))
    beta = _f(getattr(args, "bh_beta", getattr(args, "beta", 1.0)), 1.0)
    if not (0.0 < beta <= 1.0):
        beta = 1.0
    habit_phi = max(0.0, _f(getattr(args, "bh_habit_phi", getattr(args, "habit_phi", 0.0)), 0.0))
    regret_rho = max(0.0, _f(getattr(args, "bh_regret_rho", 0.0), 0.0))
    return BehavioralSpec(on=on, lambda_loss=lam, beta=beta, habit_phi=habit_phi, regret_rho=regret_rho)

def distort_utility(u: float, *, ref: float = 0.0, spec: BehavioralSpec) -> float:
    """
    u<ref 영역에서 (u-ref)*λ + ref 적용. (λ>=0)
    """
    if not spec.on:
        return float(u)
    v = float(u)
    if v < ref and spec.lambda_loss > 0.0:
        v = (v - ref) * spec.lambda_loss + ref
    return v

def habit_utility(u: float, prev_u: float, *, spec: BehavioralSpec) -> float:
    """
    u_t ← u_t - φ * u_{t-1}  (φ>=0)
    """
    if not spec.on or spec.habit_phi <= 1e-16:
        return float(u)
    return float(u) - float(spec.habit_phi) * float(prev_u)

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
