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
    """
    on: bool = False
    lambda_loss: float = 0.0   # >=0
    beta: float = 1.0          # (0,1]
    habit_phi: float = 0.0     # >=0

def parse_behavioral_from_args(args) -> BehavioralSpec:
    on = str(getattr(args, "bh_on", "off") or "off").lower() == "on"
    lam = max(0.0, _f(getattr(args, "bh_lambda_loss", getattr(args, "la_k", 0.0)), 0.0))
    beta = _f(getattr(args, "bh_beta", getattr(args, "beta", 1.0)), 1.0)
    if not (0.0 < beta <= 1.0):
        beta = 1.0
    habit_phi = max(0.0, _f(getattr(args, "bh_habit_phi", getattr(args, "habit_phi", 0.0)), 0.0))
    return BehavioralSpec(on=on, lambda_loss=lam, beta=beta, habit_phi=habit_phi)

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

def describe(spec: BehavioralSpec) -> Dict[str, float | int | str | bool]:
    d = asdict(spec)
    d["bh_on"] = d.pop("on")
    return d
