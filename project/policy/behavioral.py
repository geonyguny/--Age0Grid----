# project/policy/behavioral.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

@dataclass
class BehavioralSpec:
    on: bool = False
    lambda_loss: float = 1.0   # 손실가중(λ>1이면 손실 가중)
    beta: float = 1.0          # present-bias (β-δ 중 β만, 0<β<=1)
    habit_phi: float = 0.0     # 습관효용 가중

def parse_behavioral_from_args(args) -> BehavioralSpec:
    on = str(getattr(args, "bh_on", "off")).lower() == "on"
    return BehavioralSpec(
        on=on,
        lambda_loss=float(getattr(args, "la_k", 1.0) or 1.0),
        beta=float(getattr(args, "beta", 1.0) or 1.0),
        habit_phi=float(getattr(args, "habit_phi", 0.0) or 0.0),
    )

def distort_utility(u: float, *, ref: float = 0.0, spec: BehavioralSpec) -> float:
    """
    아주 단순한 형태의 유틸리티 왜곡:
      - 손실(u<ref) 구간에서 λ를 곱해 페널티 강화
      - present-bias β는 '현재 효용'에 추상적으로 곱해주는 형태의 스칼라(간단 버전)
    """
    if not spec.on:
        return u
    v = u
    if u < ref:
        v = (u - ref) * spec.lambda_loss + ref
    v *= spec.beta
    return v

def habit_utility(u: float, prev_u: float, *, spec: BehavioralSpec) -> float:
    """간단 습관효용: u - φ*prev_u."""
    if not spec.on or spec.habit_phi == 0.0:
        return u
    return u - spec.habit_phi * prev_u

def describe(spec: BehavioralSpec) -> Dict[str, float | int | str | bool]:
    return {
        "bh_on": spec.on,
        "lambda_loss": spec.lambda_loss,
        "beta": spec.beta,
        "habit_phi": spec.habit_phi,
    }
