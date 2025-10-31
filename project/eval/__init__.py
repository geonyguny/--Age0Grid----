# project/eval/__init__.py
"""
Backward-compat shim: legacy imports like
  from project.eval import evaluate, save_metrics_autocsv
should keep working after we moved implementations to project/evaluation.py
"""

# 평가/유틸 구현은 project/evaluation.py로 이전됨
from ..evaluation import (  # type: ignore
    evaluate,
    save_metrics_autocsv,
    # 편의상 자주 쓰는 유틸도 함께 re-export
    path_expected_utility,
    monthly_discount_from_annual,
    crra_u,
)

__all__ = [
    "evaluate",
    "save_metrics_autocsv",
    "path_expected_utility",
    "monthly_discount_from_annual",
    "crra_u",
]
