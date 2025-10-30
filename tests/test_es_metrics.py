# tests/test_es_metrics.py
from __future__ import annotations
import numpy as np
from project.runner.run import _es_tail_mean, _cvar_loss_from_wealth

def test_es_tail_mean_uniform():
    rng = np.random.default_rng(0)
    xs = rng.uniform(0.0, 1.0, size=10000)
    es = _es_tail_mean(xs, alpha=0.95)  # mean of worst 5%
    # For U(0,1), bottom 5% average ~ 0.025
    assert abs(es - 0.025) < 0.01

def test_cvar_loss_from_wealth_simple():
    xs = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    F = 1.0
    es = _cvar_loss_from_wealth(xs, F_target=F, alpha=0.8)  # worst 20% of losses
    # losses: [1.0, 0.5, 0.0, 0.0, 0.0] -> top 20% is just [1.0]
    assert abs(es - 1.0) < 1e-6
