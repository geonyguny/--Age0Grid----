# tests/test_es95.py
import numpy as np
from project.metrics.es import es95_wealth

def test_es95_ordering():
    rng = np.random.default_rng(0)
    base = rng.lognormal(0.0, 0.3, 10000)
    safer = base + 0.1
    assert es95_wealth(safer) > es95_wealth(base)

def test_es95_alpha_edge():
    x = np.array([0.1, 0.2, 0.3])
    v = es95_wealth(x, alpha=0.5)  # 하위 50% 평균
    assert 0.1 <= v <= 0.2
