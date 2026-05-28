import numpy as np

from xanes_oxstate.physics.analysis import (
    pairwise_logistic_coefficients,
)


def test_lr_finds_discriminative_region():
    rng = np.random.default_rng(0)
    n = 200
    grid = np.linspace(0, 1, 50)
    # Class A: peak at idx 10. Class B: peak at idx 40.
    X_a = np.zeros((n, 50))
    X_a[:, 10] = 1.0
    X_a += 0.05 * rng.standard_normal(X_a.shape)
    X_b = np.zeros((n, 50))
    X_b[:, 40] = 1.0
    X_b += 0.05 * rng.standard_normal(X_b.shape)
    X = np.vstack([X_a, X_b])
    y = np.array([0] * n + [1] * n)

    coefs = pairwise_logistic_coefficients(X, y)
    # The discriminative region should be near idx 10 (neg) and 40 (pos).
    assert coefs.shape == (50,)
    assert coefs[40] > np.abs(coefs[20]) * 5
    assert coefs[10] < -np.abs(coefs[20]) * 5
