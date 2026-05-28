import numpy as np

from xanes_oxstate.eval.calibration import (
    fit_temperature,
    apply_temperature,
    expected_calibration_error,
)


def _miscalibrated_logits(n=1000, n_classes=3, seed=0, scale=5.0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, n)
    logits = rng.standard_normal((n, n_classes))
    # Push correct class up.
    logits[np.arange(n), y] += 2.0
    return scale * logits, y


def test_fit_temperature_returns_scalar_above_zero():
    logits, y = _miscalibrated_logits()
    T = fit_temperature(logits, y)
    assert T > 0


def test_temperature_lowers_ece_on_overconfident_logits():
    logits, y = _miscalibrated_logits(scale=5.0)
    probs = _softmax(logits)
    ece_before = expected_calibration_error(probs, y)
    T = fit_temperature(logits, y)
    probs_after = _softmax(apply_temperature(logits, T))
    ece_after = expected_calibration_error(probs_after, y)
    assert ece_after < ece_before


def _softmax(x):
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
