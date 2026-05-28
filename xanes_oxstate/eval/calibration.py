"""Temperature scaling and ECE (Guo et al. 2017)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def _log_softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def _nll(T: float, logits: np.ndarray, y: np.ndarray) -> float:
    log_p = _log_softmax(logits / T)
    return float(-log_p[np.arange(y.size), y].mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    res = minimize_scalar(_nll, args=(logits, y), bounds=(0.05, 10.0),
                          method="bounded")
    return float(res.x)


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    return logits / T


def expected_calibration_error(
    probs: np.ndarray, y: np.ndarray, n_bins: int = 15
) -> float:
    conf = probs.max(axis=-1)
    pred = probs.argmax(axis=-1)
    correct = (pred == y).astype(np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = probs.shape[0]
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if not mask.any():
            continue
        bin_acc = correct[mask].mean()
        bin_conf = conf[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)
