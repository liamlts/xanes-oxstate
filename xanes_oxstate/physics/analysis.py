"""Confused-pair analysis: overlay, LR feature inspection, structural bins."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression


def mean_spectrum_by_class(
    spectra: list[np.ndarray],
    classes: list[int],
    energies: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, np.ndarray]]]:
    by_class: dict[int, list[np.ndarray]] = defaultdict(list)
    for s, c in zip(spectra, classes):
        by_class[int(c)].append(np.asarray(s))

    means: dict[int, np.ndarray] = {}
    envelopes: dict[int, dict[str, np.ndarray]] = {}
    for c, ys in by_class.items():
        arr = np.stack(ys)
        means[c] = arr.mean(axis=0)
        envelopes[c] = {
            "lo": np.percentile(arr, 5, axis=0),
            "hi": np.percentile(arr, 95, axis=0),
        }
    return means, envelopes


def pairwise_logistic_coefficients(
    X: np.ndarray, y: np.ndarray, C: float = 0.1
) -> np.ndarray:
    """L2-regularized binary LR; returns per-energy-point coefficients."""
    if set(map(int, np.unique(y))) != {0, 1}:
        raise ValueError("y must be binary (0/1) for pairwise LR")
    clf = LogisticRegression(C=C, max_iter=2000)
    clf.fit(X, y)
    return clf.coef_.ravel()


def group_failures_by_field(
    failures: list[dict],
    field: str,
    true_class: int,
    pred_class: int,
) -> dict:
    matched = [
        f for f in failures
        if f.get("true") == true_class and f.get("pred") == pred_class
        and field in f
    ]
    return dict(Counter(f[field] for f in matched))
