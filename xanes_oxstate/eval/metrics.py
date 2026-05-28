"""Classification metrics for the XANES oxidation-state task."""
from __future__ import annotations

import numpy as np


def top1_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> list[float]:
    out = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fp == 0 or tp + fn == 0:
            out.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        out.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
    return out


def ensemble_disagreement(preds: np.ndarray) -> float:
    """preds: [n_models, n_samples] integer class indices."""
    n_models, n_samples = preds.shape
    n_unique = np.array(
        [len(np.unique(preds[:, i])) for i in range(n_samples)]
    )
    # Normalize to [0, 1]: 1 model agreement -> 0; n_models distinct -> 1.
    return float((n_unique - 1).mean() / (n_models - 1))
