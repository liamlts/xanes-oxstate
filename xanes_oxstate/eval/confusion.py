"""Row-normalized confusion + confused-pair extraction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConfusedPair:
    true: int
    pred: int
    rate: float
    count: int


def build_confusion(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return cm / row_sums


def find_confused_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    threshold: float = 0.20,
) -> list[ConfusedPair]:
    cm_norm = build_confusion(y_true, y_pred, n_classes)
    counts = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        counts[int(t), int(p)] += 1

    pairs: list[ConfusedPair] = []
    for t in range(n_classes):
        for p in range(n_classes):
            if t == p:
                continue
            if cm_norm[t, p] >= threshold:
                pairs.append(ConfusedPair(t, p, float(cm_norm[t, p]),
                                          int(counts[t, p])))
    pairs.sort(key=lambda x: -x.rate)
    return pairs
