"""Predict the most-common training class. Sanity floor."""
from __future__ import annotations

from collections import Counter

import numpy as np


class MajorityClassifier:
    def __init__(self) -> None:
        self.majority_class: int | None = None

    def fit(self, y: np.ndarray) -> "MajorityClassifier":
        self.majority_class = int(Counter(map(int, y)).most_common(1)[0][0])
        return self

    def predict(self, X) -> list[int]:
        if self.majority_class is None:
            raise RuntimeError("call .fit first")
        n = len(X)
        return [self.majority_class] * n

    def score(self, X, y_true: np.ndarray) -> float:
        return float((np.asarray(self.predict(X)) == y_true).mean())
