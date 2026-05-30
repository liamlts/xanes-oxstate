"""Gradient-boosting baseline over the hand-feature vector.

Uses sklearn's HistGradientBoostingClassifier rather than LightGBM to avoid
the macOS torch + lightgbm libomp dual-load segfault.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


class GBDTBaseline:
    def __init__(self, n_classes: int, random_state: int = 0) -> None:
        self.n_classes = n_classes
        self.model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GBDTBaseline":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
