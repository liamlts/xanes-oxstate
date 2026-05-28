"""LightGBM wrapper over the hand-feature vector."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np


class GBDTBaseline:
    def __init__(self, n_classes: int, random_state: int = 0) -> None:
        self.n_classes = n_classes
        self.model = lgb.LGBMClassifier(
            objective="multiclass" if n_classes > 2 else "binary",
            num_class=n_classes if n_classes > 2 else 1,
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            verbose=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GBDTBaseline":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
