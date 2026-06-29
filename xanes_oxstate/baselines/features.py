"""Hand-engineered features for the GBDT baseline.

Six features per normalized spectrum:
    0: pre-edge area               (-10 to -1 eV)
    1: pre-edge peak height        (max in [-10, -1])
    2: edge inflection position    (eV relative to E0)
    3: white-line height           (max in [0, 10])
    4: white-line position         (eV relative to E0)
    5: post-edge slope             (linear fit on [20, 40])
"""
from __future__ import annotations

import numpy as np

# np.trapezoid was added in NumPy 2.0 and np.trapz was removed in NumPy 2.x.
# Pick lazily so np.trapz is only referenced on NumPy 1.x where it still exists
# (getattr(np, "trapezoid", np.trapz) would eagerly evaluate np.trapz and raise
# AttributeError on NumPy 2.x before the fallback is needed).
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def extract_features(
    energy: np.ndarray, intensity: np.ndarray, e0: float
) -> np.ndarray:
    rel = energy - e0
    out = np.zeros(6, dtype=np.float64)

    pre_mask = (rel >= -10) & (rel <= -1)
    if pre_mask.any():
        out[0] = float(_trapezoid(intensity[pre_mask], rel[pre_mask]))
        out[1] = float(intensity[pre_mask].max())

    edge_mask = (rel >= -5) & (rel <= 10)
    if edge_mask.sum() >= 3:
        dy = np.gradient(intensity[edge_mask], rel[edge_mask])
        out[2] = float(rel[edge_mask][np.argmax(dy)])

    wl_mask = (rel >= 0) & (rel <= 10)
    if wl_mask.any():
        out[3] = float(intensity[wl_mask].max())
        out[4] = float(rel[wl_mask][np.argmax(intensity[wl_mask])])

    post_mask = (rel >= 20) & (rel <= 40)
    if post_mask.sum() >= 2:
        a, _ = np.polyfit(rel[post_mask], intensity[post_mask], 1)
        out[5] = float(a)

    return out
