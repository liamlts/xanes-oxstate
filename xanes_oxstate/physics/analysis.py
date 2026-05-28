"""Confused-pair analysis: overlay, LR feature inspection, structural bins."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


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
