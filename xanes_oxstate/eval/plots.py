"""Headline figures for the README."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def per_element_accuracy_bar(
    results: dict[str, dict[str, float]],
    errs: dict[str, dict[str, float]] | None = None,
    out_path: Path | None = None,
):
    elements = list(results.keys())
    estimators = ["majority", "gbdt", "cnn"]
    x = np.arange(len(elements))
    w = 0.25

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, est in enumerate(estimators):
        vals = [results[e].get(est, 0.0) for e in elements]
        yerr = (
            [errs.get(e, {}).get(est, 0.0) for e in elements]
            if errs else None
        )
        ax.bar(x + (i - 1) * w, vals, w, yerr=yerr, label=est, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(elements)
    ax.axhline(0.75, color="grey", linestyle="--", linewidth=0.8)
    ax.axhline(0.85, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("top-1 accuracy")
    ax.set_title("Per-element accuracy: majority vs. GBDT vs. CNN ensemble")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)
    return fig
