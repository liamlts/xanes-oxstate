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


def confusion_small_multiples(
    matrices: dict[str, tuple[np.ndarray, list[int]]],
    n_cols: int = 4,
    highlight_threshold: float = 0.20,
    out_path: Path | None = None,
):
    elements = list(matrices.keys())
    n_rows = (len(elements) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=False
    )

    for k, elem in enumerate(elements):
        cm, labels = matrices[elem]
        r, c = divmod(k, n_cols)
        ax = axes[r][c]
        im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:.2f}",
                        ha="center", va="center",
                        color="white" if cm[i, j] < 0.5 else "black",
                        fontsize=7)
                if i != j and cm[i, j] >= highlight_threshold:
                    ax.add_patch(
                        plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor="red",
                                       linewidth=1.5)
                    )
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(elem)

    # Hide unused axes
    for k in range(len(elements), n_rows * n_cols):
        r, c = divmod(k, n_cols)
        axes[r][c].axis("off")

    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)
    return fig
