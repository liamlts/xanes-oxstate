"""Aggregate per-element metrics → headline figures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from xanes_oxstate.eval.plots import (
    confusion_small_multiples,
    per_element_accuracy_bar,
)


ELEMENTS = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"]


def main() -> None:
    metrics_dir = Path("metrics")
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)

    results, errs, matrices = {}, {}, {}
    for elem in ELEMENTS:
        path = metrics_dir / f"{elem}.json"
        if not path.exists():
            print(f"  skip {elem}: no metrics")
            continue
        m = json.loads(path.read_text())
        results[elem] = m["accuracy"]
        # Ensemble disagreement as a rough error bar
        errs[elem] = {"cnn": float(m.get("ensemble_disagreement", 0.0)) / 2}

        cm = np.load(metrics_dir / f"{elem}_cm.npy")
        labels = [m["class_to_ox_state"][str(i)] for i in range(cm.shape[0])]
        matrices[elem] = (cm, labels)

    if results:
        per_element_accuracy_bar(
            results, errs=errs,
            out_path=fig_dir / "per_element_accuracy.png",
        )
        per_element_accuracy_bar(
            results, errs=errs,
            out_path=fig_dir / "per_element_accuracy.pdf",
        )
    if matrices:
        confusion_small_multiples(
            matrices, n_cols=4,
            out_path=fig_dir / "confusions.png",
        )
        confusion_small_multiples(
            matrices, n_cols=4,
            out_path=fig_dir / "confusions.pdf",
        )


if __name__ == "__main__":
    main()
