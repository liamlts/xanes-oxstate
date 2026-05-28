import json
from pathlib import Path

import numpy as np
import pandas as pd

from xanes_oxstate.eval.run import evaluate_element
from xanes_oxstate.model.dataset import XanesDataset


def _synthetic_records(n_per_class=20, classes=(2, 3, 4)):
    rng = np.random.default_rng(0)
    e0 = 6539.0
    recs = []
    for ci, c in enumerate(classes):
        for k in range(n_per_class):
            e = np.linspace(e0 - 30, e0 + 70, 400)
            shift = ci * 1.5
            step = 1.0 / (1.0 + np.exp(-(e - (e0 + shift)) / 1.0))
            recs.append({
                "mp_id": f"mp-{ci}-{k}",
                "element": "Mn",
                "ox_state": c,
                "formula": f"Mn_{ci}_{k}",
                "energies": e.tolist(),
                "intensities": (step + 0.01 * rng.standard_normal(400)).tolist(),
            })
    return recs


def _write_parquet_splits(records, processed_dir: Path):
    from xanes_oxstate.data.split import split_records
    splits = split_records(records, seed=42)
    for name, recs in splits.items():
        pd.DataFrame(recs).to_parquet(processed_dir / f"Mn_{name}.parquet")


def test_end_to_end_smoke(tmp_path):
    records = _synthetic_records()
    processed = tmp_path / "processed"
    ckpts = tmp_path / "ckpts"
    metrics = tmp_path / "metrics"
    processed.mkdir()

    _write_parquet_splits(records, processed)

    out = evaluate_element(
        "Mn",
        processed_dir=processed,
        ckpt_dir=ckpts,
        metrics_dir=metrics,
        seeds=(0, 1),
        epochs=3,
    )
    assert out["element"] == "Mn"
    assert 0 <= out["accuracy"]["cnn"] <= 1
    assert (metrics / "Mn.json").exists()
    assert (metrics / "Mn_cm.npy").exists()
    assert (metrics / "Mn_failures.parquet").exists()
    assert (metrics / "Mn_reliability.png").exists()
