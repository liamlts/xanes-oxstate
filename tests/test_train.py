import pandas as pd

from xanes_oxstate.model.dataset import XanesDataset
from xanes_oxstate.model.train import train_one_seed


def _records(n_per_class=15, classes=(2, 3, 4)):
    import numpy as np
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
                "formula": f"Mn_{ci}_{k % 4}",
                "energies": e.tolist(),
                "intensities": (step + 0.01 * rng.standard_normal(400)).tolist(),
            })
    return recs


def test_training_loss_decreases(tmp_path):
    train_recs = _records()
    val_recs = _records(n_per_class=8)
    train_ds = XanesDataset(train_recs)
    val_ds = XanesDataset(val_recs)

    out = train_one_seed(
        train_ds, val_ds,
        epochs=5, batch_size=16, lr=1e-3, seed=0,
        ckpt_path=tmp_path / "ckpt.pt",
    )
    assert out.history["train_loss"][-1] < out.history["train_loss"][0]
    assert (tmp_path / "ckpt.pt").exists()
