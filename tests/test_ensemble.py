import torch

from xanes_oxstate.model.dataset import XanesDataset
from xanes_oxstate.model.ensemble import train_ensemble, load_ensemble


def _records():
    import numpy as np
    rng = np.random.default_rng(0)
    e0 = 6539.0
    recs = []
    for ci, c in enumerate((2, 3, 4)):
        for k in range(20):
            e = np.linspace(e0 - 30, e0 + 70, 400)
            step = 1.0 / (1.0 + np.exp(-(e - (e0 + 1.5 * ci))))
            recs.append({
                "mp_id": f"mp-{ci}-{k}",
                "element": "Mn",
                "ox_state": c,
                "formula": f"Mn_{ci}_{k % 4}",
                "energies": e.tolist(),
                "intensities": (step + 0.01 * rng.standard_normal(400)).tolist(),
            })
    return recs


def test_train_ensemble_writes_checkpoints(tmp_path):
    train_ds = XanesDataset(_records())
    val_ds = XanesDataset(_records())
    paths = train_ensemble(
        train_ds, val_ds, ckpt_dir=tmp_path,
        seeds=(0, 1), epochs=3, batch_size=16,
    )
    assert len(paths) == 2
    for p in paths:
        assert p.exists()


def test_load_ensemble_returns_models(tmp_path):
    train_ds = XanesDataset(_records())
    val_ds = XanesDataset(_records())
    train_ensemble(
        train_ds, val_ds, ckpt_dir=tmp_path,
        seeds=(0, 1), epochs=3, batch_size=16,
    )
    models = load_ensemble(tmp_path, element="Mn")
    assert len(models) == 2
    x = torch.zeros(2, 1, 200)
    for m in models:
        assert m(x).shape == (2, train_ds.n_classes)
