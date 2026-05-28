import numpy as np
import torch

from xanes_oxstate.model.dataset import XanesDataset


def _records(n=30):
    rng = np.random.default_rng(0)
    e0 = 6539.0
    recs = []
    for i in range(n):
        e = np.linspace(e0 - 30, e0 + 70, 400)
        step = 1.0 / (1.0 + np.exp(-(e - e0)))
        recs.append({
            "mp_id": f"mp-{i}",
            "element": "Mn",
            "ox_state": (i % 3) + 2,
            "formula": f"Mn_{i % 5}",
            "energies": e.tolist(),
            "intensities": (step + 0.01 * rng.standard_normal(400)).tolist(),
        })
    return recs


def test_dataset_returns_tensor_and_label():
    ds = XanesDataset(_records())
    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape == (1, 200)
    assert isinstance(y, int)


def test_dataset_class_index_is_dense():
    ds = XanesDataset(_records())
    labels = sorted({int(ds[i][1]) for i in range(len(ds))})
    assert labels == list(range(len(labels)))


def test_dataset_exposes_ox_state_mapping():
    ds = XanesDataset(_records())
    assert ds.class_to_ox_state[0] in {2, 3, 4}
    inverse = {v: k for k, v in ds.class_to_ox_state.items()}
    assert ds[0][1] == inverse[ds.records[0]["ox_state"]]
