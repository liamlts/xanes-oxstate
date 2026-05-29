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


def test_dataset_cache_populated_when_cache_true():
    import numpy as np
    rng = np.random.default_rng(0)
    e0 = 6539.0
    recs = []
    for i in range(5):
        e = np.linspace(e0 - 30, e0 + 70, 400)
        step = 1.0 / (1.0 + np.exp(-(e - e0)))
        recs.append({
            "mp_id": f"mp-{i}", "element": "Mn", "ox_state": 2,
            "formula": f"Mn_{i}",
            "energies": e.tolist(),
            "intensities": (step + 0.01 * rng.standard_normal(400)).tolist(),
        })
    ds = XanesDataset(recs, cache=True)
    assert hasattr(ds, "_cache")
    assert len(ds._cache) == 5
    # Verify cache is consistent with __getitem__
    x0, y0 = ds[0]
    assert x0.shape == (1, 200)


def test_dataset_labels_property():
    recs = [
        {"mp_id": "a", "element": "Mn", "ox_state": 2, "formula": "MnO",
         "energies": [0,1,2], "intensities": [0,1,2]},
        {"mp_id": "b", "element": "Mn", "ox_state": 3, "formula": "MnO2",
         "energies": [0,1,2], "intensities": [0,1,2]},
    ]
    ds = XanesDataset(recs, cache=False)
    assert ds.labels == [0, 1]


def test_dataset_augmentation_changes_outputs():
    import numpy as np
    from xanes_oxstate.model.dataset import XanesDataset
    rng = np.random.default_rng(0)
    e0 = 6539.0
    recs = []
    for i in range(3):
        e = np.linspace(e0 - 30, e0 + 70, 400)
        step = 1.0 / (1.0 + np.exp(-(e - e0)))
        recs.append({
            "mp_id": f"mp-{i}", "element": "Mn", "ox_state": 2,
            "formula": f"Mn_{i}",
            "energies": e.tolist(),
            "intensities": (step + 0.01 * rng.standard_normal(400)).tolist(),
        })
    ds = XanesDataset(recs, cache=True, augment=True, roll_max=5)
    seen = set()
    torch.random.manual_seed(0)
    for _ in range(20):
        x, _ = ds[0]
        seen.add(int(x.argmax().item()))
    # With 20 random rolls in [-5, 5] we should see at least 2 distinct argmax positions
    assert len(seen) >= 2, f"augmentation didn't vary output (saw {seen})"
