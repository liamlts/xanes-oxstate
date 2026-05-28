import json
from pathlib import Path

import pytest

from xanes_oxstate.data.split import split_records, LeakageError


def _records(n_per_formula=2, n_formulas=12, classes=(2, 3, 4)):
    recs = []
    idx = 0
    for c in classes:
        for f in range(n_formulas):
            for k in range(n_per_formula):
                recs.append({
                    "mp_id": f"mp-{idx}",
                    "element": "Mn",
                    "ox_state": c,
                    "formula": f"MnO_{c}_{f}",
                })
                idx += 1
    return recs


def test_split_is_formula_disjoint():
    recs = _records()
    splits = split_records(recs, seed=42, ratios=(0.7, 0.15, 0.15))
    train_f = {r["formula"] for r in splits["train"]}
    val_f = {r["formula"] for r in splits["val"]}
    test_f = {r["formula"] for r in splits["test"]}
    assert train_f.isdisjoint(val_f)
    assert train_f.isdisjoint(test_f)
    assert val_f.isdisjoint(test_f)


def test_split_keeps_all_classes_in_train():
    recs = _records()
    splits = split_records(recs, seed=42, ratios=(0.7, 0.15, 0.15))
    train_classes = {r["ox_state"] for r in splits["train"]}
    assert train_classes == {2, 3, 4}


def test_split_ratios_approx():
    recs = _records(n_per_formula=2, n_formulas=20, classes=(2, 3, 4))
    splits = split_records(recs, seed=42, ratios=(0.7, 0.15, 0.15))
    total = sum(len(v) for v in splits.values())
    assert total == len(recs)
    assert 0.6 < len(splits["train"]) / total < 0.8
    assert 0.05 < len(splits["val"]) / total < 0.25
    assert 0.05 < len(splits["test"]) / total < 0.25


def test_split_raises_on_leakage_post_check():
    recs = _records()
    splits = split_records(recs, seed=42, ratios=(0.7, 0.15, 0.15))
    # Inject a leak.
    splits["val"].append(splits["train"][0])
    with pytest.raises(LeakageError):
        from xanes_oxstate.data.split import assert_no_leakage
        assert_no_leakage(splits)
