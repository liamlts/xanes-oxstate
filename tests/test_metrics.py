import numpy as np

from xanes_oxstate.eval.metrics import (
    top1_accuracy,
    per_class_f1,
    ensemble_disagreement,
)


def test_top1_accuracy():
    y_true = np.array([0, 1, 2, 2, 0])
    y_pred = np.array([0, 1, 2, 1, 0])
    assert top1_accuracy(y_true, y_pred) == 0.8


def test_per_class_f1_returns_one_per_class():
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 2])
    f1 = per_class_f1(y_true, y_pred, n_classes=3)
    assert len(f1) == 3
    assert all(0.0 <= v <= 1.0 for v in f1)


def test_ensemble_disagreement_is_zero_for_identical_preds():
    preds = np.stack([np.array([0, 1, 2])] * 3)
    assert ensemble_disagreement(preds) == 0.0


def test_ensemble_disagreement_is_one_for_max_disagreement():
    preds = np.array([[0, 0], [1, 1], [2, 2]])
    # Every sample has 3 distinct predictions across 3 models.
    assert ensemble_disagreement(preds) == 1.0
