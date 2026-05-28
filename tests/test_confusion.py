import numpy as np

from xanes_oxstate.eval.confusion import build_confusion, find_confused_pairs


def test_build_confusion_row_normalized():
    y_true = np.array([0, 0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 0, 1, 1, 2])
    cm = build_confusion(y_true, y_pred, n_classes=3)
    # Row-normalized: each row sums to 1
    assert np.allclose(cm.sum(axis=1), 1.0)
    assert cm[0, 0] == 2 / 3
    assert cm[0, 1] == 1 / 3


def test_find_confused_pairs_above_threshold():
    y_true = np.array([0] * 10 + [1] * 10)
    y_pred = np.array([0] * 7 + [1] * 3 + [1] * 8 + [0] * 2)
    pairs = find_confused_pairs(y_true, y_pred, n_classes=2, threshold=0.2)
    # Class 0 → predicted 1 = 30%, class 1 → predicted 0 = 20%
    assert (0, 1) in [(p.true, p.pred) for p in pairs]
    assert (1, 0) in [(p.true, p.pred) for p in pairs]
