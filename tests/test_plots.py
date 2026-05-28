import matplotlib
matplotlib.use("Agg")

from xanes_oxstate.eval.plots import per_element_accuracy_bar


def test_per_element_accuracy_bar_returns_figure(tmp_path):
    results = {
        "Mn": {"majority": 0.4, "gbdt": 0.7, "cnn": 0.85},
        "Fe": {"majority": 0.5, "gbdt": 0.75, "cnn": 0.88},
    }
    errs = {
        "Mn": {"cnn": 0.02},
        "Fe": {"cnn": 0.015},
    }
    out = tmp_path / "acc.png"
    fig = per_element_accuracy_bar(results, errs=errs, out_path=out)
    assert out.exists()
    assert fig.axes


import numpy as np
from xanes_oxstate.eval.plots import confusion_small_multiples


def test_confusion_small_multiples_writes_file(tmp_path):
    cms = {
        "Mn": (np.eye(3) * 0.9 + 0.05, [2, 3, 4]),
        "Fe": (np.eye(2) * 0.95 + 0.05, [2, 3]),
    }
    out = tmp_path / "cm.png"
    fig = confusion_small_multiples(cms, n_cols=2, out_path=out)
    assert out.exists()
    assert len(fig.axes) >= 2


import numpy as np
from xanes_oxstate.eval.plots import reliability_diagram


def test_reliability_diagram_writes_file(tmp_path):
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(alpha=[1, 1, 1], size=200)
    y = probs.argmax(axis=1)
    out = tmp_path / "rel.png"
    fig = reliability_diagram(probs, y, n_bins=10, out_path=out)
    assert out.exists()
    assert fig.axes
