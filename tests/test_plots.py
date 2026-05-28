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
