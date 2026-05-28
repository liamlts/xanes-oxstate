import numpy as np

from xanes_oxstate.baselines.gbdt import GBDTBaseline


def test_gbdt_fits_and_predicts():
    rng = np.random.default_rng(0)
    n_per = 100
    X0 = rng.normal(loc=0, scale=0.3, size=(n_per, 6))
    X1 = rng.normal(loc=2, scale=0.3, size=(n_per, 6))
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per + [1] * n_per)

    clf = GBDTBaseline(n_classes=2, random_state=0).fit(X, y)
    acc = (clf.predict(X) == y).mean()
    assert acc > 0.9
