import numpy as np

from xanes_oxstate.baselines.majority import MajorityClassifier


def test_majority_predicts_most_common():
    y_train = np.array([0, 0, 0, 1, 2])
    clf = MajorityClassifier().fit(y_train)
    assert clf.predict(np.zeros(4)) == [0, 0, 0, 0]
    assert clf.majority_class == 0


def test_majority_score_matches_proportion():
    y_train = np.array([0, 0, 1, 1, 1])
    clf = MajorityClassifier().fit(y_train)
    acc = clf.score(np.zeros(5), np.array([1, 1, 1, 0, 0]))
    assert acc == 0.6
