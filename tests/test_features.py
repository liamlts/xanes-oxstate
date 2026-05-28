import numpy as np

from xanes_oxstate.baselines.features import extract_features


def _norm_step(e0=6539.0, n=200, shift=0.0, height=1.0):
    e = np.linspace(e0 - 10, e0 + 40, n)
    step = height / (1.0 + np.exp(-(e - (e0 + shift))))
    bump = 0.3 * np.exp(-((e - (e0 + 2)) ** 2) / 4.0)
    return e, step + bump


def test_feature_vector_length():
    e, y = _norm_step()
    f = extract_features(e, y, e0=6539.0)
    assert len(f) == 6


def test_pre_edge_area_is_nonzero_when_bump_present():
    e, y = _norm_step()
    f = extract_features(e, y, e0=6539.0)
    assert f[0] > 0


def test_edge_position_tracks_shift():
    e, y1 = _norm_step(shift=0.0)
    _, y2 = _norm_step(shift=2.0)
    f1 = extract_features(e, y1, e0=6539.0)
    f2 = extract_features(e, y2, e0=6539.0)
    assert f2[2] > f1[2]
