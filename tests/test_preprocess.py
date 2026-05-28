import numpy as np
import pytest

from xanes_oxstate.data.preprocess import (
    resample_to_grid,
    edge_jump_normalize,
    detect_e0,
    preprocess,
)


def _step_spectrum(e0=6539.0, n=400):
    e = np.linspace(e0 - 30, e0 + 70, n)
    pre = 0.05 * (e - (e0 - 30))
    step = 1.0 / (1.0 + np.exp(-(e - e0) / 1.0))
    intensity = pre + step
    return e, intensity


def test_resample_returns_fixed_grid():
    e, y = _step_spectrum()
    grid, y_r = resample_to_grid(e, y, e0=6539.0, n_points=200,
                                 lo_eV=-10, hi_eV=40)
    assert grid.shape == (200,)
    assert y_r.shape == (200,)
    assert np.isclose(grid[0], 6539.0 - 10)
    assert np.isclose(grid[-1], 6539.0 + 40)


def test_resample_rejects_excessive_extrapolation():
    e = np.linspace(6535, 6545, 50)  # too narrow
    y = np.linspace(0, 1, 50)
    with pytest.raises(ValueError):
        resample_to_grid(e, y, e0=6539.0, n_points=200, lo_eV=-10, hi_eV=40,
                        max_extrap_eV=5)


def test_edge_jump_normalization_yields_unit_jump():
    e, y = _step_spectrum()
    y_n = edge_jump_normalize(e, y, e0=6539.0)
    pre_mask = (e >= 6539.0 - 10) & (e <= 6539.0 - 5)
    post_mask = (e >= 6539.0 + 30) & (e <= 6539.0 + 40)
    assert abs(y_n[pre_mask].mean()) < 0.05
    assert abs(y_n[post_mask].mean() - 1.0) < 0.05


def test_detect_e0_from_inflection():
    e, y = _step_spectrum(e0=6539.0)
    assert abs(detect_e0(e, y) - 6539.0) < 0.5


def test_preprocess_end_to_end():
    from tests.fixtures.synthetic_spectra import fake_xanes

    s = fake_xanes(element="Mn", ox_state=2, e0=6539.0)
    grid, y = preprocess(s.energy, s.intensity, e0=6539.0)
    assert grid.shape == (200,)
    assert y.shape == (200,)
    pre_mask = (grid >= 6539.0 - 10) & (grid <= 6539.0 - 5)
    post_mask = (grid >= 6539.0 + 30) & (grid <= 6539.0 + 40)
    assert abs(y[pre_mask].mean()) < 0.1
    assert abs(y[post_mask].mean() - 1.0) < 0.2
