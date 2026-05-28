import numpy as np

from xanes_oxstate.physics.analysis import mean_spectrum_by_class


def test_mean_spectrum_per_class():
    energies = np.linspace(0, 1, 50)
    intens = [
        np.zeros(50), np.zeros(50),
        np.ones(50), np.ones(50),
    ]
    classes = [0, 0, 1, 1]
    means, envelopes = mean_spectrum_by_class(intens, classes, energies)
    assert means[0].shape == (50,)
    assert means[1].shape == (50,)
    assert np.allclose(means[0], 0)
    assert np.allclose(means[1], 1)
    assert envelopes[0]["lo"].shape == (50,)
