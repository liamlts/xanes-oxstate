import numpy as np
import pytest

from xanes_oxstate.spectrum import Spectrum


def test_spectrum_holds_required_fields():
    s = Spectrum(
        energy=np.linspace(6500, 6600, 100),
        intensity=np.zeros(100),
        element="Mn",
        ox_state=2,
        formula="MnO",
        mp_id="mp-19006",
    )
    assert s.element == "Mn"
    assert s.ox_state == 2
    assert s.energy.shape == (100,)
    assert s.intensity.shape == (100,)


def test_spectrum_requires_matching_lengths():
    with pytest.raises(ValueError):
        Spectrum(
            energy=np.zeros(100),
            intensity=np.zeros(99),
            element="Mn",
            ox_state=2,
            formula="MnO",
            mp_id="mp-19006",
        )


def test_spectrum_is_frozen():
    s = Spectrum(
        energy=np.zeros(10),
        intensity=np.zeros(10),
        element="Fe",
        ox_state=3,
        formula="Fe2O3",
        mp_id="mp-19770",
    )
    with pytest.raises((AttributeError, TypeError)):
        s.element = "Mn"


def test_fake_xanes_fixture(mn_spectrum):
    assert mn_spectrum.element == "Mn"
    assert mn_spectrum.energy.shape == mn_spectrum.intensity.shape == (120,)


def test_make_dataset_fixture(mn_dataset):
    assert len(mn_dataset) == 60
    assert {s.ox_state for s in mn_dataset} == {2, 3, 4}
