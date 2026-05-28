import pytest
from tests.fixtures.synthetic_spectra import fake_xanes, make_dataset


@pytest.fixture
def mn_spectrum():
    return fake_xanes(element="Mn", ox_state=2)


@pytest.fixture
def mn_dataset():
    return make_dataset(n_per_class=20, classes=(2, 3, 4), element="Mn")
