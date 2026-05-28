import torch

from xanes_oxstate.model.cnn import OxStateCNN


def test_forward_shape():
    model = OxStateCNN(n_classes=5)
    x = torch.zeros(4, 1, 200)
    out = model(x)
    assert out.shape == (4, 5)


def test_param_count_under_200k():
    model = OxStateCNN(n_classes=5)
    n = sum(p.numel() for p in model.parameters())
    assert n < 200_000, f"too many params: {n}"


def test_supports_variable_n_classes():
    for k in (2, 3, 5, 8):
        out = OxStateCNN(n_classes=k)(torch.zeros(2, 1, 200))
        assert out.shape == (2, k)
