import numpy as np
import torch

from xanes_oxstate.model.cnn import OxStateCNN
from xanes_oxstate.eval.predict import ensemble_predict_proba


class _IdentityCNN(OxStateCNN):
    def __init__(self, n_classes=3, target_class=0):
        super().__init__(n_classes=n_classes)
        self._tc = target_class

    def forward(self, x):
        # Always favor the target class
        out = torch.zeros(x.shape[0], 3)
        out[:, self._tc] = 5.0
        return out


def test_ensemble_predict_proba_averages_probs():
    models = [_IdentityCNN(target_class=0), _IdentityCNN(target_class=1)]
    x = torch.zeros(4, 1, 200)
    probs = ensemble_predict_proba(models, x)
    assert probs.shape == (4, 3)
    # Two equally-favored classes after averaging
    assert np.isclose(probs[0, 0], probs[0, 1])
    assert np.allclose(probs.sum(axis=1), 1.0)
