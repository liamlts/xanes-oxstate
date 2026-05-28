"""Ensemble inference: average softmax probabilities."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..model.dataset import XanesDataset


def ensemble_predict_proba(
    models: list[torch.nn.Module], x: torch.Tensor
) -> np.ndarray:
    if not models:
        raise ValueError("no models supplied")
    probs = []
    with torch.no_grad():
        for m in models:
            m.eval()
            logits = m(x)
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.stack(probs).mean(axis=0)


def ensemble_predict_dataset(
    models: list[torch.nn.Module],
    dataset: XanesDataset,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_probs, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            all_probs.append(ensemble_predict_proba(models, x))
            all_y.append(np.asarray(y))
    return np.concatenate(all_probs), np.concatenate(all_y)


def ensemble_logits_dataset(
    models: list[torch.nn.Module],
    dataset: XanesDataset,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns averaged logits (not softmax). Used for temperature fitting."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_logits, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            l = []
            for m in models:
                m.eval()
                l.append(m(x).cpu().numpy())
            all_logits.append(np.stack(l).mean(axis=0))
            all_y.append(np.asarray(y))
    return np.concatenate(all_logits), np.concatenate(all_y)


def ensemble_per_model_logits_dataset(
    models: list[torch.nn.Module],
    dataset: XanesDataset,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns logits per model: (n_models, n_samples, n_classes), plus labels."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    per_model = [[] for _ in models]
    all_y = []
    with torch.no_grad():
        for x, y in loader:
            for i, m in enumerate(models):
                m.eval()
                per_model[i].append(m(x).cpu().numpy())
            all_y.append(np.asarray(y))
    stacked = np.stack([np.concatenate(pm) for pm in per_model])
    return stacked, np.concatenate(all_y)
