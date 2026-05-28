"""PyTorch Dataset wrapping cleaned JSONL records."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.preprocess import preprocess


class XanesDataset(Dataset):
    def __init__(
        self, records: list[dict], e0: float | None = None, cache: bool = True
    ):
        self.records = records
        self.e0 = e0
        ox_states = sorted({r["ox_state"] for r in records})
        self.class_to_ox_state = {i: ox for i, ox in enumerate(ox_states)}
        self._ox_to_class = {ox: i for i, ox in self.class_to_ox_state.items()}
        self._cache: list[tuple[torch.Tensor, int]] = []
        if cache:
            for rec in records:
                energy = np.asarray(rec["energies"], dtype=np.float64)
                intensity = np.asarray(rec["intensities"], dtype=np.float64)
                _, y = preprocess(energy, intensity, e0=self.e0)
                x = torch.from_numpy(y).float().unsqueeze(0)  # [1, 200]
                self._cache.append((x, self._ox_to_class[rec["ox_state"]]))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        if self._cache:
            return self._cache[idx]
        rec = self.records[idx]
        energy = np.asarray(rec["energies"], dtype=np.float64)
        intensity = np.asarray(rec["intensities"], dtype=np.float64)
        _, y = preprocess(energy, intensity, e0=self.e0)
        x = torch.from_numpy(y).float().unsqueeze(0)  # [1, 200]
        return x, self._ox_to_class[rec["ox_state"]]

    @property
    def n_classes(self) -> int:
        return len(self.class_to_ox_state)

    @property
    def labels(self) -> list[int]:
        return [self._ox_to_class[r["ox_state"]] for r in self.records]
