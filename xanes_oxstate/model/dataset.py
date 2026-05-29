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
        self.e0 = e0
        self._cache: list[tuple[torch.Tensor, int]] = []

        if cache:
            kept_records: list[dict] = []
            dropped = 0
            for rec in records:
                energy = np.asarray(rec["energies"], dtype=np.float64)
                intensity = np.asarray(rec["intensities"], dtype=np.float64)
                try:
                    _, y = preprocess(energy, intensity, e0=self.e0)
                except ValueError:
                    dropped += 1
                    continue
                kept_records.append(rec)
                # Class mapping is built post-filter; we stash the tensor here
                # and finalize the label index below once class_to_ox_state exists.
                self._cache.append((torch.from_numpy(y).float().unsqueeze(0), rec["ox_state"]))
            if dropped:
                import warnings
                warnings.warn(
                    f"XanesDataset: dropped {dropped}/{len(records)} records "
                    f"that failed edge-jump normalization",
                    stacklevel=2,
                )
            self.records = kept_records
        else:
            self.records = records

        ox_states = sorted({r["ox_state"] for r in self.records})
        self.class_to_ox_state = {i: ox for i, ox in enumerate(ox_states)}
        self._ox_to_class = {ox: i for i, ox in self.class_to_ox_state.items()}

        if cache:
            # Rewrite cache entries with finalized class indices.
            self._cache = [
                (tensor, self._ox_to_class[ox]) for tensor, ox in self._cache
            ]

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
