"""Per-seed ensemble training + checkpoint loading."""
from __future__ import annotations

from pathlib import Path

import torch

from .cnn import OxStateCNN
from .dataset import XanesDataset
from .train import train_one_seed


def train_ensemble(
    train_ds: XanesDataset,
    val_ds: XanesDataset,
    ckpt_dir: Path,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    element: str = "Mn",
    **train_kwargs,
) -> list[Path]:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for s in seeds:
        path = ckpt_dir / f"{element}_seed{s}.pt"
        if not path.exists():
            train_one_seed(
                train_ds, val_ds, seed=s, ckpt_path=path, **train_kwargs
            )
        paths.append(path)
    return paths


def load_ensemble(ckpt_dir: Path, element: str) -> list[OxStateCNN]:
    ckpt_dir = Path(ckpt_dir)
    models: list[OxStateCNN] = []
    for p in sorted(ckpt_dir.glob(f"{element}_seed*.pt")):
        state = torch.load(p, map_location="cpu")
        m = OxStateCNN(n_classes=state["n_classes"])
        m.load_state_dict(state["state_dict"])
        m.eval()
        models.append(m)
    return models
