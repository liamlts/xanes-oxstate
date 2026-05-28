"""Single-seed training with cosine LR decay and early stopping."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .cnn import OxStateCNN
from .dataset import XanesDataset


@dataclass
class TrainResult:
    history: dict[str, list[float]] = field(default_factory=dict)
    best_val_acc: float = 0.0
    epochs_run: int = 0


def _class_weights(ds: XanesDataset) -> torch.Tensor:
    counts = Counter(ds.labels)
    n_classes = ds.n_classes
    total = sum(counts.values())
    w = torch.zeros(n_classes)
    for c, n in counts.items():
        w[c] = total / (n_classes * n)
    return w


def train_one_seed(
    train_ds: XanesDataset,
    val_ds: XanesDataset,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
    patience: int = 10,
    ckpt_path: Path | None = None,
    device: str | None = None,
) -> TrainResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = OxStateCNN(n_classes=train_ds.n_classes).to(device)
    opt = Adam(model.parameters(), lr=lr)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    weights = _class_weights(train_ds).to(device)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best = -1.0
    bad_epochs = 0
    result = TrainResult(history=history)

    for ep in range(epochs):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), torch.as_tensor(y, device=device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        history["train_loss"].append(float(np.mean(losses)))

        model.eval()
        v_losses, correct, total = [], 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), torch.as_tensor(y, device=device)
                logits = model(x)
                v_losses.append(F.cross_entropy(logits, y, weight=weights).item())
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += y.numel()
        history["val_loss"].append(float(np.mean(v_losses)))
        val_acc = correct / max(total, 1)
        history["val_acc"].append(val_acc)
        result.epochs_run = ep + 1

        if val_acc > best:
            best = val_acc
            result.best_val_acc = best
            bad_epochs = 0
            if ckpt_path is not None:
                Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "n_classes": train_ds.n_classes,
                        "class_to_ox_state": train_ds.class_to_ox_state,
                        "seed": seed,
                    },
                    ckpt_path,
                )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    return result
