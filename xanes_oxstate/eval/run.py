"""End-to-end per-element evaluation: train ensemble + baselines, report metrics."""
from __future__ import annotations

import torch  # noqa: F401  (must precede lightgbm to avoid libomp dual-load on macOS)

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..baselines.features import extract_features
from ..baselines.gbdt import GBDTBaseline
from ..baselines.majority import MajorityClassifier
from ..data.preprocess import detect_e0, preprocess
from ..model.dataset import XanesDataset
from ..model.ensemble import load_ensemble, train_ensemble
from .calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
)
from .confusion import build_confusion, find_confused_pairs
from .metrics import top1_accuracy, per_class_f1, ensemble_disagreement
from .plots import reliability_diagram
from .predict import ensemble_logits_dataset, ensemble_predict_dataset


def _records_from_parquet(path: Path) -> list[dict]:
    return pd.read_parquet(path).to_dict("records")


def _hand_feature_matrix(
    records: list[dict], labels: list[int] | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Returns (X, y) where rows that failed preprocessing are dropped from both.

    If `labels` is None, only X is returned with zero-fill for failures (test path).
    """
    feats, kept_labels = [], []
    for i, rec in enumerate(records):
        e = np.asarray(rec["energies"], dtype=np.float64)
        y = np.asarray(rec["intensities"], dtype=np.float64)
        try:
            grid, y_norm = preprocess(e, y)
            e0 = detect_e0(e, y)
            feats.append(extract_features(grid, y_norm, e0=e0))
            if labels is not None:
                kept_labels.append(labels[i])
        except ValueError:
            if labels is None:
                feats.append(np.zeros(6))
    X = np.vstack(feats) if feats else np.empty((0, 6))
    if labels is None:
        return X, None
    return X, np.asarray(kept_labels, dtype=int)


def evaluate_element(
    element: str,
    processed_dir: Path,
    ckpt_dir: Path,
    metrics_dir: Path,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    epochs: int = 50,
) -> dict:
    processed_dir = Path(processed_dir)
    ckpt_dir = Path(ckpt_dir)
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_recs = _records_from_parquet(processed_dir / f"{element}_train.parquet")
    val_recs = _records_from_parquet(processed_dir / f"{element}_val.parquet")
    test_recs = _records_from_parquet(processed_dir / f"{element}_test.parquet")

    train_ds = XanesDataset(train_recs)
    val_ds = XanesDataset(val_recs)
    test_ds = XanesDataset(test_recs)

    train_ensemble(
        train_ds, val_ds,
        ckpt_dir=ckpt_dir, seeds=seeds, element=element, epochs=epochs,
    )
    models = load_ensemble(ckpt_dir, element=element)

    val_logits, val_y = ensemble_logits_dataset(models, val_ds)
    T = fit_temperature(val_logits, val_y)

    test_logits, test_y = ensemble_logits_dataset(models, test_ds)
    test_probs = _softmax(apply_temperature(test_logits, T))
    test_pred = test_probs.argmax(axis=1)

    cnn_acc = top1_accuracy(test_y, test_pred)
    cnn_f1 = per_class_f1(test_y, test_pred, n_classes=train_ds.n_classes)
    ece = expected_calibration_error(test_probs, test_y)

    # Ensemble disagreement on test
    per_model_pred = []
    for m in models:
        logits, _ = ensemble_logits_dataset([m], test_ds)
        per_model_pred.append(logits.argmax(axis=1))
    disagree = ensemble_disagreement(np.stack(per_model_pred))

    # Baselines
    majority = MajorityClassifier().fit(
        np.array([train_ds._ox_to_class[r["ox_state"]] for r in train_recs])
    )
    maj_pred = np.asarray(majority.predict(test_recs))
    maj_acc = top1_accuracy(test_y, maj_pred)

    X_train, y_train = _hand_feature_matrix(train_recs, labels=train_ds.labels)
    X_test, _ = _hand_feature_matrix(test_recs, labels=None)
    gbdt = GBDTBaseline(n_classes=train_ds.n_classes, random_state=0)
    gbdt.fit(X_train, y_train)
    gbdt_pred = gbdt.predict(X_test)
    gbdt_acc = top1_accuracy(test_y, gbdt_pred)

    cm = build_confusion(test_y, test_pred, n_classes=train_ds.n_classes)

    # Failures parquet
    failures = []
    for i, rec in enumerate(test_recs):
        if test_pred[i] != test_y[i]:
            failures.append({
                "mp_id": rec["mp_id"],
                "formula": rec["formula"],
                "true_class": int(test_y[i]),
                "pred_class": int(test_pred[i]),
                "confidence": float(test_probs[i].max()),
                "spectrum": rec["intensities"],
                "energies": rec["energies"],
            })
    pd.DataFrame(failures).to_parquet(metrics_dir / f"{element}_failures.parquet")

    # Reliability diagram
    reliability_diagram(test_probs, test_y,
                        out_path=metrics_dir / f"{element}_reliability.png")

    confused = find_confused_pairs(test_y, test_pred,
                                   n_classes=train_ds.n_classes)

    metrics = {
        "element": element,
        "temperature": float(T),
        "accuracy": {"majority": float(maj_acc),
                     "gbdt": float(gbdt_acc),
                     "cnn": float(cnn_acc)},
        "per_class_f1": cnn_f1,
        "ece": float(ece),
        "ensemble_disagreement": float(disagree),
        "class_to_ox_state": {int(k): int(v)
                              for k, v in train_ds.class_to_ox_state.items()},
        "confused_pairs": [
            {"true": p.true, "pred": p.pred, "rate": p.rate, "count": p.count}
            for p in confused
        ],
    }
    (metrics_dir / f"{element}.json").write_text(json.dumps(metrics, indent=2))
    np.save(metrics_dir / f"{element}_cm.npy", cm)
    return metrics


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
