"""Formula-disjoint, ox-state-stratified train/val/test split."""
from __future__ import annotations

import random
from collections import defaultdict


class LeakageError(AssertionError):
    pass


def _assign_bucket(idx: int, ratios: tuple[float, float, float]) -> str:
    # idx is a uniform [0, 1) rank for the formula within its class
    if idx < ratios[0]:
        return "train"
    if idx < ratios[0] + ratios[1]:
        return "val"
    return "test"


def split_records(
    records: list[dict],
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, list[dict]]:
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1")

    rng = random.Random(seed)

    # Group formulas by ox_state.
    by_class: dict[int, list[str]] = defaultdict(list)
    formulas_seen: set[str] = set()
    for rec in records:
        f = rec["formula"]
        if f in formulas_seen:
            continue
        formulas_seen.add(f)
        by_class[rec["ox_state"]].append(f)

    formula_to_bucket: dict[str, str] = {}
    for c, formulas in by_class.items():
        shuffled = formulas[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        for i, f in enumerate(shuffled):
            formula_to_bucket[f] = _assign_bucket(i / max(n, 1), ratios)

    splits = {"train": [], "val": [], "test": []}
    for rec in records:
        bucket = formula_to_bucket[rec["formula"]]
        splits[bucket].append(rec)

    assert_no_leakage(splits)
    return splits


def assert_no_leakage(splits: dict[str, list[dict]]) -> None:
    sets = {k: {r["formula"] for r in v} for k, v in splits.items()}
    if sets["train"] & sets["val"]:
        raise LeakageError("train/val share formulas")
    if sets["train"] & sets["test"]:
        raise LeakageError("train/test share formulas")
    if sets["val"] & sets["test"]:
        raise LeakageError("val/test share formulas")
