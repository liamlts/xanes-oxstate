"""Top-level orchestrator: fetch → clean → split → write parquet."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .clean import clean_element
from .fetch import fetch_element_spectra, load_jsonl
from .split import split_records


def build_element_dataset(
    element: str,
    raw_dir: Path,
    processed_dir: Path,
    report_dir: Path,
    seed: int = 42,
) -> dict[str, Path]:
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    raw_path = fetch_element_spectra(element, cache_dir=raw_dir)
    cleaned_path = raw_dir / f"{element}.cleaned.jsonl"
    _, report = clean_element(
        raw_path, cleaned_path,
        report_path=report_dir / f"{element}.json",
    )
    if not cleaned_path.exists():
        return {"_skipped": True, "report": report_dir / f"{element}.json"}

    records = list(load_jsonl(cleaned_path))
    splits = split_records(records, seed=seed)

    out = {}
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, recs in splits.items():
        df = pd.DataFrame(recs)
        path = processed_dir / f"{element}_{name}.parquet"
        df.to_parquet(path)
        out[name] = path

    return out
