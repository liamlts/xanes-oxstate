"""Filter / dedup / class-prune raw MP XANES records."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .fetch import load_jsonl


@dataclass
class CleaningReport:
    element: str
    raw: int = 0
    kept: int = 0
    dropped_short: int = 0
    dropped_nan: int = 0
    dropped_dup: int = 0
    dropped_rare_class: int = 0
    final_class_counts: dict[int, int] | None = None


def _has_bad_values(rec: dict) -> bool:
    for arr in (rec["energies"], rec["intensities"]):
        for v in arr:
            if v is None:
                return True
            try:
                if math.isnan(v) or math.isinf(v):
                    return True
            except TypeError:
                return True
    return False


def clean_element(
    src: Path,
    dst: Path,
    min_points: int = 50,
    min_class_count: int = 50,
    min_classes: int = 2,
    report_path: Path | None = None,
) -> tuple[Path, CleaningReport]:
    src = Path(src)
    dst = Path(dst)
    records = list(load_jsonl(src))
    report = CleaningReport(element=_infer_element(records, src))
    report.raw = len(records)

    cleaned: list[dict] = []
    for rec in records:
        if len(rec["energies"]) < min_points:
            report.dropped_short += 1
            continue
        if _has_bad_values(rec):
            report.dropped_nan += 1
            continue
        cleaned.append(rec)

    # Dedup by (formula, ox_state); keep first (caller is responsible for
    # ordering by polymorph energy if desired — MP returns lowest-E first).
    seen: set[tuple[str, int]] = set()
    deduped: list[dict] = []
    for rec in cleaned:
        key = (rec["formula"], rec["ox_state"])
        if key in seen:
            report.dropped_dup += 1
            continue
        seen.add(key)
        deduped.append(rec)

    counts = Counter(r["ox_state"] for r in deduped)
    keep_classes = {c for c, n in counts.items() if n >= min_class_count}
    final = [r for r in deduped if r["ox_state"] in keep_classes]
    report.dropped_rare_class = len(deduped) - len(final)
    report.final_class_counts = dict(Counter(r["ox_state"] for r in final))
    report.kept = len(final)

    if len(keep_classes) < min_classes:
        # Don't write a useless file.
        if dst.exists():
            dst.unlink()
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("w") as f:
            for rec in final:
                f.write(json.dumps(rec) + "\n")

    if report_path is not None:
        Path(report_path).write_text(json.dumps(asdict(report), indent=2))

    return dst, report


def _infer_element(records: list[dict], src: Path) -> str:
    if records:
        return records[0]["element"]
    return src.stem
