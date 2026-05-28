"""Fetch K-edge XANES spectra for one element from the Materials Project.

Auth: reads MP_API_KEY from the environment. Results are cached as
JSONL one record per spectrum so the pipeline is fully resumable.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def _mp_client():
    from mp_api.client import MPRester

    key = os.environ.get("MP_API_KEY")
    if not key:
        raise RuntimeError("MP_API_KEY not set in environment")
    with MPRester(key) as r:
        yield r


def _extract_ox_state(doc) -> int | None:
    """Best-effort integer ox-state assignment for the absorbing element."""
    elem = doc.absorbing_element
    try:
        guesses = doc.structure.composition.oxi_state_guesses()
    except Exception:
        return None
    if not guesses:
        return None
    g = guesses[0]
    if elem not in g:
        return None
    v = g[elem]
    if not float(v).is_integer():
        return None
    return int(v)


def _doc_to_record(doc) -> dict | None:
    ox = _extract_ox_state(doc)
    if ox is None:
        return None
    energies = list(doc.spectrum.x)
    intensities = list(doc.spectrum.y)
    if len(energies) < 2 or len(energies) != len(intensities):
        return None
    return {
        "mp_id": str(doc.material_id),
        "element": str(doc.absorbing_element),
        "ox_state": ox,
        "formula": str(doc.formula_pretty),
        "energies": energies,
        "intensities": intensities,
    }


def fetch_element_spectra(
    element: str,
    cache_dir: Path,
    edge: str = "K",
    skip_if_exists: bool = True,
) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{element}.jsonl"

    if skip_if_exists and out.exists() and out.stat().st_size > 0:
        return out

    with _mp_client() as client:
        docs = client.materials.xas.search(
            absorbing_element=element, edge=edge, spectrum_type="XANES"
        )

    tmp = out.with_suffix(".jsonl.tmp")
    n = 0
    with tmp.open("w") as f:
        for doc in docs:
            rec = _doc_to_record(doc)
            if rec is None:
                continue
            f.write(json.dumps(rec) + "\n")
            n += 1
    os.replace(tmp, out)
    return out


def load_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
