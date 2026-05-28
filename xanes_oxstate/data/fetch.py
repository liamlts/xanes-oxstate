"""Fetch K-edge XANES spectra for one element from the Materials Project.

Auth: reads MP_API_KEY from the environment. Results are cached as
JSONL one record per spectrum so the pipeline is fully resumable.

Oxidation state is looked up from the MP summary endpoint's `possible_species`
field (BVAnalyzer pre-computed) rather than running `oxi_state_guesses` per
material — the per-call algorithm is exponential and unusable at scale.
"""
from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


_SPECIES_RE = re.compile(r"^([A-Z][a-z]?)([0-9]+)([+-])$")


@contextmanager
def _mp_client():
    from mp_api.client import MPRester

    key = os.environ.get("MP_API_KEY")
    if not key:
        raise RuntimeError("MP_API_KEY not set in environment")
    with MPRester(key) as r:
        yield r


def _parse_species(s: str) -> tuple[str, int] | None:
    """Parse 'Mn2+' / 'O2-' → ('Mn', 2) / ('O', -2). Returns None on no match."""
    m = _SPECIES_RE.match(s)
    if not m:
        return None
    sym, mag, sign = m.group(1), int(m.group(2)), m.group(3)
    return sym, mag if sign == "+" else -mag


def _ox_state_for_element(species_list, element: str) -> int | None:
    """Find integer ox-state of `element` in a possible_species list."""
    if not species_list:
        return None
    for s in species_list:
        parsed = _parse_species(str(s))
        if parsed is None:
            continue
        sym, ox = parsed
        if sym == element:
            return ox
    return None


def _extract_ox_state(doc, species_lookup: dict[str, list] | None = None) -> int | None:
    """Look up integer ox-state for the absorber.

    Fast path: MP `possible_species` (BVAnalyzer-precomputed). If the lookup
    misses or is absent, fall back to per-doc `composition.oxi_state_guesses`
    (slow, exponential — only viable for small mock fixtures).
    """
    elem_str = str(doc.absorbing_element)
    mp_id = str(doc.material_id)

    if species_lookup is not None:
        ox = _ox_state_for_element(species_lookup.get(mp_id), elem_str)
        if ox is not None:
            return ox

    comp = getattr(doc, "composition", None) or getattr(
        getattr(doc, "structure", None), "composition", None
    )
    if comp is None:
        return None
    try:
        guesses = comp.oxi_state_guesses()
    except Exception:
        return None
    if not guesses:
        return None
    for key, val in guesses[0].items():
        if str(key) == elem_str or getattr(key, "symbol", None) == elem_str:
            if isinstance(val, (int, float)) and float(val).is_integer():
                return int(val)
            return None
    return None


def _doc_to_record(doc, species_lookup: dict[str, list] | None = None) -> dict | None:
    ox = _extract_ox_state(doc, species_lookup=species_lookup)
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


def _fetch_species_lookup(client, mp_ids: list[str], chunk: int = 1000) -> dict[str, list]:
    """Batch-query summary endpoint for possible_species per material."""
    out: dict[str, list] = {}
    for i in range(0, len(mp_ids), chunk):
        batch = mp_ids[i : i + chunk]
        docs = client.materials.summary.search(
            material_ids=batch, fields=["material_id", "possible_species"]
        )
        for d in docs:
            out[str(d.material_id)] = list(d.possible_species or [])
    return out


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
        docs = list(client.materials.xas.search(
            absorbing_element=element, edge=edge, spectrum_type="XANES",
            fields=["material_id", "formula_pretty", "absorbing_element",
                    "spectrum"],
        ))
        mp_ids = [str(d.material_id) for d in docs]
        species_lookup = _fetch_species_lookup(client, mp_ids) if mp_ids else {}

    tmp = out.with_suffix(".jsonl.tmp")
    n = 0
    with tmp.open("w") as f:
        for doc in docs:
            rec = _doc_to_record(doc, species_lookup=species_lookup)
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
