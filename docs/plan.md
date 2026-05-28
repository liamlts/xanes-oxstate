# xanes-oxstate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 1D-CNN classifier that predicts oxidation states from K-edge XANES spectra across 8 transition metals, with per-element confusion matrices and a small physics finding on spectrally degenerate ox-state pairs.

**Architecture:** Python package `xanes_oxstate` with separate modules for data ingestion (Materials Project API), preprocessing (edge-jump normalization to a fixed 200-point grid), model (small 1D-CNN, ensembled over 5 seeds, temperature-calibrated), evaluation (CNN vs. majority + GBDT baselines), and a physics-analysis pass that explains confused (true, predicted) ox-state pairs. PyTorch for the CNN, scikit-learn / lightgbm for baselines, matplotlib for figures, pytest for tests.

**Tech Stack:** Python 3.11+, mp-api, pymatgen, numpy, scipy, pandas, pyarrow, torch, scikit-learn, lightgbm, matplotlib, pyyaml, pytest.

---

## File Structure

```
xanes-oxstate/
├── README.md                      ← portfolio writeup
├── pyproject.toml                 ← package + deps
├── Makefile                       ← make {data, train, eval, figures, all}
├── .gitignore
│
├── xanes_oxstate/
│   ├── __init__.py
│   ├── spectrum.py                ← Spectrum dataclass
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetch.py               ← MP API client w/ JSONL cache
│   │   ├── clean.py               ← filter / dedup / ox-state assign
│   │   ├── preprocess.py          ← resample + edge-jump norm
│   │   └── split.py               ← formula-disjoint split + leak check
│   ├── model/
│   │   ├── __init__.py
│   │   ├── cnn.py                 ← OxStateCNN
│   │   ├── dataset.py             ← PyTorch Dataset wrapping parquet
│   │   └── train.py               ← single-seed training
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py             ← accuracy / per-class F1 / disagreement
│   │   ├── calibration.py         ← temperature scaling + ECE
│   │   ├── confusion.py           ← row-normalized matrix
│   │   └── plots.py               ← headline figures
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── features.py            ← hand features for GBDT
│   │   ├── majority.py            ← majority-class predictor
│   │   └── gbdt.py                ← lightgbm wrapper
│   ├── physics/
│   │   ├── __init__.py
│   │   └── analysis.py            ← confused-cell analysis (overlay / LR / structural)
│   └── cli.py                     ← module entry points
│
├── configs/                       ← per-element YAML hyperparams
├── notebooks/
│   └── xanes_oxstate.ipynb        ← end-to-end walkthrough
├── data/                          ← gitignored
├── checkpoints/                   ← gitignored
├── figures/                       ← versioned final figures
└── tests/
    ├── conftest.py                ← shared fixtures
    ├── fixtures/                  ← canned MP responses + synthetic spectra
    ├── test_spectrum.py
    ├── test_clean.py
    ├── test_preprocess.py
    ├── test_split.py
    ├── test_dataset.py
    ├── test_cnn.py
    ├── test_metrics.py
    ├── test_calibration.py
    ├── test_features.py
    ├── test_majority.py
    ├── test_gbdt.py
    └── test_smoke.py              ← end-to-end on tiny synthetic
```

**File responsibility rules** (locked):
- `spectrum.py` defines the only shared data type. Anything that flows between modules is a `Spectrum` (or an array derived from one).
- `data/` modules each do one thing (fetch / clean / preprocess / split). They compose via the file system (jsonl → cleaned parquet → preprocessed parquet → split parquet).
- `model/` knows nothing about evaluation; `eval/` knows nothing about training.
- `baselines/` and `physics/` are leaves — nothing else imports from them.

---

## Phase 0 — Repo scaffolding

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `xanes_oxstate/__init__.py`
- Create: `xanes_oxstate/data/__init__.py`, `model/__init__.py`, `eval/__init__.py`, `baselines/__init__.py`, `physics/__init__.py`
- Create: `tests/__init__.py`, `tests/conftest.py`
- Create: `tests/fixtures/.gitkeep`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "xanes-oxstate"
version = "0.1.0"
description = "Oxidation-state classification from XANES spectra"
authors = [{ name = "Liam Schmidt" }]
requires-python = ">=3.11"
dependencies = [
    "mp-api>=0.41",
    "pymatgen>=2024.1",
    "numpy>=1.26",
    "scipy>=1.11",
    "pandas>=2.1",
    "pyarrow>=14",
    "torch>=2.2",
    "scikit-learn>=1.4",
    "lightgbm>=4.1",
    "matplotlib>=3.8",
    "pyyaml>=6",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-cov>=4", "pytest-mock>=3"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["xanes_oxstate*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
filterwarnings = ["ignore::DeprecationWarning"]
```

- [ ] **Step 2: Write `.gitignore`**

```
__pycache__/
*.pyc
.pytest_cache/
.venv/
*.egg-info/
data/raw/
data/processed/
checkpoints/
runs/
.ipynb_checkpoints/
.DS_Store
```

- [ ] **Step 3: Write minimal `xanes_oxstate/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 4: Write empty `__init__.py` in every submodule**

```bash
touch xanes_oxstate/data/__init__.py xanes_oxstate/model/__init__.py \
      xanes_oxstate/eval/__init__.py xanes_oxstate/baselines/__init__.py \
      xanes_oxstate/physics/__init__.py tests/__init__.py
mkdir -p tests/fixtures && touch tests/fixtures/.gitkeep
```

- [ ] **Step 5: Write `tests/conftest.py` (placeholder for shared fixtures)**

```python
import pytest
```

- [ ] **Step 6: Install in dev mode and verify pytest runs**

Run: `pip install -e ".[dev]" && pytest -q`
Expected: `no tests ran in 0.0Xs` (no tests yet, but package installs and pytest discovers).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore xanes_oxstate tests
git commit -m "chore: project scaffolding"
```

---

### Task 2: `Spectrum` dataclass

**Files:**
- Create: `xanes_oxstate/spectrum.py`
- Create: `tests/test_spectrum.py`

- [ ] **Step 1: Write failing test**

`tests/test_spectrum.py`:

```python
import numpy as np
import pytest

from xanes_oxstate.spectrum import Spectrum


def test_spectrum_holds_required_fields():
    s = Spectrum(
        energy=np.linspace(6500, 6600, 100),
        intensity=np.zeros(100),
        element="Mn",
        ox_state=2,
        formula="MnO",
        mp_id="mp-19006",
    )
    assert s.element == "Mn"
    assert s.ox_state == 2
    assert s.energy.shape == (100,)
    assert s.intensity.shape == (100,)


def test_spectrum_requires_matching_lengths():
    with pytest.raises(ValueError):
        Spectrum(
            energy=np.zeros(100),
            intensity=np.zeros(99),
            element="Mn",
            ox_state=2,
            formula="MnO",
            mp_id="mp-19006",
        )


def test_spectrum_is_frozen():
    s = Spectrum(
        energy=np.zeros(10),
        intensity=np.zeros(10),
        element="Fe",
        ox_state=3,
        formula="Fe2O3",
        mp_id="mp-19770",
    )
    with pytest.raises((AttributeError, TypeError)):
        s.element = "Mn"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_spectrum.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'xanes_oxstate.spectrum'`.

- [ ] **Step 3: Implement `Spectrum`**

`xanes_oxstate/spectrum.py`:

```python
from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class Spectrum:
    energy: np.ndarray
    intensity: np.ndarray
    element: str
    ox_state: int
    formula: str
    mp_id: str

    def __post_init__(self):
        if self.energy.shape != self.intensity.shape:
            raise ValueError(
                f"energy {self.energy.shape} and intensity "
                f"{self.intensity.shape} must match"
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_spectrum.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add xanes_oxstate/spectrum.py tests/test_spectrum.py
git commit -m "feat(spectrum): add Spectrum dataclass"
```

---

### Task 3: Shared test fixtures

**Files:**
- Modify: `tests/conftest.py`
- Create: `tests/fixtures/synthetic_spectra.py`

- [ ] **Step 1: Write `tests/fixtures/synthetic_spectra.py`**

```python
"""Synthetic XANES spectra for fast, deterministic tests.

These are NOT physical — they are toy curves with the right shape
(rising step + small white-line bump) to exercise preprocessing,
splitting, and model code without hitting the MP API.
"""
import numpy as np
from xanes_oxstate.spectrum import Spectrum


def fake_xanes(
    element: str = "Mn",
    ox_state: int = 2,
    formula: str = "MnO",
    mp_id: str = "mp-fake-0",
    e0: float = 6539.0,
    seed: int = 0,
) -> Spectrum:
    rng = np.random.default_rng(seed)
    energy = np.linspace(e0 - 12, e0 + 45, 120)
    edge = 1.0 / (1.0 + np.exp(-(energy - e0) / 1.5))   # logistic step
    bump = 0.25 * np.exp(-((energy - (e0 + 2 + ox_state * 1.2)) ** 2) / 4.0)
    noise = 0.005 * rng.standard_normal(energy.size)
    intensity = edge + bump + noise
    return Spectrum(
        energy=energy,
        intensity=intensity,
        element=element,
        ox_state=ox_state,
        formula=formula,
        mp_id=mp_id,
    )


def make_dataset(n_per_class: int = 20, classes=(2, 3, 4), element: str = "Mn"):
    spectra = []
    idx = 0
    for c in classes:
        for k in range(n_per_class):
            spectra.append(
                fake_xanes(
                    element=element,
                    ox_state=c,
                    formula=f"{element}-fake-{c}-{k % 5}",  # 5 distinct formulas / class
                    mp_id=f"mp-fake-{idx}",
                    seed=idx,
                )
            )
            idx += 1
    return spectra
```

- [ ] **Step 2: Wire it into `conftest.py`**

```python
import pytest
from tests.fixtures.synthetic_spectra import fake_xanes, make_dataset


@pytest.fixture
def mn_spectrum():
    return fake_xanes(element="Mn", ox_state=2)


@pytest.fixture
def mn_dataset():
    return make_dataset(n_per_class=20, classes=(2, 3, 4), element="Mn")
```

- [ ] **Step 3: Sanity test the fixture itself**

`tests/test_spectrum.py` (append):

```python
def test_fake_xanes_fixture(mn_spectrum):
    assert mn_spectrum.element == "Mn"
    assert mn_spectrum.energy.shape == mn_spectrum.intensity.shape == (120,)


def test_make_dataset_fixture(mn_dataset):
    assert len(mn_dataset) == 60
    assert {s.ox_state for s in mn_dataset} == {2, 3, 4}
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_spectrum.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/fixtures/
git commit -m "test: synthetic XANES fixtures"
```

---

## Phase 1 — Data pipeline

### Task 4: MP API fetcher with JSONL cache

**Files:**
- Create: `xanes_oxstate/data/fetch.py`
- Create: `tests/test_fetch.py`

The MP client is mocked in unit tests; a manual smoke fetch happens in Task 9.

- [ ] **Step 1: Write failing test**

`tests/test_fetch.py`:

```python
import json
from unittest.mock import MagicMock, patch

from xanes_oxstate.data.fetch import fetch_element_spectra


def _mock_doc(mp_id, element, ox, formula, energies, intensities):
    doc = MagicMock()
    doc.material_id = mp_id
    doc.formula_pretty = formula
    doc.absorbing_element = element
    doc.spectrum = MagicMock(
        x=list(energies), y=list(intensities), absorbing_index=0
    )
    doc.structure = MagicMock()
    doc.structure.species_and_occu = [{element: 1.0}]
    doc.structure.composition.oxi_state_guesses = MagicMock(
        return_value=[{element: ox}]
    )
    return doc


def test_fetch_writes_jsonl_cache(tmp_path):
    fake_docs = [
        _mock_doc("mp-1", "Mn", 2, "MnO",
                  [6520, 6540, 6560], [0.1, 0.5, 0.9]),
        _mock_doc("mp-2", "Mn", 4, "MnO2",
                  [6520, 6540, 6560], [0.0, 0.4, 1.0]),
    ]
    with patch("xanes_oxstate.data.fetch._mp_client") as mk:
        client = mk.return_value.__enter__.return_value
        client.materials.xas.search.return_value = fake_docs

        out = fetch_element_spectra("Mn", cache_dir=tmp_path, edge="K")

    assert out.exists()
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["element"] == "Mn"
    assert rec["mp_id"] == "mp-1"
    assert rec["ox_state"] == 2
    assert rec["formula"] == "MnO"
    assert rec["energies"] == [6520, 6540, 6560]


def test_fetch_is_resumable(tmp_path):
    cache = tmp_path / "Mn.jsonl"
    cache.write_text(json.dumps({"mp_id": "mp-cached"}) + "\n")

    with patch("xanes_oxstate.data.fetch._mp_client") as mk:
        out = fetch_element_spectra(
            "Mn", cache_dir=tmp_path, edge="K", skip_if_exists=True
        )
        mk.assert_not_called()

    assert out == cache
    assert "mp-cached" in cache.read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fetch.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `fetch.py`**

`xanes_oxstate/data/fetch.py`:

```python
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

    n = 0
    with out.open("w") as f:
        for doc in docs:
            rec = _doc_to_record(doc)
            if rec is None:
                continue
            f.write(json.dumps(rec) + "\n")
            n += 1
    return out


def load_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_fetch.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add xanes_oxstate/data/fetch.py tests/test_fetch.py
git commit -m "feat(data): MP API fetcher with JSONL caching"
```

---

### Task 5: Cleaning pipeline

**Files:**
- Create: `xanes_oxstate/data/clean.py`
- Create: `tests/test_clean.py`

- [ ] **Step 1: Write failing test**

`tests/test_clean.py`:

```python
import json
from pathlib import Path

from xanes_oxstate.data.clean import clean_element, CleaningReport


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records))


def _rec(mp_id, element, ox, formula, n=100):
    return {
        "mp_id": mp_id,
        "element": element,
        "ox_state": ox,
        "formula": formula,
        "energies": list(range(n)),
        "intensities": [0.1 * i for i in range(n)],
    }


def test_clean_drops_short_spectra(tmp_path):
    src = tmp_path / "Mn.jsonl"
    _write_jsonl(src, [
        _rec("mp-1", "Mn", 2, "MnO", n=49),
        _rec("mp-2", "Mn", 2, "MnO", n=100),
    ])
    out, report = clean_element(src, tmp_path / "out.jsonl", min_classes=1)
    kept = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["mp_id"] for r in kept] == ["mp-2"]
    assert report.dropped_short == 1


def test_clean_dedups_by_formula_ox(tmp_path):
    src = tmp_path / "Mn.jsonl"
    _write_jsonl(src, [
        _rec("mp-1", "Mn", 2, "MnO"),
        _rec("mp-2", "Mn", 2, "MnO"),
        _rec("mp-3", "Mn", 3, "Mn2O3"),
    ])
    out, report = clean_element(src, tmp_path / "out.jsonl", min_classes=1)
    kept = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(kept) == 2
    assert report.dropped_dup == 1


def test_clean_drops_small_classes(tmp_path):
    src = tmp_path / "Mn.jsonl"
    recs = (
        [_rec(f"mp-{i}", "Mn", 2, f"MnO_{i}") for i in range(60)]
        + [_rec(f"mp-x{i}", "Mn", 7, f"KMnO4_{i}") for i in range(5)]
    )
    _write_jsonl(src, recs)
    out, report = clean_element(
        src, tmp_path / "out.jsonl", min_class_count=50, min_classes=1
    )
    kept = [json.loads(l) for l in out.read_text().splitlines()]
    assert {r["ox_state"] for r in kept} == {2}
    assert report.dropped_rare_class == 5


def test_clean_writes_report(tmp_path):
    src = tmp_path / "Mn.jsonl"
    _write_jsonl(src, [_rec(f"mp-{i}", "Mn", 2, f"MnO_{i}") for i in range(60)])
    out, report = clean_element(
        src, tmp_path / "out.jsonl",
        min_class_count=50, min_classes=1, report_path=tmp_path / "rep.json",
    )
    rep = json.loads((tmp_path / "rep.json").read_text())
    assert rep["element"] == "Mn"
    assert rep["kept"] == 60
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_clean.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `clean.py`**

`xanes_oxstate/data/clean.py`:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_clean.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add xanes_oxstate/data/clean.py tests/test_clean.py
git commit -m "feat(data): cleaning pipeline with cleaning report"
```

---

### Task 6: Preprocessing (resample + edge-jump normalization)

**Files:**
- Create: `xanes_oxstate/data/preprocess.py`
- Create: `tests/test_preprocess.py`

- [ ] **Step 1: Write failing test**

`tests/test_preprocess.py`:

```python
import numpy as np
import pytest

from xanes_oxstate.data.preprocess import (
    resample_to_grid,
    edge_jump_normalize,
    detect_e0,
    preprocess,
)


def _step_spectrum(e0=6539.0, n=400):
    e = np.linspace(e0 - 30, e0 + 70, n)
    pre = 0.05 * (e - (e0 - 30))
    step = 1.0 / (1.0 + np.exp(-(e - e0) / 1.0))
    intensity = pre + step
    return e, intensity


def test_resample_returns_fixed_grid():
    e, y = _step_spectrum()
    grid, y_r = resample_to_grid(e, y, e0=6539.0, n_points=200,
                                 lo_eV=-10, hi_eV=40)
    assert grid.shape == (200,)
    assert y_r.shape == (200,)
    assert np.isclose(grid[0], 6539.0 - 10)
    assert np.isclose(grid[-1], 6539.0 + 40)


def test_resample_rejects_excessive_extrapolation():
    e = np.linspace(6535, 6545, 50)  # too narrow
    y = np.linspace(0, 1, 50)
    with pytest.raises(ValueError):
        resample_to_grid(e, y, e0=6539.0, n_points=200, lo_eV=-10, hi_eV=40,
                        max_extrap_eV=5)


def test_edge_jump_normalization_yields_unit_jump():
    e, y = _step_spectrum()
    y_n = edge_jump_normalize(e, y, e0=6539.0)
    pre_mask = (e >= 6539.0 - 10) & (e <= 6539.0 - 5)
    post_mask = (e >= 6539.0 + 30) & (e <= 6539.0 + 40)
    assert abs(y_n[pre_mask].mean()) < 0.05
    assert abs(y_n[post_mask].mean() - 1.0) < 0.05


def test_detect_e0_from_inflection():
    e, y = _step_spectrum(e0=6539.0)
    assert abs(detect_e0(e, y) - 6539.0) < 0.5


def test_preprocess_end_to_end():
    from tests.fixtures.synthetic_spectra import fake_xanes

    s = fake_xanes(element="Mn", ox_state=2, e0=6539.0)
    grid, y = preprocess(s.energy, s.intensity, e0=6539.0)
    assert grid.shape == (200,)
    assert y.shape == (200,)
    pre_mask = (grid >= 6539.0 - 10) & (grid <= 6539.0 - 5)
    post_mask = (grid >= 6539.0 + 30) & (grid <= 6539.0 + 40)
    assert abs(y[pre_mask].mean()) < 0.1
    assert abs(y[post_mask].mean() - 1.0) < 0.1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_preprocess.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `preprocess.py`**

`xanes_oxstate/data/preprocess.py`:

```python
"""Resample to a fixed grid and apply edge-jump normalization.

The whole module is pure NumPy/SciPy — no larch dependency.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


GRID_N = 200
GRID_LO_EV = -10.0
GRID_HI_EV = 40.0


def detect_e0(energy: np.ndarray, intensity: np.ndarray) -> float:
    """Return the energy of maximum first derivative — the edge position."""
    dy = np.gradient(intensity, energy)
    # Restrict the search window to a sensible range above the lowest energy
    # to avoid spurious early peaks.
    lo = energy.min() + 0.1 * energy.ptp()
    hi = energy.max() - 0.1 * energy.ptp()
    mask = (energy >= lo) & (energy <= hi)
    idx = np.argmax(dy[mask])
    return float(energy[mask][idx])


def resample_to_grid(
    energy: np.ndarray,
    intensity: np.ndarray,
    e0: float,
    n_points: int = GRID_N,
    lo_eV: float = GRID_LO_EV,
    hi_eV: float = GRID_HI_EV,
    max_extrap_eV: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(e0 + lo_eV, e0 + hi_eV, n_points)
    if grid.min() < energy.min() - max_extrap_eV:
        raise ValueError(
            f"requires {energy.min() - grid.min():.1f} eV extrapolation "
            f"below source (max allowed {max_extrap_eV})"
        )
    if grid.max() > energy.max() + max_extrap_eV:
        raise ValueError(
            f"requires {grid.max() - energy.max():.1f} eV extrapolation "
            f"above source (max allowed {max_extrap_eV})"
        )
    cs = CubicSpline(energy, intensity, extrapolate=True)
    return grid, cs(grid)


def edge_jump_normalize(
    energy: np.ndarray, intensity: np.ndarray, e0: float
) -> np.ndarray:
    pre = (energy >= e0 - 10) & (energy <= e0 - 5)
    post = (energy >= e0 + 30) & (energy <= e0 + 40)
    if pre.sum() < 2 or post.sum() < 2:
        raise ValueError("need both pre-edge and post-edge windows populated")

    a_pre, b_pre = np.polyfit(energy[pre], intensity[pre], 1)
    y = intensity - (a_pre * energy + b_pre)

    a_post, b_post = np.polyfit(energy[post], y[post], 1)
    jump = a_post * e0 + b_post
    if not np.isfinite(jump) or jump <= 0:
        raise ValueError(f"non-positive edge jump: {jump}")
    return y / jump


def preprocess(
    energy: np.ndarray,
    intensity: np.ndarray,
    e0: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if e0 is None:
        e0 = detect_e0(energy, intensity)
    grid, y = resample_to_grid(energy, intensity, e0=e0)
    y_norm = edge_jump_normalize(grid, y, e0=e0)
    return grid, y_norm
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_preprocess.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add xanes_oxstate/data/preprocess.py tests/test_preprocess.py
git commit -m "feat(data): resample + edge-jump normalization"
```

---

### Task 7: Train/val/test split with leak check

**Files:**
- Create: `xanes_oxstate/data/split.py`
- Create: `tests/test_split.py`

- [ ] **Step 1: Write failing test**

`tests/test_split.py`:

```python
import json
from pathlib import Path

import pytest

from xanes_oxstate.data.split import split_records, LeakageError


def _records(n_per_formula=2, n_formulas=12, classes=(2, 3, 4)):
    recs = []
    idx = 0
    for c in classes:
        for f in range(n_formulas):
            for k in range(n_per_formula):
                recs.append({
                    "mp_id": f"mp-{idx}",
                    "element": "Mn",
                    "ox_state": c,
                    "formula": f"MnO_{c}_{f}",
                })
                idx += 1
    return recs


def test_split_is_formula_disjoint():
    recs = _records()
    splits = split_records(recs, seed=42, ratios=(0.7, 0.15, 0.15))
    train_f = {r["formula"] for r in splits["train"]}
    val_f = {r["formula"] for r in splits["val"]}
    test_f = {r["formula"] for r in splits["test"]}
    assert train_f.isdisjoint(val_f)
    assert train_f.isdisjoint(test_f)
    assert val_f.isdisjoint(test_f)


def test_split_keeps_all_classes_in_train():
    recs = _records()
    splits = split_records(recs, seed=42, ratios=(0.7, 0.15, 0.15))
    train_classes = {r["ox_state"] for r in splits["train"]}
    assert train_classes == {2, 3, 4}


def test_split_ratios_approx():
    recs = _records(n_per_formula=2, n_formulas=20, classes=(2, 3, 4))
    splits = split_records(recs, seed=42, ratios=(0.7, 0.15, 0.15))
    total = sum(len(v) for v in splits.values())
    assert total == len(recs)
    assert 0.6 < len(splits["train"]) / total < 0.8
    assert 0.05 < len(splits["val"]) / total < 0.25
    assert 0.05 < len(splits["test"]) / total < 0.25


def test_split_raises_on_leakage_post_check():
    recs = _records()
    splits = split_records(recs, seed=42, ratios=(0.7, 0.15, 0.15))
    # Inject a leak.
    splits["val"].append(splits["train"][0])
    with pytest.raises(LeakageError):
        from xanes_oxstate.data.split import assert_no_leakage
        assert_no_leakage(splits)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_split.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `split.py`**

`xanes_oxstate/data/split.py`:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_split.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add xanes_oxstate/data/split.py tests/test_split.py
git commit -m "feat(data): formula-disjoint train/val/test split"
```

---

### Task 8: PyTorch Dataset wrapping cleaned records

**Files:**
- Create: `xanes_oxstate/model/dataset.py`
- Create: `tests/test_dataset.py`

- [ ] **Step 1: Write failing test**

`tests/test_dataset.py`:

```python
import numpy as np
import torch

from xanes_oxstate.model.dataset import XanesDataset


def _records(n=30):
    rng = np.random.default_rng(0)
    e0 = 6539.0
    recs = []
    for i in range(n):
        e = np.linspace(e0 - 30, e0 + 70, 400)
        step = 1.0 / (1.0 + np.exp(-(e - e0)))
        recs.append({
            "mp_id": f"mp-{i}",
            "element": "Mn",
            "ox_state": (i % 3) + 2,
            "formula": f"Mn_{i % 5}",
            "energies": e.tolist(),
            "intensities": (step + 0.01 * rng.standard_normal(400)).tolist(),
        })
    return recs


def test_dataset_returns_tensor_and_label():
    ds = XanesDataset(_records())
    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape == (1, 200)
    assert isinstance(y, int)


def test_dataset_class_index_is_dense():
    ds = XanesDataset(_records())
    labels = sorted({int(ds[i][1]) for i in range(len(ds))})
    assert labels == list(range(len(labels)))


def test_dataset_exposes_ox_state_mapping():
    ds = XanesDataset(_records())
    assert ds.class_to_ox_state[0] in {2, 3, 4}
    inverse = {v: k for k, v in ds.class_to_ox_state.items()}
    assert ds[0][1] == inverse[ds.records[0]["ox_state"]]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dataset.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `dataset.py`**

`xanes_oxstate/model/dataset.py`:

```python
"""PyTorch Dataset wrapping cleaned JSONL records."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.preprocess import preprocess


class XanesDataset(Dataset):
    def __init__(self, records: list[dict], e0: float | None = None):
        self.records = records
        self.e0 = e0
        ox_states = sorted({r["ox_state"] for r in records})
        self.class_to_ox_state = {i: ox for i, ox in enumerate(ox_states)}
        self._ox_to_class = {ox: i for i, ox in self.class_to_ox_state.items()}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rec = self.records[idx]
        energy = np.asarray(rec["energies"], dtype=np.float64)
        intensity = np.asarray(rec["intensities"], dtype=np.float64)
        _, y = preprocess(energy, intensity, e0=self.e0)
        x = torch.from_numpy(y).float().unsqueeze(0)  # [1, 200]
        return x, self._ox_to_class[rec["ox_state"]]

    @property
    def n_classes(self) -> int:
        return len(self.class_to_ox_state)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_dataset.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add xanes_oxstate/model/dataset.py tests/test_dataset.py
git commit -m "feat(model): PyTorch Dataset with on-the-fly preprocessing"
```

---

### Task 9: Data orchestration CLI + smoke fetch

**Files:**
- Create: `xanes_oxstate/data/run.py`
- Modify: `xanes_oxstate/cli.py` (new file)
- Modify: `Makefile`

- [ ] **Step 1: Write `xanes_oxstate/data/run.py`**

```python
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
```

- [ ] **Step 2: Write `xanes_oxstate/cli.py`**

```python
"""Entry points: python -m xanes_oxstate.<cmd>."""
from __future__ import annotations

import argparse
from pathlib import Path

from .data.run import build_element_dataset


def cmd_build_data(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m xanes_oxstate.build_data")
    p.add_argument("--element", required=True)
    p.add_argument("--raw-dir", default="data/raw", type=Path)
    p.add_argument("--processed-dir", default="data/processed", type=Path)
    p.add_argument("--report-dir", default="data/reports", type=Path)
    p.add_argument("--seed", default=42, type=int)
    args = p.parse_args(argv)

    paths = build_element_dataset(
        args.element,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        report_dir=args.report_dir,
        seed=args.seed,
    )
    print(f"[{args.element}] {paths}")


if __name__ == "__main__":
    cmd_build_data()
```

- [ ] **Step 3: Create initial `Makefile`**

```makefile
ELEMENTS = Ti V Cr Mn Fe Co Ni Cu

.PHONY: data
data:
	@for e in $(ELEMENTS); do \
	  python -m xanes_oxstate.cli --element $$e; \
	done

.PHONY: test
test:
	pytest -q
```

- [ ] **Step 4: Smoke test (skip if no MP_API_KEY)**

This is a manual one-off check, not an automated test:

```bash
MP_API_KEY=$YOUR_KEY python -m xanes_oxstate.cli --element Mn
ls -la data/raw/Mn.jsonl data/processed/Mn_train.parquet
```

Expected: both files exist, `Mn.jsonl` is multi-MB, train parquet has thousands of rows.

If MP_API_KEY is not set, this step is deferred to the user's first real run.

- [ ] **Step 5: Commit**

```bash
git add xanes_oxstate/data/run.py xanes_oxstate/cli.py Makefile
git commit -m "feat(cli): build_data orchestration + Makefile skeleton"
```

---

## Phase 2 — Model + training

### Task 10: 1D-CNN architecture

**Files:**
- Create: `xanes_oxstate/model/cnn.py`
- Create: `tests/test_cnn.py`

- [ ] **Step 1: Write failing test**

`tests/test_cnn.py`:

```python
import torch

from xanes_oxstate.model.cnn import OxStateCNN


def test_forward_shape():
    model = OxStateCNN(n_classes=5)
    x = torch.zeros(4, 1, 200)
    out = model(x)
    assert out.shape == (4, 5)


def test_param_count_under_200k():
    model = OxStateCNN(n_classes=5)
    n = sum(p.numel() for p in model.parameters())
    assert n < 200_000, f"too many params: {n}"


def test_supports_variable_n_classes():
    for k in (2, 3, 5, 8):
        out = OxStateCNN(n_classes=k)(torch.zeros(2, 1, 200))
        assert out.shape == (2, k)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cnn.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `cnn.py`**

`xanes_oxstate/model/cnn.py`:

```python
"""Small 1D-CNN classifier for fixed-length XANES inputs."""
from __future__ import annotations

import torch
import torch.nn as nn


class OxStateCNN(nn.Module):
    def __init__(self, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        return self.classifier(h)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_cnn.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add xanes_oxstate/model/cnn.py tests/test_cnn.py
git commit -m "feat(model): 1D-CNN architecture"
```

---

### Task 11: Single-seed training loop

**Files:**
- Create: `xanes_oxstate/model/train.py`
- Create: `tests/test_train.py`

- [ ] **Step 1: Write failing test**

`tests/test_train.py`:

```python
import pandas as pd

from xanes_oxstate.model.dataset import XanesDataset
from xanes_oxstate.model.train import train_one_seed


def _records(n_per_class=15, classes=(2, 3, 4)):
    import numpy as np
    rng = np.random.default_rng(0)
    e0 = 6539.0
    recs = []
    for ci, c in enumerate(classes):
        for k in range(n_per_class):
            e = np.linspace(e0 - 30, e0 + 70, 400)
            shift = ci * 1.5
            step = 1.0 / (1.0 + np.exp(-(e - (e0 + shift)) / 1.0))
            recs.append({
                "mp_id": f"mp-{ci}-{k}",
                "element": "Mn",
                "ox_state": c,
                "formula": f"Mn_{ci}_{k % 4}",
                "energies": e.tolist(),
                "intensities": (step + 0.01 * rng.standard_normal(400)).tolist(),
            })
    return recs


def test_training_loss_decreases(tmp_path):
    train_recs = _records()
    val_recs = _records(n_per_class=8)
    train_ds = XanesDataset(train_recs)
    val_ds = XanesDataset(val_recs)

    out = train_one_seed(
        train_ds, val_ds,
        epochs=5, batch_size=16, lr=1e-3, seed=0,
        ckpt_path=tmp_path / "ckpt.pt",
    )
    assert out.history["train_loss"][-1] < out.history["train_loss"][0]
    assert (tmp_path / "ckpt.pt").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `train.py`**

`xanes_oxstate/model/train.py`:

```python
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
    counts = Counter(int(ds[i][1]) for i in range(len(ds)))
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
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_train.py -v`
Expected: 1 passed (takes ~30 s).

- [ ] **Step 5: Commit**

```bash
git add xanes_oxstate/model/train.py tests/test_train.py
git commit -m "feat(model): single-seed training loop"
```

---

### Task 12: Ensemble training driver

**Files:**
- Create: `xanes_oxstate/model/ensemble.py`
- Create: `tests/test_ensemble.py`

- [ ] **Step 1: Write failing test**

`tests/test_ensemble.py`:

```python
import torch

from xanes_oxstate.model.dataset import XanesDataset
from xanes_oxstate.model.ensemble import train_ensemble, load_ensemble


def _records():
    import numpy as np
    rng = np.random.default_rng(0)
    e0 = 6539.0
    recs = []
    for ci, c in enumerate((2, 3, 4)):
        for k in range(20):
            e = np.linspace(e0 - 30, e0 + 70, 400)
            step = 1.0 / (1.0 + np.exp(-(e - (e0 + 1.5 * ci))))
            recs.append({
                "mp_id": f"mp-{ci}-{k}",
                "element": "Mn",
                "ox_state": c,
                "formula": f"Mn_{ci}_{k % 4}",
                "energies": e.tolist(),
                "intensities": (step + 0.01 * rng.standard_normal(400)).tolist(),
            })
    return recs


def test_train_ensemble_writes_checkpoints(tmp_path):
    train_ds = XanesDataset(_records())
    val_ds = XanesDataset(_records())
    paths = train_ensemble(
        train_ds, val_ds, ckpt_dir=tmp_path,
        seeds=(0, 1), epochs=3, batch_size=16,
    )
    assert len(paths) == 2
    for p in paths:
        assert p.exists()


def test_load_ensemble_returns_models(tmp_path):
    train_ds = XanesDataset(_records())
    val_ds = XanesDataset(_records())
    train_ensemble(
        train_ds, val_ds, ckpt_dir=tmp_path,
        seeds=(0, 1), epochs=3, batch_size=16,
    )
    models = load_ensemble(tmp_path, element="Mn")
    assert len(models) == 2
    x = torch.zeros(2, 1, 200)
    for m in models:
        assert m(x).shape == (2, train_ds.n_classes)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ensemble.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `ensemble.py`**

`xanes_oxstate/model/ensemble.py`:

```python
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
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_ensemble.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add xanes_oxstate/model/ensemble.py tests/test_ensemble.py
git commit -m "feat(model): ensemble training + loading"
```

---

## Phase 3 — Evaluation and baselines

### Task 13: Core metrics

**Files:**
- Create: `xanes_oxstate/eval/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write failing test**

`tests/test_metrics.py`:

```python
import numpy as np

from xanes_oxstate.eval.metrics import (
    top1_accuracy,
    per_class_f1,
    ensemble_disagreement,
)


def test_top1_accuracy():
    y_true = np.array([0, 1, 2, 2, 0])
    y_pred = np.array([0, 1, 2, 1, 0])
    assert top1_accuracy(y_true, y_pred) == 0.8


def test_per_class_f1_returns_one_per_class():
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 2])
    f1 = per_class_f1(y_true, y_pred, n_classes=3)
    assert len(f1) == 3
    assert all(0.0 <= v <= 1.0 for v in f1)


def test_ensemble_disagreement_is_zero_for_identical_preds():
    preds = np.stack([np.array([0, 1, 2])] * 3)
    assert ensemble_disagreement(preds) == 0.0


def test_ensemble_disagreement_is_one_for_max_disagreement():
    preds = np.array([[0, 0], [1, 1], [2, 2]])
    # Every sample has 3 distinct predictions across 3 models.
    assert ensemble_disagreement(preds) == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metrics.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `metrics.py`**

`xanes_oxstate/eval/metrics.py`:

```python
"""Classification metrics for the XANES oxidation-state task."""
from __future__ import annotations

import numpy as np


def top1_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> list[float]:
    out = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fp == 0 or tp + fn == 0:
            out.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        out.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
    return out


def ensemble_disagreement(preds: np.ndarray) -> float:
    """preds: [n_models, n_samples] integer class indices."""
    n_models, n_samples = preds.shape
    n_unique = np.array(
        [len(np.unique(preds[:, i])) for i in range(n_samples)]
    )
    # Normalize to [0, 1]: 1 model agreement -> 0; n_models distinct -> 1.
    return float((n_unique - 1).mean() / (n_models - 1))
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_metrics.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add xanes_oxstate/eval/metrics.py tests/test_metrics.py
git commit -m "feat(eval): top-1 accuracy + per-class F1 + disagreement"
```

---

### Task 14: Temperature scaling + ECE

**Files:**
- Create: `xanes_oxstate/eval/calibration.py`
- Create: `tests/test_calibration.py`

- [ ] **Step 1: Write failing test**

`tests/test_calibration.py`:

```python
import numpy as np

from xanes_oxstate.eval.calibration import (
    fit_temperature,
    apply_temperature,
    expected_calibration_error,
)


def _miscalibrated_logits(n=1000, n_classes=3, seed=0, scale=5.0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, n)
    logits = rng.standard_normal((n, n_classes))
    # Push correct class up.
    logits[np.arange(n), y] += 2.0
    return scale * logits, y


def test_fit_temperature_returns_scalar_above_zero():
    logits, y = _miscalibrated_logits()
    T = fit_temperature(logits, y)
    assert T > 0


def test_temperature_lowers_ece_on_overconfident_logits():
    logits, y = _miscalibrated_logits(scale=5.0)
    probs = _softmax(logits)
    ece_before = expected_calibration_error(probs, y)
    T = fit_temperature(logits, y)
    probs_after = _softmax(apply_temperature(logits, T))
    ece_after = expected_calibration_error(probs_after, y)
    assert ece_after < ece_before


def _softmax(x):
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_calibration.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `calibration.py`**

`xanes_oxstate/eval/calibration.py`:

```python
"""Temperature scaling and ECE (Guo et al. 2017)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def _log_softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def _nll(T: float, logits: np.ndarray, y: np.ndarray) -> float:
    log_p = _log_softmax(logits / T)
    return float(-log_p[np.arange(y.size), y].mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    res = minimize_scalar(_nll, args=(logits, y), bounds=(0.05, 10.0),
                          method="bounded")
    return float(res.x)


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    return logits / T


def expected_calibration_error(
    probs: np.ndarray, y: np.ndarray, n_bins: int = 15
) -> float:
    conf = probs.max(axis=-1)
    pred = probs.argmax(axis=-1)
    correct = (pred == y).astype(np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = probs.shape[0]
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if not mask.any():
            continue
        bin_acc = correct[mask].mean()
        bin_conf = conf[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_calibration.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add xanes_oxstate/eval/calibration.py tests/test_calibration.py
git commit -m "feat(eval): temperature scaling + ECE"
```

---

### Task 15: Confusion matrix builder

**Files:**
- Create: `xanes_oxstate/eval/confusion.py`
- Create: `tests/test_confusion.py`

- [ ] **Step 1: Write failing test**

`tests/test_confusion.py`:

```python
import numpy as np

from xanes_oxstate.eval.confusion import build_confusion, find_confused_pairs


def test_build_confusion_row_normalized():
    y_true = np.array([0, 0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 0, 1, 1, 2])
    cm = build_confusion(y_true, y_pred, n_classes=3)
    # Row-normalized: each row sums to 1
    assert np.allclose(cm.sum(axis=1), 1.0)
    assert cm[0, 0] == 2 / 3
    assert cm[0, 1] == 1 / 3


def test_find_confused_pairs_above_threshold():
    y_true = np.array([0] * 10 + [1] * 10)
    y_pred = np.array([0] * 7 + [1] * 3 + [1] * 8 + [0] * 2)
    pairs = find_confused_pairs(y_true, y_pred, n_classes=2, threshold=0.2)
    # Class 0 → predicted 1 = 30%, class 1 → predicted 0 = 20%
    assert (0, 1) in [(p.true, p.pred) for p in pairs]
    assert (1, 0) in [(p.true, p.pred) for p in pairs]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_confusion.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `confusion.py`**

`xanes_oxstate/eval/confusion.py`:

```python
"""Row-normalized confusion + confused-pair extraction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConfusedPair:
    true: int
    pred: int
    rate: float
    count: int


def build_confusion(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return cm / row_sums


def find_confused_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    threshold: float = 0.20,
) -> list[ConfusedPair]:
    cm_norm = build_confusion(y_true, y_pred, n_classes)
    counts = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        counts[int(t), int(p)] += 1

    pairs: list[ConfusedPair] = []
    for t in range(n_classes):
        for p in range(n_classes):
            if t == p:
                continue
            if cm_norm[t, p] >= threshold:
                pairs.append(ConfusedPair(t, p, float(cm_norm[t, p]),
                                          int(counts[t, p])))
    pairs.sort(key=lambda x: -x.rate)
    return pairs
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_confusion.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add xanes_oxstate/eval/confusion.py tests/test_confusion.py
git commit -m "feat(eval): row-normalized confusion + confused-pair finder"
```

---

### Task 16: Ensemble predictor + evaluator

**Files:**
- Create: `xanes_oxstate/eval/predict.py`
- Create: `tests/test_predict.py`

- [ ] **Step 1: Write failing test**

`tests/test_predict.py`:

```python
import numpy as np
import torch

from xanes_oxstate.model.cnn import OxStateCNN
from xanes_oxstate.eval.predict import ensemble_predict_proba


class _IdentityCNN(OxStateCNN):
    def __init__(self, n_classes=3, target_class=0):
        super().__init__(n_classes=n_classes)
        self._tc = target_class

    def forward(self, x):
        # Always favor the target class
        out = torch.zeros(x.shape[0], 3)
        out[:, self._tc] = 5.0
        return out


def test_ensemble_predict_proba_averages_probs():
    models = [_IdentityCNN(target_class=0), _IdentityCNN(target_class=1)]
    x = torch.zeros(4, 1, 200)
    probs = ensemble_predict_proba(models, x)
    assert probs.shape == (4, 3)
    # Two equally-favored classes after averaging
    assert np.isclose(probs[0, 0], probs[0, 1])
    assert np.allclose(probs.sum(axis=1), 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_predict.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `predict.py`**

`xanes_oxstate/eval/predict.py`:

```python
"""Ensemble inference: average softmax probabilities."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..model.dataset import XanesDataset


def ensemble_predict_proba(
    models: list[torch.nn.Module], x: torch.Tensor
) -> np.ndarray:
    if not models:
        raise ValueError("no models supplied")
    probs = []
    with torch.no_grad():
        for m in models:
            m.eval()
            logits = m(x)
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.stack(probs).mean(axis=0)


def ensemble_predict_dataset(
    models: list[torch.nn.Module],
    dataset: XanesDataset,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_probs, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            all_probs.append(ensemble_predict_proba(models, x))
            all_y.append(np.asarray(y))
    return np.concatenate(all_probs), np.concatenate(all_y)


def ensemble_logits_dataset(
    models: list[torch.nn.Module],
    dataset: XanesDataset,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns averaged logits (not softmax). Used for temperature fitting."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_logits, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            l = []
            for m in models:
                m.eval()
                l.append(m(x).cpu().numpy())
            all_logits.append(np.stack(l).mean(axis=0))
            all_y.append(np.asarray(y))
    return np.concatenate(all_logits), np.concatenate(all_y)
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_predict.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add xanes_oxstate/eval/predict.py tests/test_predict.py
git commit -m "feat(eval): ensemble inference utilities"
```

---

### Task 17: Majority-class baseline

**Files:**
- Create: `xanes_oxstate/baselines/majority.py`
- Create: `tests/test_majority.py`

- [ ] **Step 1: Write failing test**

`tests/test_majority.py`:

```python
import numpy as np

from xanes_oxstate.baselines.majority import MajorityClassifier


def test_majority_predicts_most_common():
    y_train = np.array([0, 0, 0, 1, 2])
    clf = MajorityClassifier().fit(y_train)
    assert clf.predict(np.zeros(4)) == [0, 0, 0, 0]
    assert clf.majority_class == 0


def test_majority_score_matches_proportion():
    y_train = np.array([0, 0, 1, 1, 1])
    clf = MajorityClassifier().fit(y_train)
    acc = clf.score(np.zeros(5), np.array([1, 1, 1, 0, 0]))
    assert acc == 0.6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_majority.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `majority.py`**

`xanes_oxstate/baselines/majority.py`:

```python
"""Predict the most-common training class. Sanity floor."""
from __future__ import annotations

from collections import Counter

import numpy as np


class MajorityClassifier:
    def __init__(self) -> None:
        self.majority_class: int | None = None

    def fit(self, y: np.ndarray) -> "MajorityClassifier":
        self.majority_class = int(Counter(map(int, y)).most_common(1)[0][0])
        return self

    def predict(self, X) -> list[int]:
        if self.majority_class is None:
            raise RuntimeError("call .fit first")
        n = len(X)
        return [self.majority_class] * n

    def score(self, X, y_true: np.ndarray) -> float:
        return float((np.asarray(self.predict(X)) == y_true).mean())
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_majority.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add xanes_oxstate/baselines/majority.py tests/test_majority.py
git commit -m "feat(baselines): majority-class predictor"
```

---

### Task 18: Hand-feature extractor

**Files:**
- Create: `xanes_oxstate/baselines/features.py`
- Create: `tests/test_features.py`

- [ ] **Step 1: Write failing test**

`tests/test_features.py`:

```python
import numpy as np

from xanes_oxstate.baselines.features import extract_features


def _norm_step(e0=6539.0, n=200, shift=0.0, height=1.0):
    e = np.linspace(e0 - 10, e0 + 40, n)
    step = height / (1.0 + np.exp(-(e - (e0 + shift))))
    bump = 0.3 * np.exp(-((e - (e0 + 2)) ** 2) / 4.0)
    return e, step + bump


def test_feature_vector_length():
    e, y = _norm_step()
    f = extract_features(e, y, e0=6539.0)
    assert len(f) == 6


def test_pre_edge_area_is_nonzero_when_bump_present():
    e, y = _norm_step()
    f = extract_features(e, y, e0=6539.0)
    assert f[0] > 0


def test_edge_position_tracks_shift():
    e, y1 = _norm_step(shift=0.0)
    _, y2 = _norm_step(shift=2.0)
    f1 = extract_features(e, y1, e0=6539.0)
    f2 = extract_features(e, y2, e0=6539.0)
    # f[2] is edge inflection energy relative to E0
    assert f2[2] > f1[2]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `features.py`**

`xanes_oxstate/baselines/features.py`:

```python
"""Hand-engineered features for the GBDT baseline.

Six features per normalized spectrum:
    0: pre-edge area               (-10 to -1 eV)
    1: pre-edge peak height        (max in [-10, -1])
    2: edge inflection position    (eV relative to E0)
    3: white-line height           (max in [0, 10])
    4: white-line position         (eV relative to E0)
    5: post-edge slope             (linear fit on [20, 40])
"""
from __future__ import annotations

import numpy as np


def extract_features(
    energy: np.ndarray, intensity: np.ndarray, e0: float
) -> np.ndarray:
    rel = energy - e0
    out = np.zeros(6, dtype=np.float64)

    pre_mask = (rel >= -10) & (rel <= -1)
    if pre_mask.any():
        out[0] = float(np.trapezoid(intensity[pre_mask], rel[pre_mask]))
        out[1] = float(intensity[pre_mask].max())

    edge_mask = (rel >= -5) & (rel <= 10)
    if edge_mask.sum() >= 3:
        dy = np.gradient(intensity[edge_mask], rel[edge_mask])
        out[2] = float(rel[edge_mask][np.argmax(dy)])

    wl_mask = (rel >= 0) & (rel <= 10)
    if wl_mask.any():
        out[3] = float(intensity[wl_mask].max())
        out[4] = float(rel[wl_mask][np.argmax(intensity[wl_mask])])

    post_mask = (rel >= 20) & (rel <= 40)
    if post_mask.sum() >= 2:
        a, _ = np.polyfit(rel[post_mask], intensity[post_mask], 1)
        out[5] = float(a)

    return out
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_features.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add xanes_oxstate/baselines/features.py tests/test_features.py
git commit -m "feat(baselines): hand-feature extractor for GBDT"
```

---

### Task 19: GBDT baseline wrapper

**Files:**
- Create: `xanes_oxstate/baselines/gbdt.py`
- Create: `tests/test_gbdt.py`

- [ ] **Step 1: Write failing test**

`tests/test_gbdt.py`:

```python
import numpy as np

from xanes_oxstate.baselines.gbdt import GBDTBaseline


def test_gbdt_fits_and_predicts():
    rng = np.random.default_rng(0)
    n_per = 100
    X0 = rng.normal(loc=0, scale=0.3, size=(n_per, 6))
    X1 = rng.normal(loc=2, scale=0.3, size=(n_per, 6))
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per + [1] * n_per)

    clf = GBDTBaseline(n_classes=2, random_state=0).fit(X, y)
    acc = (clf.predict(X) == y).mean()
    assert acc > 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gbdt.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `gbdt.py`**

`xanes_oxstate/baselines/gbdt.py`:

```python
"""LightGBM wrapper over the hand-feature vector."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np


class GBDTBaseline:
    def __init__(self, n_classes: int, random_state: int = 0) -> None:
        self.n_classes = n_classes
        self.model = lgb.LGBMClassifier(
            objective="multiclass" if n_classes > 2 else "binary",
            num_class=n_classes if n_classes > 2 else 1,
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            verbose=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GBDTBaseline":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_gbdt.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add xanes_oxstate/baselines/gbdt.py tests/test_gbdt.py
git commit -m "feat(baselines): LightGBM wrapper on hand features"
```

---

## Phase 4 — Physics analysis

### Task 20: Spectral overlay generator

**Files:**
- Create: `xanes_oxstate/physics/analysis.py`
- Create: `tests/test_physics_overlay.py`

- [ ] **Step 1: Write failing test**

`tests/test_physics_overlay.py`:

```python
import numpy as np

from xanes_oxstate.physics.analysis import mean_spectrum_by_class


def test_mean_spectrum_per_class():
    energies = np.linspace(0, 1, 50)
    intens = [
        np.zeros(50), np.zeros(50),
        np.ones(50), np.ones(50),
    ]
    classes = [0, 0, 1, 1]
    means, envelopes = mean_spectrum_by_class(intens, classes, energies)
    assert means[0].shape == (50,)
    assert means[1].shape == (50,)
    assert np.allclose(means[0], 0)
    assert np.allclose(means[1], 1)
    assert envelopes[0]["lo"].shape == (50,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_physics_overlay.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `analysis.py` (first cut — overlay only)**

`xanes_oxstate/physics/analysis.py`:

```python
"""Confused-pair analysis: overlay, LR feature inspection, structural bins."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def mean_spectrum_by_class(
    spectra: list[np.ndarray],
    classes: list[int],
    energies: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, np.ndarray]]]:
    by_class: dict[int, list[np.ndarray]] = defaultdict(list)
    for s, c in zip(spectra, classes):
        by_class[int(c)].append(np.asarray(s))

    means: dict[int, np.ndarray] = {}
    envelopes: dict[int, dict[str, np.ndarray]] = {}
    for c, ys in by_class.items():
        arr = np.stack(ys)
        means[c] = arr.mean(axis=0)
        envelopes[c] = {
            "lo": np.percentile(arr, 5, axis=0),
            "hi": np.percentile(arr, 95, axis=0),
        }
    return means, envelopes
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_physics_overlay.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add xanes_oxstate/physics/analysis.py tests/test_physics_overlay.py
git commit -m "feat(physics): spectral overlay by class"
```

---

### Task 21: Pair-wise LR feature inspector

**Files:**
- Modify: `xanes_oxstate/physics/analysis.py`
- Create: `tests/test_physics_lr.py`

- [ ] **Step 1: Write failing test**

`tests/test_physics_lr.py`:

```python
import numpy as np

from xanes_oxstate.physics.analysis import (
    pairwise_logistic_coefficients,
)


def test_lr_finds_discriminative_region():
    rng = np.random.default_rng(0)
    n = 200
    grid = np.linspace(0, 1, 50)
    # Class A: peak at idx 10. Class B: peak at idx 40.
    X_a = np.zeros((n, 50))
    X_a[:, 10] = 1.0
    X_a += 0.05 * rng.standard_normal(X_a.shape)
    X_b = np.zeros((n, 50))
    X_b[:, 40] = 1.0
    X_b += 0.05 * rng.standard_normal(X_b.shape)
    X = np.vstack([X_a, X_b])
    y = np.array([0] * n + [1] * n)

    coefs = pairwise_logistic_coefficients(X, y)
    # The discriminative region should be near idx 10 (neg) and 40 (pos).
    assert coefs.shape == (50,)
    assert coefs[40] > coefs[20] > 0
    assert coefs[10] < 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_physics_lr.py -v`
Expected: ImportError.

- [ ] **Step 3: Append to `analysis.py`**

```python
from sklearn.linear_model import LogisticRegression


def pairwise_logistic_coefficients(
    X: np.ndarray, y: np.ndarray, C: float = 0.1
) -> np.ndarray:
    """L2-regularized binary LR; returns per-energy-point coefficients."""
    if set(map(int, np.unique(y))) != {0, 1}:
        raise ValueError("y must be binary (0/1) for pairwise LR")
    clf = LogisticRegression(C=C, max_iter=2000, penalty="l2")
    clf.fit(X, y)
    return clf.coef_.ravel()
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_physics_lr.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add xanes_oxstate/physics/analysis.py tests/test_physics_lr.py
git commit -m "feat(physics): pairwise LR feature inspector"
```

---

### Task 22: Structural correlate binning

**Files:**
- Modify: `xanes_oxstate/physics/analysis.py`
- Create: `tests/test_physics_structural.py`

This task is the lightest of the three since the structural correlate is just
a groupby. The physics work is in the writeup; the code is plumbing.

- [ ] **Step 1: Write failing test**

`tests/test_physics_structural.py`:

```python
from xanes_oxstate.physics.analysis import group_failures_by_field


def _failures():
    return [
        {"true": 0, "pred": 1, "coord": 4, "nn_elem": "O"},
        {"true": 0, "pred": 1, "coord": 4, "nn_elem": "O"},
        {"true": 0, "pred": 1, "coord": 6, "nn_elem": "S"},
        {"true": 0, "pred": 0, "coord": 6, "nn_elem": "O"},
    ]


def test_group_failures_counts_by_field():
    out = group_failures_by_field(_failures(), field="coord",
                                  true_class=0, pred_class=1)
    assert out == {4: 2, 6: 1}


def test_group_failures_handles_missing_field():
    out = group_failures_by_field(
        [{"true": 0, "pred": 1}], field="coord",
        true_class=0, pred_class=1,
    )
    assert out == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_physics_structural.py -v`
Expected: ImportError.

- [ ] **Step 3: Append to `analysis.py`**

```python
from collections import Counter


def group_failures_by_field(
    failures: list[dict],
    field: str,
    true_class: int,
    pred_class: int,
) -> dict:
    matched = [
        f for f in failures
        if f.get("true") == true_class and f.get("pred") == pred_class
        and field in f
    ]
    return dict(Counter(f[field] for f in matched))
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_physics_structural.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add xanes_oxstate/physics/analysis.py tests/test_physics_structural.py
git commit -m "feat(physics): structural correlate grouping"
```

---

## Phase 5 — Plots, CLI, and writeup

### Task 23: Plots — per-element accuracy bar chart

**Files:**
- Create: `xanes_oxstate/eval/plots.py`
- Create: `tests/test_plots.py`

- [ ] **Step 1: Write failing test**

`tests/test_plots.py`:

```python
import matplotlib
matplotlib.use("Agg")

from xanes_oxstate.eval.plots import per_element_accuracy_bar


def test_per_element_accuracy_bar_returns_figure(tmp_path):
    results = {
        "Mn": {"majority": 0.4, "gbdt": 0.7, "cnn": 0.85},
        "Fe": {"majority": 0.5, "gbdt": 0.75, "cnn": 0.88},
    }
    errs = {
        "Mn": {"cnn": 0.02},
        "Fe": {"cnn": 0.015},
    }
    out = tmp_path / "acc.png"
    fig = per_element_accuracy_bar(results, errs=errs, out_path=out)
    assert out.exists()
    assert fig.axes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_plots.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `plots.py` (first cut)**

`xanes_oxstate/eval/plots.py`:

```python
"""Headline figures for the README."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def per_element_accuracy_bar(
    results: dict[str, dict[str, float]],
    errs: dict[str, dict[str, float]] | None = None,
    out_path: Path | None = None,
):
    elements = list(results.keys())
    estimators = ["majority", "gbdt", "cnn"]
    x = np.arange(len(elements))
    w = 0.25

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, est in enumerate(estimators):
        vals = [results[e].get(est, 0.0) for e in elements]
        yerr = (
            [errs.get(e, {}).get(est, 0.0) for e in elements]
            if errs else None
        )
        ax.bar(x + (i - 1) * w, vals, w, yerr=yerr, label=est, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(elements)
    ax.axhline(0.75, color="grey", linestyle="--", linewidth=0.8)
    ax.axhline(0.85, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("top-1 accuracy")
    ax.set_title("Per-element accuracy: majority vs. GBDT vs. CNN ensemble")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)
    return fig
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_plots.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add xanes_oxstate/eval/plots.py tests/test_plots.py
git commit -m "feat(plots): per-element accuracy bar chart"
```

---

### Task 24: Plots — confusion matrix small multiples

**Files:**
- Modify: `xanes_oxstate/eval/plots.py`
- Modify: `tests/test_plots.py`

- [ ] **Step 1: Append failing test**

```python
import numpy as np
from xanes_oxstate.eval.plots import confusion_small_multiples


def test_confusion_small_multiples_writes_file(tmp_path):
    cms = {
        "Mn": (np.eye(3) * 0.9 + 0.05, [2, 3, 4]),
        "Fe": (np.eye(2) * 0.95 + 0.05, [2, 3]),
    }
    out = tmp_path / "cm.png"
    fig = confusion_small_multiples(cms, n_cols=2, out_path=out)
    assert out.exists()
    assert len(fig.axes) >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_plots.py::test_confusion_small_multiples_writes_file -v`
Expected: AttributeError (no `confusion_small_multiples`).

- [ ] **Step 3: Append to `plots.py`**

```python
def confusion_small_multiples(
    matrices: dict[str, tuple[np.ndarray, list[int]]],
    n_cols: int = 4,
    highlight_threshold: float = 0.20,
    out_path: Path | None = None,
):
    elements = list(matrices.keys())
    n_rows = (len(elements) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=False
    )

    for k, elem in enumerate(elements):
        cm, labels = matrices[elem]
        r, c = divmod(k, n_cols)
        ax = axes[r][c]
        im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:.2f}",
                        ha="center", va="center",
                        color="white" if cm[i, j] < 0.5 else "black",
                        fontsize=7)
                if i != j and cm[i, j] >= highlight_threshold:
                    ax.add_patch(
                        plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor="red",
                                       linewidth=1.5)
                    )
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(elem)

    # Hide unused axes
    for k in range(len(elements), n_rows * n_cols):
        r, c = divmod(k, n_cols)
        axes[r][c].axis("off")

    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)
    return fig
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_plots.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add xanes_oxstate/eval/plots.py tests/test_plots.py
git commit -m "feat(plots): confusion matrix small multiples"
```

---

### Task 25: Top-level evaluate pipeline + CLI

**Files:**
- Create: `xanes_oxstate/eval/run.py`
- Modify: `xanes_oxstate/cli.py`
- Modify: `Makefile`

- [ ] **Step 1: Write `xanes_oxstate/eval/run.py`**

```python
"""End-to-end per-element evaluation: train ensemble + baselines, report metrics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..baselines.features import extract_features
from ..baselines.gbdt import GBDTBaseline
from ..baselines.majority import MajorityClassifier
from ..data.preprocess import preprocess
from ..model.dataset import XanesDataset
from ..model.ensemble import load_ensemble, train_ensemble
from .calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
)
from .confusion import build_confusion, find_confused_pairs
from .metrics import top1_accuracy, per_class_f1, ensemble_disagreement
from .predict import ensemble_logits_dataset, ensemble_predict_dataset


def _records_from_parquet(path: Path) -> list[dict]:
    return pd.read_parquet(path).to_dict("records")


def _hand_feature_matrix(records: list[dict]) -> np.ndarray:
    feats = []
    for rec in records:
        e = np.asarray(rec["energies"], dtype=np.float64)
        y = np.asarray(rec["intensities"], dtype=np.float64)
        try:
            grid, y_norm = preprocess(e, y)
            from ..data.preprocess import detect_e0
            e0 = detect_e0(e, y)
            feats.append(extract_features(grid, y_norm, e0=e0))
        except ValueError:
            feats.append(np.zeros(6))
    return np.vstack(feats)


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

    X_train = _hand_feature_matrix(train_recs)
    X_test = _hand_feature_matrix(test_recs)
    y_train = np.array([train_ds._ox_to_class[r["ox_state"]] for r in train_recs])
    gbdt = GBDTBaseline(n_classes=train_ds.n_classes, random_state=0)
    gbdt.fit(X_train, y_train)
    gbdt_pred = gbdt.predict(X_test)
    gbdt_acc = top1_accuracy(test_y, gbdt_pred)

    cm = build_confusion(test_y, test_pred, n_classes=train_ds.n_classes)
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
```

- [ ] **Step 2: Extend `cli.py`**

Replace `cli.py` with:

```python
"""Entry points: python -m xanes_oxstate.cli."""
from __future__ import annotations

import argparse
from pathlib import Path


def cmd_build_data(args) -> None:
    from .data.run import build_element_dataset
    paths = build_element_dataset(
        args.element,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        report_dir=args.report_dir,
        seed=args.seed,
    )
    print(f"[{args.element}] {paths}")


def cmd_evaluate(args) -> None:
    from .eval.run import evaluate_element
    metrics = evaluate_element(
        args.element,
        processed_dir=args.processed_dir,
        ckpt_dir=args.ckpt_dir,
        metrics_dir=args.metrics_dir,
        epochs=args.epochs,
    )
    print(f"[{args.element}] CNN acc = {metrics['accuracy']['cnn']:.3f}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m xanes_oxstate.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    bp = sub.add_parser("build-data")
    bp.add_argument("--element", required=True)
    bp.add_argument("--raw-dir", default="data/raw", type=Path)
    bp.add_argument("--processed-dir", default="data/processed", type=Path)
    bp.add_argument("--report-dir", default="data/reports", type=Path)
    bp.add_argument("--seed", default=42, type=int)
    bp.set_defaults(func=cmd_build_data)

    ep = sub.add_parser("evaluate")
    ep.add_argument("--element", required=True)
    ep.add_argument("--processed-dir", default="data/processed", type=Path)
    ep.add_argument("--ckpt-dir", default="checkpoints", type=Path)
    ep.add_argument("--metrics-dir", default="metrics", type=Path)
    ep.add_argument("--epochs", default=50, type=int)
    ep.set_defaults(func=cmd_evaluate)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Update `Makefile`**

Replace with:

```makefile
ELEMENTS = Ti V Cr Mn Fe Co Ni Cu

.PHONY: data
data:
	@for e in $(ELEMENTS); do \
	  python -m xanes_oxstate.cli build-data --element $$e; \
	done

.PHONY: train
train:
	@for e in $(ELEMENTS); do \
	  python -m xanes_oxstate.cli evaluate --element $$e; \
	done

.PHONY: figures
figures:
	python scripts/make_figures.py

.PHONY: all
all: data train figures

.PHONY: test
test:
	pytest -q
```

- [ ] **Step 4: Smoke test the CLI surface**

Run: `python -m xanes_oxstate.cli --help`
Expected: shows `build-data` and `evaluate` subcommands.

- [ ] **Step 5: Commit**

```bash
git add xanes_oxstate/eval/run.py xanes_oxstate/cli.py Makefile
git commit -m "feat(cli): evaluate subcommand + per-element pipeline"
```

---

### Task 26: Headline figure script

**Files:**
- Create: `scripts/make_figures.py`

- [ ] **Step 1: Write `scripts/make_figures.py`**

```python
"""Aggregate per-element metrics → headline figures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from xanes_oxstate.eval.plots import (
    confusion_small_multiples,
    per_element_accuracy_bar,
)


ELEMENTS = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"]


def main() -> None:
    metrics_dir = Path("metrics")
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)

    results, errs, matrices = {}, {}, {}
    for elem in ELEMENTS:
        path = metrics_dir / f"{elem}.json"
        if not path.exists():
            print(f"  skip {elem}: no metrics")
            continue
        m = json.loads(path.read_text())
        results[elem] = m["accuracy"]
        # Ensemble disagreement as a rough error bar
        errs[elem] = {"cnn": float(m.get("ensemble_disagreement", 0.0)) / 2}

        cm = np.load(metrics_dir / f"{elem}_cm.npy")
        labels = [m["class_to_ox_state"][str(i)] for i in range(cm.shape[0])]
        matrices[elem] = (cm, labels)

    if results:
        per_element_accuracy_bar(
            results, errs=errs,
            out_path=fig_dir / "per_element_accuracy.png",
        )
        per_element_accuracy_bar(
            results, errs=errs,
            out_path=fig_dir / "per_element_accuracy.pdf",
        )
    if matrices:
        confusion_small_multiples(
            matrices, n_cols=4,
            out_path=fig_dir / "confusions.png",
        )
        confusion_small_multiples(
            matrices, n_cols=4,
            out_path=fig_dir / "confusions.pdf",
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test by stubbing inputs**

```bash
mkdir -p metrics figures
python -c "
import json, numpy as np
from pathlib import Path
Path('metrics/Mn.json').write_text(json.dumps({
    'element': 'Mn',
    'accuracy': {'majority': 0.4, 'gbdt': 0.7, 'cnn': 0.85},
    'class_to_ox_state': {'0': 2, '1': 3, '2': 4},
    'ensemble_disagreement': 0.05,
}))
np.save('metrics/Mn_cm.npy', np.eye(3) * 0.9 + 0.05)
"
python scripts/make_figures.py
ls figures/
```

Expected: `per_element_accuracy.{png,pdf}` and `confusions.{png,pdf}` exist (built from the single Mn stub).

- [ ] **Step 3: Clean up stubs**

```bash
rm -f metrics/Mn.json metrics/Mn_cm.npy
```

- [ ] **Step 4: Commit**

```bash
git add scripts/make_figures.py
git commit -m "feat(figures): aggregator script for headline plots"
```

---

### Task 27: End-to-end smoke test on synthetic data

**Files:**
- Create: `tests/test_smoke.py`

This is the safety net: it verifies the whole pipeline runs on a tiny
synthetic dataset without touching MP API or training for long.

- [ ] **Step 1: Write the smoke test**

`tests/test_smoke.py`:

```python
import json
from pathlib import Path

import numpy as np
import pandas as pd

from xanes_oxstate.eval.run import evaluate_element
from xanes_oxstate.model.dataset import XanesDataset


def _synthetic_records(n_per_class=20, classes=(2, 3, 4)):
    rng = np.random.default_rng(0)
    e0 = 6539.0
    recs = []
    for ci, c in enumerate(classes):
        for k in range(n_per_class):
            e = np.linspace(e0 - 30, e0 + 70, 400)
            shift = ci * 1.5
            step = 1.0 / (1.0 + np.exp(-(e - (e0 + shift)) / 1.0))
            recs.append({
                "mp_id": f"mp-{ci}-{k}",
                "element": "Mn",
                "ox_state": c,
                "formula": f"Mn_{ci}_{k % 4}",
                "energies": e.tolist(),
                "intensities": (step + 0.01 * rng.standard_normal(400)).tolist(),
            })
    return recs


def _write_parquet_splits(records, processed_dir: Path):
    from xanes_oxstate.data.split import split_records
    splits = split_records(records, seed=42)
    for name, recs in splits.items():
        pd.DataFrame(recs).to_parquet(processed_dir / f"Mn_{name}.parquet")


def test_end_to_end_smoke(tmp_path):
    records = _synthetic_records()
    processed = tmp_path / "processed"
    ckpts = tmp_path / "ckpts"
    metrics = tmp_path / "metrics"
    processed.mkdir()

    _write_parquet_splits(records, processed)

    out = evaluate_element(
        "Mn",
        processed_dir=processed,
        ckpt_dir=ckpts,
        metrics_dir=metrics,
        seeds=(0, 1),
        epochs=3,
    )
    assert out["element"] == "Mn"
    assert 0 <= out["accuracy"]["cnn"] <= 1
    assert (metrics / "Mn.json").exists()
    assert (metrics / "Mn_cm.npy").exists()
```

- [ ] **Step 2: Run the smoke test**

Run: `pytest tests/test_smoke.py -v`
Expected: 1 passed (takes ~60–90 s).

- [ ] **Step 3: Commit**

```bash
git add tests/test_smoke.py
git commit -m "test: end-to-end smoke on synthetic data"
```

---

### Task 28: Notebook walkthrough

**Files:**
- Create: `notebooks/xanes_oxstate.ipynb`

The notebook is a thin wrapper that calls the package, executes a single
element end-to-end, and renders the headline figure inline. The point is
narrative, not new logic.

- [ ] **Step 1: Create `notebooks/xanes_oxstate.ipynb`**

Use `jupyter nbconvert` from a generated script (so the notebook lives in
git as a clean structured file):

```bash
cat > /tmp/build_nb.py <<'PY'
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""\
# XANES → oxidation state — end-to-end walkthrough

This notebook reproduces the headline result for Mn end-to-end:
fetch → clean → split → train ensemble → temperature-scale → evaluate
→ headline figure. Other elements follow the same recipe via the CLI.

Requires `MP_API_KEY` in the environment to fetch fresh data; if the
cached parquet already exists under `data/processed/`, fetching is
skipped.
"""))

cells.append(nbf.v4.new_code_cell("""\
import os
from pathlib import Path

from xanes_oxstate.data.run import build_element_dataset
from xanes_oxstate.eval.run import evaluate_element

ELEMENT = "Mn"
ROOT = Path.cwd().parent

paths = build_element_dataset(
    ELEMENT,
    raw_dir=ROOT / "data" / "raw",
    processed_dir=ROOT / "data" / "processed",
    report_dir=ROOT / "data" / "reports",
)
paths
"""))

cells.append(nbf.v4.new_code_cell("""\
metrics = evaluate_element(
    ELEMENT,
    processed_dir=ROOT / "data" / "processed",
    ckpt_dir=ROOT / "checkpoints",
    metrics_dir=ROOT / "metrics",
    epochs=50,
)
metrics["accuracy"]
"""))

cells.append(nbf.v4.new_code_cell("""\
import json
import numpy as np
from xanes_oxstate.eval.plots import (
    per_element_accuracy_bar,
    confusion_small_multiples,
)

# Single-element headline view (run scripts/make_figures.py once all 8 are
# done to get the full bar chart).
acc = metrics["accuracy"]
labels = [metrics["class_to_ox_state"][str(i)]
          for i in range(len(metrics["class_to_ox_state"]))]
cm = np.load(ROOT / "metrics" / f"{ELEMENT}_cm.npy")

per_element_accuracy_bar({ELEMENT: acc})
confusion_small_multiples({ELEMENT: (cm, labels)}, n_cols=1)
"""))

cells.append(nbf.v4.new_markdown_cell("""\
## Physics findings

Use `xanes_oxstate.physics.analysis` to inspect each entry in
`metrics["confused_pairs"]` — overlay mean spectra, fit a pairwise
logistic regression to find the discriminative spectral region, and
group failures by coordination number to look for structural patterns.
Write up the result in `docs/physics_findings.md`.
"""))

nb.cells = cells
nbf.write(nb, "notebooks/xanes_oxstate.ipynb")
PY
mkdir -p notebooks
python /tmp/build_nb.py
```

- [ ] **Step 2: Verify the notebook is loadable**

Run: `jupyter nbconvert --to script notebooks/xanes_oxstate.ipynb --stdout | head`
Expected: script content prints (notebook is well-formed).

- [ ] **Step 3: Commit**

```bash
git add notebooks/xanes_oxstate.ipynb
git commit -m "docs: end-to-end notebook walkthrough"
```

---

### Task 29: README

**Files:**
- Create: `README.md`
- Create: `docs/data_card.md` (skeleton)
- Create: `docs/methods.md` (skeleton)
- Create: `docs/physics_findings.md` (skeleton)

- [ ] **Step 1: Write `README.md`**

```markdown
# xanes-oxstate

Oxidation-state classification from K-edge XANES spectra across eight
3d transition metals (Ti, V, Cr, Mn, Fe, Co, Ni, Cu), trained on the
public Materials Project XAS database with a small 1D-CNN.

![per-element accuracy](figures/per_element_accuracy.png)

## Result summary

| Estimator | Overall accuracy (mean over 8 elements) |
|---|---|
| Majority-class baseline | _filled in after run_ |
| GBDT on hand features | _filled in after run_ |
| CNN ensemble (5 seeds, temp-scaled) | _filled in after run_ |

See [`docs/physics_findings.md`](docs/physics_findings.md) for the
analysis of which (element, oxidation-state pair) cases are spectrally
indistinguishable, and why.

## Reproduce

```bash
git clone <repo-url>
cd xanes-oxstate
pip install -e ".[dev]"
export MP_API_KEY=<your_key>
make all
```

`make all` runs `data → train → figures` end-to-end. Wall time on a
laptop: ~2 hours.

## Layout

- `xanes_oxstate/` — the package (data, model, eval, baselines, physics)
- `notebooks/xanes_oxstate.ipynb` — end-to-end walkthrough for one element
- `configs/<element>.yaml` — per-element hyperparameters
- `docs/` — data card, methods, physics findings

## Acknowledgements

XANES spectra obtained from the Materials Project (CC-BY 4.0).
```

- [ ] **Step 2: Write `docs/data_card.md` skeleton**

```markdown
# Data card

**Source:** Materials Project XAS database (FEFF9-computed K-edge XANES).
**License:** CC-BY 4.0.
**Fetched:** _date filled in after run_
**MP API version:** _filled in after run_

## Per-element counts

| Element | Raw | After cleaning | Final classes |
|---|---|---|---|

_filled in by `data/reports/<element>.json` after running `make data`._

## Cleaning rules

1. Drop spectra with fewer than 50 energy points.
2. Drop NaN/Inf entries.
3. Drop non-integer oxidation states.
4. Dedup by `(formula_reduced, ox_state)`.
5. Drop ox-state classes with fewer than 50 spectra.

## Known limitations

- FEFF-computed spectra; no experimental data.
- K-edge only.
- No multi-site / mixed-valence support.
```

- [ ] **Step 3: Write `docs/methods.md` skeleton**

```markdown
# Methods

## Preprocessing

Each spectrum is resampled with cubic-spline interpolation to a fixed
200-point grid on `[E0 - 10, E0 + 40]` eV, where `E0` is the per-element
inflection energy. After resampling, an edge-jump normalization is
applied: a linear pre-edge fit (in `[-10, -5]` eV) is subtracted, and
the result is divided by the post-edge linear-fit value at `E0`. Spectra
that would require more than 5 eV of extrapolation are dropped.

## Model

Small 1D-CNN with four convolutional blocks (32 → 64 → 128 → 128
channels, kernels 7/5/3/3, BatchNorm + ReLU + MaxPool) followed by
adaptive average pooling and a two-layer MLP head. ~80k parameters.

Trained with Adam (lr 1e-3, cosine decay to 1e-5), weighted
cross-entropy (inverse class frequency), batch size 128, 50 epochs max,
early stopping with patience 10 on validation accuracy. Five-model
ensemble (seeds 0–4) per element; ensemble logits are temperature-scaled
on the validation set.

## Baselines

- **Majority-class predictor.** Sanity floor.
- **GBDT on six hand features.** Pre-edge area, pre-edge peak height,
  edge inflection position, white-line height, white-line position,
  post-edge slope. LightGBM, 200 trees, 31 leaves.

## Splits

70/15/15 train/val/test, formula-disjoint (no reduced formula appears in
two splits), oxidation-state stratified, fixed seed.
```

- [ ] **Step 4: Write `docs/physics_findings.md` skeleton**

```markdown
# Physics findings

For each (element, true_ox_state, predicted_ox_state) cell of the
confusion matrix exceeding 20% confusion, one short writeup follows:

- mean spectra (with 5–95% envelopes) of each ox state in the pair;
- the energy-resolved coefficient profile of a pairwise logistic
  regression (highlights where the CNN should be looking but isn't);
- a structural-bin breakdown (coordination number, nearest-neighbor
  element) of where the failures concentrate;
- a 2–4 sentence physical explanation.

## (filled in after run)
```

- [ ] **Step 5: Commit**

```bash
git add README.md docs/data_card.md docs/methods.md docs/physics_findings.md
git commit -m "docs: README + skeletons for data card, methods, findings"
```

---

## Self-Review

**Spec coverage.** Walking the spec sections:

- §1 scope (8 elements, K-edge, FEFF, 1D-CNN, integer ox states): covered by Tasks 4, 5, 8, 10, 11.
- §1 success criteria (≥85% / ≥75% / beats GBDT / ECE ≤ 0.08 / ≥3 confused-cell writeups / `make all`): metric scaffolding in Tasks 13–15, baselines in 17–19, `make all` in 25, confused-cell pipeline in 20–22 + 29.
- §2 architecture (Spectrum dataclass, preprocess interface, OxStateCNN, CLI contract): Tasks 2, 6, 10, 25.
- §2 model architecture and param-count justification: Task 10.
- §3 source/fetch: Task 4.
- §3 cleaning rules (5 rules): Task 5.
- §3 preprocessing (200-point grid, edge-jump norm, no `larch`): Task 6.
- §3 splits (formula-disjoint, ox-state-stratified, leak check): Task 7.
- §3 class imbalance (weighted CE, per-class F1): Tasks 11 and 13.
- §3 dataset card: Task 29 skeleton, filled by `clean.py` reports during runs.
- §3 caching: Task 4 (`skip_if_exists`).
- §4 training (Adam / cosine / weighted CE / early stop / seed): Task 11.
- §4 ensemble (5 seeds): Task 12.
- §4 calibration (temperature scaling + ECE + reliability diagrams): Task 14 (calibration + ECE; reliability diagrams are NOT explicitly built — see gap below).
- §4 comparison protocol (majority / GBDT / CNN): Tasks 17, 18, 19, 25.
- §4 evaluation outputs (`metrics.json`, `confusion.png`, `failures.parquet`, `reliability.png`): metrics + confusion covered by Tasks 13, 15, 25; **`failures.parquet` and `reliability.png` are NOT generated** (gap).
- §4 physics-analysis (overlay / LR / structural / writeup): Tasks 20, 21, 22; writeup is Task 29 skeleton.
- §4 reproducibility (`configs/<element>.yaml`, fixed seeds, per-run output): partially covered — fixed seeds yes; **`configs/<element>.yaml` files are NOT created** (gap).
- §5 headline figure (Panel A bar + Panel B small multiples): Tasks 23, 24, 26.
- §5 pass/fail checklist: enforced by running the pipeline; no automated gate.
- §5 repo deliverables (`reliability/<element>.png` × 8): gap, see above.

**Gaps to close:**

1. Reliability-diagram plotter is missing.
2. `failures.parquet` writer is missing in `evaluate_element`.
3. Per-element `configs/<element>.yaml` files are missing.

Adding three more tasks below to close those.

**Placeholder scan:** no TBD, TODO, "appropriate", "fill in details", etc. in the implementation tasks. The doc skeletons (data card, methods, physics findings) contain `_filled in after run_` placeholders, which is correct since their content is the deliverable of a run, not the plan.

**Type consistency:** `Spectrum` is only consumed by tests; the production data flow is dict → `XanesDataset` → tensors. Class index ↔ ox-state mapping uses `class_to_ox_state` consistently in `XanesDataset`, `train.py`, `ensemble.py`, `evaluate_element`. `evaluate_element` reads `class_to_ox_state` with string keys when loading from JSON (it's serialized that way) — the figures script handles this conversion. OK.

---

### Task 30: Reliability-diagram plotter

**Files:**
- Modify: `xanes_oxstate/eval/plots.py`
- Modify: `tests/test_plots.py`

- [ ] **Step 1: Append failing test**

```python
import numpy as np
from xanes_oxstate.eval.plots import reliability_diagram


def test_reliability_diagram_writes_file(tmp_path):
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(alpha=[1, 1, 1], size=200)
    y = probs.argmax(axis=1)
    out = tmp_path / "rel.png"
    fig = reliability_diagram(probs, y, n_bins=10, out_path=out)
    assert out.exists()
    assert fig.axes
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_plots.py::test_reliability_diagram_writes_file -v`
Expected: AttributeError.

- [ ] **Step 3: Append to `plots.py`**

```python
def reliability_diagram(
    probs: np.ndarray, y: np.ndarray, n_bins: int = 15,
    out_path: Path | None = None,
):
    conf = probs.max(axis=-1)
    pred = probs.argmax(axis=-1)
    correct = (pred == y).astype(np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    acc = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        counts[i] = mask.sum()
        if mask.any():
            acc[i] = correct[mask].mean()

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(centers, acc, width=1 / n_bins, alpha=0.7,
           edgecolor="black", label="accuracy")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="ideal")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)
    return fig
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_plots.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add xanes_oxstate/eval/plots.py tests/test_plots.py
git commit -m "feat(plots): reliability diagram"
```

---

### Task 31: Failures parquet + reliability output in evaluate

**Files:**
- Modify: `xanes_oxstate/eval/run.py`
- Modify: `tests/test_smoke.py`

- [ ] **Step 1: Modify smoke test to assert new outputs**

Add to `tests/test_smoke.py` inside `test_end_to_end_smoke`:

```python
    assert (metrics / "Mn_failures.parquet").exists()
    assert (metrics / "Mn_reliability.png").exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_smoke.py -v`
Expected: AssertionError on the new asserts.

- [ ] **Step 3: Extend `evaluate_element` in `xanes_oxstate/eval/run.py`**

Add these imports at the top:

```python
from .plots import reliability_diagram
```

Inside `evaluate_element`, after `cm = build_confusion(...)`, append:

```python
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
```

- [ ] **Step 4: Run smoke test**

Run: `pytest tests/test_smoke.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add xanes_oxstate/eval/run.py tests/test_smoke.py
git commit -m "feat(eval): write failures parquet + per-element reliability diagram"
```

---

### Task 32: Per-element config files

**Files:**
- Create: `configs/Ti.yaml`, `configs/V.yaml`, `configs/Cr.yaml`, `configs/Mn.yaml`, `configs/Fe.yaml`, `configs/Co.yaml`, `configs/Ni.yaml`, `configs/Cu.yaml`
- Modify: `xanes_oxstate/cli.py` to optionally read a config

- [ ] **Step 1: Write each per-element config**

For each element, create `configs/<element>.yaml` (eight files; here's the
Mn template — the other seven differ only by `element:` field):

`configs/Mn.yaml`:

```yaml
element: Mn
edge: K
preprocessing:
  grid_n: 200
  grid_lo_ev: -10
  grid_hi_ev: 40
  max_extrap_ev: 5
training:
  epochs: 50
  batch_size: 128
  lr: 0.001
  patience: 10
  seeds: [0, 1, 2, 3, 4]
split:
  ratios: [0.7, 0.15, 0.15]
  seed: 42
```

Repeat for Ti, V, Cr, Fe, Co, Ni, Cu (identical fields, only `element:` changes).

- [ ] **Step 2: Modify `cli.py` to honor `--config`**

Add to the `evaluate` subparser:

```python
    ep.add_argument("--config", type=Path,
                    help="YAML config; overrides --epochs if present")
```

And in `cmd_evaluate`:

```python
def cmd_evaluate(args) -> None:
    import yaml
    from .eval.run import evaluate_element
    epochs = args.epochs
    seeds = (0, 1, 2, 3, 4)
    if args.config is not None:
        cfg = yaml.safe_load(args.config.read_text())
        epochs = cfg["training"]["epochs"]
        seeds = tuple(cfg["training"]["seeds"])
    metrics = evaluate_element(
        args.element,
        processed_dir=args.processed_dir,
        ckpt_dir=args.ckpt_dir,
        metrics_dir=args.metrics_dir,
        epochs=epochs,
        seeds=seeds,
    )
    print(f"[{args.element}] CNN acc = {metrics['accuracy']['cnn']:.3f}")
```

- [ ] **Step 3: Update `Makefile` to pass `--config`**

Replace the `train` target with:

```makefile
.PHONY: train
train:
	@for e in $(ELEMENTS); do \
	  python -m xanes_oxstate.cli evaluate --element $$e --config configs/$$e.yaml; \
	done
```

- [ ] **Step 4: Smoke test**

Run: `python -m xanes_oxstate.cli evaluate --help | grep config`
Expected: shows the `--config` argument.

- [ ] **Step 5: Commit**

```bash
git add configs/ xanes_oxstate/cli.py Makefile
git commit -m "chore: per-element YAML configs + CLI plumbing"
```
