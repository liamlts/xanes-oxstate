# xanes-oxstate — Design Spec

**Date:** 2026-05-28
**Status:** Draft, pre-implementation
**Author:** Liam Schmidt

---

## 1. Scope, deliverable, success criteria

### Pitch

A 1D-CNN classifier that predicts the absorbing atom's formal oxidation
state from K-edge XANES spectra alone, trained on the public Materials
Project XAS database, with per-element confusion matrices that identify
spectrally degenerate oxidation states as a small physics result.

### Portfolio context

This is project #1 in a four-project physics-ML portfolio series. The
later projects (phonon DOS prediction, calibrated band-gap regression,
SuperCon Tc regression) are out of scope here; each will get its own
spec when #1 is done.

### Deliverables (priority order)

1. GitHub repo + README with one-command training and evaluation.
2. One end-to-end notebook (`xanes_oxstate.ipynb`).
3. Headline figure: per-element accuracy bar chart + per-element
   confusion matrices (small multiples).
4. Physics finding: 1–2 paragraphs in README documenting which
   (element, ox-state-pair) cases are spectrally indistinguishable and
   why.
5. *Optional:* HuggingFace model card or small Gradio demo.

### Scope (locked)

- **Elements:** Ti, V, Cr, Mn, Fe, Co, Ni, Cu (eight 3d transition metals,
  eight per-element models).
- **Edge:** K-edge only.
- **Source:** Materials Project FEFF9-computed XANES.
- **Task:** integer-oxidation-state classification, per-element class set
  determined empirically from data.
- **Model:** 1D-CNN, ~80k params. No pretraining, no transfer learning.

### Out of scope (explicit)

- L-edges and other edge structures.
- Experimental spectra (FEFF-computed only).
- Non-integer / mixed-valence oxidation states.
- Architecture variants beyond the 1D-CNN (transformer, ResNet) unless
  baseline is suspiciously weak.
- Per-site spectra for multi-site materials beyond what MP exposes.

### Success criteria

- ≥85% overall top-1 accuracy averaged across the 8 elements.
- ≥75% per-element accuracy, or an honest discussion of why a specific
  element fails.
- CNN ensemble beats GBDT-on-hand-features baseline by ≥5 points overall.
  (If it doesn't, that itself is reportable.)
- ECE ≤ 0.08 per element after temperature scaling.
- ≥3 (element, ox-state-pair) confused-cells with written
  physics-grounded explanations.
- `pip install -e . && make all` reproduces the headline figure in <2 h
  on a laptop.

### Timeline

~1 week part-time, four phases:
- Day 1–2: data pull + clean.
- Day 3–4: model + training loop.
- Day 5: per-element runs + figures.
- Day 6–7: writeup + repo polish.

---

## 2. Architecture

### Repo layout

```
xanes-oxstate/
├── README.md
├── pyproject.toml
├── Makefile                  # make data, make train, make eval, make figures
│
├── xanes_oxstate/
│   ├── data/
│   │   ├── fetch.py          # MP API → raw spectra cache
│   │   ├── clean.py          # filter, dedup, oxidation-state labeling
│   │   ├── preprocess.py     # interpolate to common grid, normalize
│   │   └── split.py          # element-stratified train/val/test
│   │
│   ├── model/
│   │   ├── cnn.py            # 1D-CNN architecture
│   │   └── train.py          # training loop, checkpoints
│   │
│   ├── eval/
│   │   ├── metrics.py        # accuracy, per-class F1, calibration
│   │   ├── confusion.py      # confusion matrix builder
│   │   └── plots.py          # bar chart + small-multiples figure
│   │
│   └── cli.py                # python -m xanes_oxstate.train --element Mn
│
├── notebooks/
│   └── xanes_oxstate.ipynb   # end-to-end walkthrough
│
├── data/                     # gitignored
│   ├── raw/                  # per-element raw cache
│   └── processed/            # per-element train/val/test parquet
│
├── checkpoints/              # gitignored
├── figures/                  # versioned final figures
├── configs/                  # per-element YAML
└── tests/
```

### Data flow

```
MP API → fetch.py → raw/<element>.jsonl
                       ↓
                   clean.py (filter, dedup, label)
                       ↓
                   preprocess.py (resample to 200-pt grid, edge-jump norm)
                       ↓
                   split.py (70/15/15, formula-disjoint, ox-state-stratified)
                       ↓
                   processed/<element>_{train,val,test}.parquet
                       ↓
                   model/train.py → checkpoints/<element>_seed{0..4}.pt
                       ↓
                   eval/metrics.py + eval/confusion.py → metrics + matrices
                       ↓
                   eval/plots.py headline() → figures/*
```

### Key interfaces

1. **`Spectrum` dataclass** — single source of truth from `clean.py`
   onward. Fields: `(energy, intensity, element, ox_state, formula, mp_id)`.
2. **`preprocess.normalize(spectrum) → np.ndarray[200]`** — every model
   input goes through this.
3. **`cnn.OxStateCNN`** — `forward(x: [B, 1, 200]) → logits: [B, n_classes]`.
   `n_classes` set per element from training data.
4. **CLI contract** —
   `python -m xanes_oxstate.train --element Mn --epochs 50 --seed 0`
   reproduces published results.

### Model architecture

```
Input:  [B, 1, 200]                  # normalized XANES, 200 points

Conv1d(1   → 32,  k=7, p=3) → BN → ReLU → MaxPool(2)   # [B, 32, 100]
Conv1d(32  → 64,  k=5, p=2) → BN → ReLU → MaxPool(2)   # [B, 64,  50]
Conv1d(64  → 128, k=3, p=1) → BN → ReLU → MaxPool(2)   # [B, 128, 25]
Conv1d(128 → 128, k=3, p=1) → BN → ReLU → AdaptivePool(1) # [B, 128, 1]

Flatten → Dropout(0.3) → Linear(128 → 64) → ReLU → Linear(64 → n_classes)
```

~80k params. Trains in <5 min/element on a laptop CPU.

### Architecture justification

- Per-element XANES subsets are 3k–15k spectra — modest scale.
- XANES features are local in energy (pre-edge feature, white line,
  edge oscillations); convolutions are the natural inductive bias.
- 1D-CNN matches published SOTA-equivalent on this kind of task
  (Carbone et al., Torrisi et al.).
- Going bigger gains nothing on this data scale; the time savings are
  spent on the physics-analysis contribution.

---

## 3. Data

### Source

Materials Project XAS database, accessed via `mp-api`.
FEFF9-computed K-edge spectra, ~0–50 eV above edge.
License: CC-BY 4.0 (Materials Project attribution required).
Auth: `MP_API_KEY` env var.

### Expected per-element coverage

| Element | Expected spectra | Likely classes |
|---|---|---|
| Ti | ~5–8k  | +2, +3, +4                  |
| V  | ~3–5k  | +2, +3, +4, +5              |
| Cr | ~3–5k  | +2, +3, +4, +6              |
| Mn | ~6–10k | +2, +3, +4 (+5, +6, +7 rare)|
| Fe | ~10–15k| +2, +3                      |
| Co | ~5–8k  | +2, +3 (+4 rare)            |
| Ni | ~4–6k  | +2 (+3, +4 rare)            |
| Cu | ~5–8k  | +1, +2 (+3 rare)            |

If actual counts deviate >2× from these, flag and revisit class
definitions before training.

### Cleaning rules (in order)

1. Drop entries with `len(energies) < 50` or any NaN/Inf.
2. Drop non-integer oxidation states (mixed valence, fractional).
3. Drop oxidation-state classes with fewer than 50 spectra after
   cleaning. Document drops per element.
4. Dedup by `(formula_reduced, ox_state)`; keep lowest-energy polymorph.
5. Site filter: one spectrum per inequivalent absorbing site; if
   multiple sites share an ox-state, average them.

Every drop logged to `data/cleaning_report.json` with counts.

### Preprocessing

**Energy grid:** per-element E₀ from median inflection of training set.
Resample to 200 linearly-spaced points on `[E₀ − 10, E₀ + 40]` eV.
Cubic-spline interpolate; exclude spectra requiring >5 eV extrapolation.

**Intensity normalization** (edge-jump, NumPy, no `larch` dep):
1. Pre-edge linear fit on `[E₀ − 10, E₀ − 5]` → subtract.
2. Post-edge linear fit on `[E₀ + 30, E₀ + 40]` → divide by edge jump.
3. Result: pre-edge ≈ 0, post-edge plateau ≈ 1.

### Train/val/test split

70/15/15. Three-level stratification:
1. **Formula-disjoint** — a given reduced formula appears in exactly one
   split. Critical anti-leakage measure.
2. **Oxidation-state-stratified** — each split contains roughly the same
   per-class fraction.
3. **Element-stratified** — per-element splits built independently.

`SPLIT_SEED = 42`, version-controlled. Post-split leak check: assert no
formula appears in two splits per element; fail loudly otherwise.

### Class imbalance

- Training: inverse-frequency class weighting in cross-entropy.
- Evaluation: per-class F1 alongside top-1 accuracy.
- No oversampling.

### Dataset card

`docs/data_card.md` documents source, license, MP API version, fetch
date, per-element cleaning counts, final class distribution,
limitations (FEFF-computed not experimental; no hybrid-charge materials;
K-edge only).

### Caching

`data/raw/<element>.jsonl` (gitignored). Pipeline resumable — re-running
`make data` skips elements already fetched.

---

## 4. Methods

### Training (per element)

| Setting | Value |
|---|---|
| Optimizer | Adam, lr=1e-3 |
| Schedule | Cosine decay to 1e-5 over 50 epochs |
| Batch size | 128 |
| Loss | Weighted cross-entropy (inverse-frequency) |
| Epochs | 50 max, early stop patience 10 on val accuracy |
| Seed | Fixed per run |

No data augmentation in v1.

**Five-model ensemble** per element (seeds 0–4). Each model trains in
<5 min on CPU; ensemble gives free uncertainty estimates.

### Calibration

**Temperature scaling**: after training, fit scalar `T` on val logits by
minimizing val NLL (Guo et al. 2017).

**Reported metrics**: ECE, reliability diagrams per element, confidence
histograms.

### Comparison protocol (three baselines per element)

| Estimator | What it tests |
|---|---|
| Majority-class predictor | Sanity floor |
| GBDT on hand features (pre-edge area, edge position, white-line height, post-edge slope) | "What would a chemist do without ML?" — the bar to beat |
| 1D-CNN ensemble | Main contribution |

If the CNN doesn't beat the GBDT meaningfully, that's a reportable
finding, not a hidden one.

### Evaluation outputs (per element)

- `metrics.json` — top-1 accuracy, per-class F1, ECE, ensemble
  disagreement.
- `confusion.png` — row-normalized, with counts.
- `reliability.png` — reliability diagram.
- `failures.parquet` — every misclassified test example with spectrum,
  true/pred labels, confidence.

### Physics-analysis step (the actual contribution)

For each confusion-matrix cell with >20% confusion:

1. **Spectral overlay**: plot mean ± envelope of spectra for each
   confused ox state. Are they actually similar?
2. **Distinguishing features**: train a tiny logistic regression on the
   pair; inspect coefficients to find which spectral region the CNN
   isn't using.
3. **Structural correlate**: bin failures by coordination number /
   nearest-neighbor element. Is the confusion concentrated in a
   structural family?
4. **Physics explanation**: 2–4 sentences per confused pair grounded in
   the spectral physics (covalency, distortion, spin state, etc.).

### Reproducibility

- All hyperparameters in `configs/<element>.yaml`, version-controlled.
- Fixed seeds for split, init, training, ensemble.
- Per-run output: `runs/<timestamp>_<element>_<seed>/` with checkpoint,
  config, metrics, plots.
- `make all` reproduces the headline figure end-to-end in <2 h on a
  laptop.

### Explicitly not in scope

- Hyperparameter search.
- Transformer / attention / pretraining variants.
- Per-class threshold tuning.
- Mixup / CutMix / fancy augmentation.

---

## 5. Evaluation & deliverables

### Headline figure (the deliverable image)

One figure, two panels:

**Panel A — per-element accuracy bar chart.** 8 elements × 3 estimators
(majority, GBDT, CNN ensemble). Error bars from ensemble std. Annotated
lines at 75% (per-element bar) and 85% (overall bar).

**Panel B — confusion matrix small multiples.** 2×4 grid, row-normalized,
viridis. Boxed cells highlight (true, pred) pairs with >20% confusion —
the cases discussed in the physics-analysis section.

PDF + PNG, 300 DPI, caption-ready.

### Pass/fail checklist

- [ ] Overall accuracy ≥ 85% (CNN ensemble).
- [ ] Per-element accuracy ≥ 75% for each element, or honest discussion.
- [ ] CNN ensemble beats GBDT by ≥ 5 points overall.
- [ ] ECE ≤ 0.08 per element after temperature scaling.
- [ ] ≥ 3 confused-cell physics writeups.
- [ ] `pip install -e . && make all` reproduces the headline figure in
  <2 h.
- [ ] Data card complete; MP attribution in place.

### Repository deliverables

```
README.md              ← portfolio-facing writeup (800–1500 words)
notebooks/xanes_oxstate.ipynb
figures/per_element_accuracy.{pdf,png}
figures/confusions.{pdf,png}
figures/reliability/<element>.png   (×8)
docs/data_card.md
docs/methods.md        ← short methods writeup (~500 words)
docs/physics_findings.md
configs/<element>.yaml (×8)
```

The README is the public-facing product. Sections:
1. One-line pitch + headline figure inline.
2. Pass/fail table with green checkmarks.
3. How to reproduce.
4. Short methods overview pointing at `docs/methods.md`.
5. Physics findings (bullets, pointing at `docs/physics_findings.md`).
6. Limitations + future work.

---

## 6. Open questions for implementation phase

These are flagged for resolution during implementation, not in this spec:

1. **MP API rate limits** — current limits and key requirements may
   require batching adjustments. Discover empirically in Phase 1.
2. **`pymatgen` BVAnalyzer reliability** — may need a fallback path for
   oxidation-state assignment. Verify against MP-supplied labels where
   available.
3. **Per-element E₀ choice** — using training-set median inflection vs.
   tabulated value. Default to the data-driven choice; revisit if a
   per-element model has obvious normalization issues.
4. **Notebook vs. CLI as primary entry point** — assume CLI for
   reproducibility, notebook as walkthrough. If MP API requires
   interactive auth, revisit.
