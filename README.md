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
