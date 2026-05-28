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
