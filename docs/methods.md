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
