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
    lo = energy.min() + 0.1 * np.ptp(energy)
    hi = energy.max() - 0.1 * np.ptp(energy)
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
