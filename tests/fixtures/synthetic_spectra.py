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
