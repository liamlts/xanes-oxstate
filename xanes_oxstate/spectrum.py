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
