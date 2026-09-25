"""Shared Trajectory type: a sequence of predicted (t, x, y) samples plus the
per-sample uncertainty (filled in later by uncertainty/covariance.py) and a
mode label/probability (filled in by multimodal/modes.py). A bare
constant-velocity call before uncertainty/multimodal runs just leaves those
fields at their defaults (zero std-devs, "nominal" mode, probability 1.0) --
every field is present from the start so nothing downstream needs an
Optional check.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def sample_times(horizon_s: float, time_step_s: float) -> np.ndarray:
    """t = 0, step, 2*step, ... up to and including horizon (or the closest
    multiple of step <= horizon + step/2, to avoid float rounding dropping
    the last sample)."""
    n = int(round(horizon_s / time_step_s))
    return np.linspace(0.0, n * time_step_s, n + 1)


@dataclass
class Trajectory:
    track_id: int
    mode: str                    # "nominal" | "stop" | "lateral"
    probability: float           # this mode's probability, see multimodal/modes.py
    times: np.ndarray            # shape (N,), seconds from the prediction's t0
    points: np.ndarray           # shape (N, 2), world-frame [x, y]
    along_sigma: np.ndarray = field(default=None)    # shape (N,), meters -- along-track std-dev at each sample
    lateral_sigma: np.ndarray = field(default=None)  # shape (N,), meters -- cross-track std-dev at each sample
    heading_rad: float = 0.0     # heading used to orient along/lateral sigma into x/y (see uncertainty/covariance.py)

    def __post_init__(self) -> None:
        n = len(self.times)
        if self.along_sigma is None:
            self.along_sigma = np.zeros(n)
        if self.lateral_sigma is None:
            self.lateral_sigma = np.zeros(n)

    def position_at(self, t: float) -> np.ndarray:
        """Linear interpolation into `points` at time t (t clamped to the
        trajectory's own [0, horizon] range)."""
        t = float(np.clip(t, self.times[0], self.times[-1]))
        return np.array([np.interp(t, self.times, self.points[:, 0]),
                          np.interp(t, self.times, self.points[:, 1])])
