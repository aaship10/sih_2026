"""Constant-velocity baseline (section 10.A, section 25's PRIMARY choice --
see class_specific.py's module docstring for why CV, not CA, is primary).

p(t) = p0 + v0 * t

The simplest possible model, and a deliberately strong baseline: M3's own
Kalman filter is itself constant-velocity, so a track's `velocity` is
already the best linear estimate M3 could produce from the measurements it
has -- projecting it forward is consistent with, not fighting, M3's own
model. Everything else in M4 (class-specific tuning, multimodal branching,
uncertainty growth) sits ON TOP of this, not instead of it.
"""
from __future__ import annotations

import numpy as np

from models.trajectory import Trajectory, sample_times


def predict(track_id: int, position: np.ndarray, velocity: np.ndarray,
            horizon_s: float, time_step_s: float) -> Trajectory:
    times = sample_times(horizon_s, time_step_s)
    points = position[None, :] + velocity[None, :] * times[:, None]
    return Trajectory(track_id=track_id, mode="nominal", probability=1.0, times=times, points=points)
