"""Constant-acceleration model (section 10.B). Acceleration is not sent by
M3 (its own filter is constant-velocity -- M3 README 7B.6/7.1) so M4 derives
it from history here, and the result WILL be noisy: a is a second
derivative, and M3's positions/velocities already carry their own tracking
noise (see M3 README's velocity-jump finding). This is why constant velocity
is the PRIMARY model (class_specific.py) and this is offered as a secondary
option, used only for classes/situations where the noise is worth the extra
responsiveness (see class_specific.py's `use_acceleration` heuristic) --
never blindly preferred over CV.

p(t) = p0 + v0*t + 0.5*a*t^2

Acceleration is estimated as the least-squares slope of velocity vs.
timestamp over the last few MEASURED (not coasted -- see tracking_input/
history.py) samples, which is robust to M3's irregular packet spacing
(section 7.2: never assume a fixed dt) and averages down noise better than a
single finite difference between the last two samples would.
"""
from __future__ import annotations

import numpy as np

from models.trajectory import Trajectory, sample_times
from tracking_input.history import HistorySample

MIN_SAMPLES_FOR_ACCELERATION = 3


def estimate_acceleration(history: list[HistorySample]) -> np.ndarray | None:
    measured = [s for s in history if s.measured]
    if len(measured) < MIN_SAMPLES_FOR_ACCELERATION:
        return None
    t = np.array([s.timestamp for s in measured])
    t = t - t[0]
    if t[-1] <= 1e-3:
        return None
    vx = np.array([s.velocity[0] for s in measured])
    vy = np.array([s.velocity[1] for s in measured])
    # np.polyfit degree-1 slope = acceleration (least-squares over all points,
    # not just the endpoints).
    ax = float(np.polyfit(t, vx, 1)[0])
    ay = float(np.polyfit(t, vy, 1)[0])
    return np.array([ax, ay])


def predict(track_id: int, position: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray,
            horizon_s: float, time_step_s: float, max_speed_mps: float | None = None) -> Trajectory:
    times = sample_times(horizon_s, time_step_s)
    points = np.zeros((len(times), 2))
    v = velocity.copy()
    p = position.copy()
    prev_t = 0.0
    for i, t in enumerate(times):
        dt = t - prev_t
        p = p + v * dt + 0.5 * acceleration * dt * dt
        v = v + acceleration * dt
        if max_speed_mps is not None:
            speed = float(np.linalg.norm(v))
            if speed > max_speed_mps > 0:
                v = v * (max_speed_mps / speed)
        points[i] = p
        prev_t = t
    return Trajectory(track_id=track_id, mode="nominal", probability=1.0, times=times, points=points)
