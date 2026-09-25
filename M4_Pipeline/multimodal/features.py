"""Windowed motion features derived from a track's recent history: a
smoothed velocity (for building trajectories from, instead of trusting a
single raw current-frame reading) and two volatility measures that feed
multimodal/fuzzy.py's mode-probability weighting.

Only MEASURED samples (HistorySample.measured=True) are used -- a coasted
sample is M3's own dead-reckoned guess, not a real observation, and
including it here would smooth toward M3's extrapolation instead of actual
measured motion (see tracking_input/history.py's docstring).

Live CARLA testing (scenario3.py junction run) found the dominant source of
large prediction error was NOT an implausible velocity, but a plausible-but-
wrong one: a track whose reported velocity climbed smoothly across several
consecutive frames (0.81 -> 2.87 -> 5.79 m/s) while the real object was
still stationary -- camera-only position noise being misread as motion by
M3's Kalman filter. speed_volatility exists specifically to catch this
pattern: a track whose recent speed readings disagree with each other is
exactly the case a single-snapshot constant-velocity/acceleration
extrapolation handles worst.

Phase 5 evaluation against a full live run (M4 README's evaluation/run_eval.py)
found a SECOND, opposite failure mode a plain mean-of-velocity-readings
smoother does not fix: M3's per-frame velocity state can also OSCILLATE in
sign frame-to-frame for a genuinely, steadily-moving object (observed on a
moving car: (-0.46,0.04) -> (0.55,-0.06) -> (-0.41,0.17) m/s while the car
actually moved at a fairly constant ~3.3 m/s) -- a mean of those readings
cancels toward ~0 instead of recovering the real speed, which is worse than
doing nothing for a genuinely moving object. `smoothed_velocity` is
therefore fit by ordinary least squares on the window's own MEASURED
(position, timestamp) pairs (the velocity implied by the actual observed
displacement trend), not by averaging M3's own per-frame velocity readings:
position measurements are direct observations, so this is not fooled by
noise in M3's internal Kalman velocity state either climbing smoothly or
oscillating in sign, and still reduces to the same one-sample passthrough
below `MOTION_FEATURE_WINDOW`'s minimum.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tracking_input.history import HistorySample


@dataclass
class MotionFeatures:
    smoothed_velocity: np.ndarray   # least-squares position/time fit over the window -- feeds the nominal/stop/lateral trajectories
    speed_volatility: float         # std-dev of speed magnitude over the window, m/s
    heading_volatility_rad: float   # mean abs consecutive heading change over the window, radians
    n_samples_used: int


def _fit_velocity_from_positions(measured: list[HistorySample]) -> np.ndarray:
    """Ordinary-least-squares velocity: the slope of position vs. timestamp
    across the window, fit independently per axis, using the window's own
    observed positions rather than M3's per-frame velocity readings (see
    this module's docstring). Falls back to a plain mean of the raw
    velocity readings if the window spans ~0 time (degenerate for a slope
    fit -- e.g. duplicate timestamps)."""
    times = np.array([s.timestamp for s in measured])
    times = times - times[0]
    if times[-1] < 1e-6:
        return np.array([s.velocity for s in measured]).mean(axis=0)
    positions = np.array([s.position for s in measured])
    vx = np.polyfit(times, positions[:, 0], 1)[0]
    vy = np.polyfit(times, positions[:, 1], 1)[0]
    return np.array([vx, vy])


def compute_motion_features(history: list[HistorySample], current_velocity: np.ndarray, window_n: int) -> MotionFeatures:
    measured = [s for s in history if s.measured][-window_n:]
    if len(measured) < 2:
        # Not enough real observations to say anything about smoothing or
        # volatility -- pass the current reading through unchanged. Callers
        # (multimodal/modes.py) treat n_samples_used below a threshold as
        # "not enough data for fuzzy weighting" and fall back to the static
        # per-class defaults instead.
        return MotionFeatures(current_velocity, 0.0, 0.0, len(measured))

    velocities = np.array([s.velocity for s in measured])
    smoothed = _fit_velocity_from_positions(measured)

    speeds = np.linalg.norm(velocities, axis=1)
    speed_volatility = float(np.std(speeds))

    # Heading is undefined (noisy) for near-zero-speed samples -- exclude
    # them from the heading-volatility computation the same way
    # _recent_turn_rate_rad_s already does elsewhere in this package.
    headings = [math.atan2(v[1], v[0]) for v in velocities if float(np.linalg.norm(v)) > 0.3]
    diffs = [
        abs(math.atan2(math.sin(headings[i] - headings[i - 1]), math.cos(headings[i] - headings[i - 1])))
        for i in range(1, len(headings))
    ]
    heading_volatility = float(np.mean(diffs)) if diffs else 0.0

    return MotionFeatures(smoothed, speed_volatility, heading_volatility, len(measured))
