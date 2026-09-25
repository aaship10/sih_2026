"""Growing along-track / cross-track std-dev, the ONE primary uncertainty
representation for this project (section 14 asks for exactly one).

Why this over the alternatives section 14 lists:
  - A single covariance MATRIX per object (constant for the whole horizon)
    can't express "uncertainty grows the further out we predict", which is
    the actual dominant effect here.
  - Full multivariate Gaussian sampling / particle clouds are more general
    than anything M4 needs to feed M5 a usable occupancy region (section 15
    only needs "possible region at each t", not a full posterior).
  - An along/lateral pair, oriented along the trajectory's own heading, is
    the smallest representation that still captures the one asymmetry that
    actually matters for driving: an object is far more likely to be
    further along (or behind) its predicted path than it is to have
    teleported sideways off it -- EXCEPT for classes that can turn/cross
    (pedestrians, animals), where config.ClassProfile.lateral_growth_mps is
    set close to (or above) along_growth_mps specifically to represent that.
  - It converts directly to an ellipse for visualization (section 48) and
    to a simple x/y covariance matrix if any downstream consumer (M5) wants
    one: see `to_xy_std`.

sigma(t) = sigma0 + growth_rate * t * quality_penalty
where quality_penalty = 2 - quality (quality in [0,1] from
tracking_input/quality.py) -- a quality of 1.0 leaves growth unscaled, a
quality of 0.0 (worst case: a coasting, low-hit, LiDAR-only track) DOUBLES
the growth rate, so the uncertainty band opens up much faster for input M4
already has reason to distrust.
"""
from __future__ import annotations

import math

import numpy as np

import config
from models.trajectory import Trajectory


def _heading_from_velocity(velocity: np.ndarray, fallback: float) -> float:
    speed = float(np.linalg.norm(velocity))
    if speed < 0.3:
        return fallback
    return float(math.atan2(velocity[1], velocity[0]))


def apply_uncertainty(traj: Trajectory, profile: "config.ClassProfile", velocity: np.ndarray,
                       quality: float, prior_heading_rad: float = 0.0) -> Trajectory:
    """Mutates and returns `traj` with along_sigma/lateral_sigma/heading_rad
    filled in. `quality` is the object's tracking-quality score (0-1, see
    tracking_input/quality.py) for the frame the prediction was made from."""
    heading = _heading_from_velocity(velocity, prior_heading_rad)
    traj.heading_rad = heading
    penalty = 2.0 - max(0.0, min(1.0, quality))
    traj.along_sigma = profile.along_sigma0_m + profile.along_growth_mps * penalty * traj.times
    traj.lateral_sigma = profile.lateral_sigma0_m + profile.lateral_growth_mps * penalty * traj.times
    return traj


def to_xy_std(traj: Trajectory, index: int) -> tuple[float, float, float]:
    """Return (sigma_x, sigma_y, rotation_rad) for the ellipse at
    `traj.points[index]` -- rotation_rad is the ellipse's major-axis
    heading (== traj.heading_rad), for a caller that wants to draw it."""
    return float(traj.along_sigma[index]), float(traj.lateral_sigma[index]), traj.heading_rad


def ellipse_area(along_sigma: float, lateral_sigma: float) -> float:
    """pi * a * b, a simple scalar 'how big is the uncertainty region' --
    used by `normalized_spread` below and available to any caller (e.g.
    visualization) that wants a single number for "how big is this
    ellipse" rather than the two separate axes."""
    return math.pi * along_sigma * lateral_sigma


def normalized_spread(along_sigma_m: float, lateral_sigma_m: float, max_sigma_m: float) -> float:
    """How wide has the uncertainty band grown, as a single scalar in
    [0, 1]: the ellipse area at (along_sigma_m, lateral_sigma_m) relative to
    a reference ellipse of semi-axes (max_sigma_m, max_sigma_m), clipped to
    1. 0 means "no growth yet" (t=0), 1 means "at or beyond the reference
    spread" -- used by predictor.py to fold uncertainty growth into the
    overall prediction `confidence` it reports (a technically high-quality
    track whose class is inherently unpredictable should still end up with
    lower confidence than its input quality alone would suggest)."""
    area = ellipse_area(along_sigma_m, lateral_sigma_m)
    max_area = ellipse_area(max_sigma_m, max_sigma_m)
    return max(0.0, min(1.0, area / max_area)) if max_area > 0 else 0.0
