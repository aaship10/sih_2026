"""2D local costmap (spec section 2): obstacles inflated into Red
(Critical) / Yellow (Medium) / Green (Low) risk zones via a Euclidean
Distance Transform, so a risk lookup anywhere on the grid is "distance to
nearest obstacle cell", not a per-obstacle loop.

The grid is built fresh each tick, in WORLD frame, centered on wherever the
caller places it (planner.py centers it on the ego's current position) --
it is a snapshot, not a persistent map, since M5 has no static drivable-
space/road-edge source of its own (spec section 4: that comes from M1).

ANISOTROPIC (live-testing finding): a plain circular (isotropic) distance
check made a roadside cone/sign 1.6-2.8m to the SIDE of the ego read as the
same collision risk as something 1.6-2.8m directly AHEAD -- these scenarios
deliberately place static clutter just off the lane, and with no lane/
drivable-space source (spec section 4, still deferred), that isotropy was
the only thing standing between "safely beside the car" and "blocking the
lane". Fixed by rotating the grid into the EGO'S OWN heading frame and
stretching the LATERAL axis before the distance transform runs -- an
object purely to the side reads farther away than the same raw distance
directly ahead, while genuine head-on proximity is completely unaffected
(along-heading distances are never stretched). This is a linear (rotate +
per-axis scale) transform applied identically to both marked obstacles and
query points, so the EDT still returns a mathematically consistent
pairwise distance between any two points -- just an elliptical one instead
of a circular one."""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import distance_transform_edt

import config

RED, YELLOW, GREEN, NONE = "RED", "YELLOW", "GREEN", "NONE"


def anisotropic_offset(dx: float, dy: float, heading_rad: float, lateral_stretch: float) -> tuple[float, float]:
    """Decomposes (dx, dy) into (along-heading, lateral) components and
    stretches the lateral one. The perpendicular basis's sign is arbitrary
    (only the magnitude after stretching is ever used) -- see
    M1_Pipeline/m5_control.py's _clamp_lateral_offset_to_lane for the same
    non-requirement."""
    along = dx * math.cos(heading_rad) + dy * math.sin(heading_rad)
    lateral = -dx * math.sin(heading_rad) + dy * math.cos(heading_rad)
    return along, lateral * lateral_stretch


class Costmap:
    def __init__(self, center_x: float, center_y: float, heading_rad: float = 0.0,
                 lateral_stretch: float = 1.0,
                 half_extent_m: float = config.COSTMAP_HALF_EXTENT_M,
                 resolution_m: float = config.COSTMAP_RESOLUTION_M) -> None:
        self.center_x, self.center_y = center_x, center_y
        self.heading_rad, self.lateral_stretch = heading_rad, lateral_stretch
        self.half_extent_m, self.resolution_m = half_extent_m, resolution_m
        self.n = int(round(2 * half_extent_m / resolution_m)) + 1
        self._occupied = np.zeros((self.n, self.n), dtype=bool)
        self._dist_m: np.ndarray | None = None

    def _to_index(self, x: float, y: float) -> tuple[int, int] | None:
        u, v = anisotropic_offset(x - self.center_x, y - self.center_y, self.heading_rad, self.lateral_stretch)
        col = int(round((u + self.half_extent_m) / self.resolution_m))
        row = int(round((v + self.half_extent_m) / self.resolution_m))
        if 0 <= row < self.n and 0 <= col < self.n:
            return row, col
        return None

    def mark_obstacle(self, x: float, y: float) -> None:
        idx = self._to_index(x, y)
        if idx is not None:
            self._occupied[idx] = True
        self._dist_m = None  # invalidate cached transform

    def _ensure_transform(self) -> np.ndarray:
        if self._dist_m is None:
            if not self._occupied.any():
                self._dist_m = np.full((self.n, self.n), math.inf)
            else:
                # distance_transform_edt gives distance (in CELLS) from each
                # False cell to the nearest True cell when fed ~occupied.
                self._dist_m = distance_transform_edt(~self._occupied) * self.resolution_m
        return self._dist_m

    def distance_to_nearest_obstacle_m(self, x: float, y: float) -> float:
        """Distance (in the possibly-stretched frame -- see module
        docstring) from (x, y) to the nearest marked obstacle cell. Returns
        +inf if (x, y) is outside the grid or nothing has been marked --
        both mean "no known risk here", not "zero risk"."""
        idx = self._to_index(x, y)
        if idx is None:
            return math.inf
        return float(self._ensure_transform()[idx])

    def risk_at(self, x: float, y: float, footprint_radius_m: float = 0.0) -> str:
        """Risk zone at (x, y), accounting for a circular footprint of
        `footprint_radius_m` centered there (a wider footprint has less
        effective clearance to the same obstacle)."""
        clearance = self.distance_to_nearest_obstacle_m(x, y) - footprint_radius_m
        return classify_distance(clearance)


def classify_distance(clearance_m: float) -> str:
    """Same Red/Yellow/Green/None thresholds Costmap.risk_at uses, exposed
    standalone for a point-to-point check that doesn't need a full grid
    (planner.py's future-conflict check: many (time, position) samples
    against many predicted-trajectory points, where building a fresh grid
    per timestep would be pure overhead)."""
    if clearance_m <= config.COSTMAP_RED_RADIUS_M:
        return RED
    if clearance_m <= config.COSTMAP_YELLOW_RADIUS_M:
        return YELLOW
    if clearance_m <= config.COSTMAP_GREEN_RADIUS_M:
        return GREEN
    return NONE


RISK_SEVERITY = {NONE: 0, GREEN: 1, YELLOW: 2, RED: 3}


def worst_risk(risks: list[str]) -> str:
    if not risks:
        return NONE
    return max(risks, key=lambda r: RISK_SEVERITY[r])
