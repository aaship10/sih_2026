"""Ego 3-circle physical footprint (spec section 2): front/center/rear
circles along the vehicle's own heading, each projected onto a Costmap to
check for zone intersections -- a single point at the vehicle's center
would miss a corner-clip that the actual (rectangular-ish) body would hit."""
from __future__ import annotations

import math

import config
from costmap import Costmap, worst_risk


def circle_centers(ego_x: float, ego_y: float, ego_yaw_rad: float,
                    offsets_m: tuple[float, float, float] = config.EGO_FOOTPRINT_OFFSETS_M) -> list[tuple[float, float]]:
    """offsets_m are signed distances along the heading axis from the ego's
    reference point (rear-most negative, front-most positive)."""
    heading = (math.cos(ego_yaw_rad), math.sin(ego_yaw_rad))
    return [(ego_x + heading[0] * off, ego_y + heading[1] * off) for off in offsets_m]


def footprint_risk(ego_x: float, ego_y: float, ego_yaw_rad: float, costmap: Costmap,
                    radius_m: float = config.EGO_FOOTPRINT_RADIUS_M) -> str:
    centers = circle_centers(ego_x, ego_y, ego_yaw_rad)
    return worst_risk([costmap.risk_at(cx, cy, radius_m) for cx, cy in centers])
