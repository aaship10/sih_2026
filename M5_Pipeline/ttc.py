"""Time-to-collision, strictly by relative kinematics (spec section 2):

    TTC = d / v_rel

where d is Euclidean ego-obstacle distance and v_rel is the CLOSING speed
along the line of sight (the rate at which d itself is shrinking), not raw
relative speed -- two objects can have a large relative speed while moving
apart, which must NOT read as a small/urgent TTC.

Known limitation (spec open item E, M5's job to work around, not fix here):
TTC is only meaningful for objects actually closing on the ego. A crossing
or laterally-offset path can have TTC undefined (not closing along the line
of sight right now) while still being a real predicted conflict a few
seconds out -- planner.py checks the COSTMAP against predicted trajectories
for exactly that case, TTC alone is not the whole safety check."""
from __future__ import annotations

import math

MIN_DISTANCE_M = 1e-3


def compute_ttc(ego_pos: tuple[float, float], ego_vel: tuple[float, float],
                 obs_pos: tuple[float, float], obs_vel: tuple[float, float]) -> float | None:
    """Returns None if the object is not closing (open item E's case:
    stationary relative to ego along the sightline, or moving away) --
    callers must treat None as "TTC does not apply", never as "infinite
    but safe"."""
    rx, ry = obs_pos[0] - ego_pos[0], obs_pos[1] - ego_pos[1]
    d = math.hypot(rx, ry)
    if d < MIN_DISTANCE_M:
        return 0.0  # already overlapping
    rvx, rvy = obs_vel[0] - ego_vel[0], obs_vel[1] - ego_vel[1]
    closing_speed = -(rx * rvx + ry * rvy) / d
    if closing_speed <= 1e-6:
        return None
    return d / closing_speed


def min_ttc(ego_pos: tuple[float, float], ego_vel: tuple[float, float],
            obstacles: list[tuple[tuple[float, float], tuple[float, float]]]) -> float | None:
    """obstacles: list of (pos, vel) pairs. Returns the smallest defined TTC
    across all obstacles, or None if none are closing."""
    values = [t for pos, vel in obstacles if (t := compute_ttc(ego_pos, ego_vel, pos, vel)) is not None]
    return min(values) if values else None
