"""
ttc.py
======
Two related but distinct metrics:

TIME-TO-COLLISION (TTC) -- valid when ego and object are closing on
roughly the same line (e.g. following, head-on, same-direction
overtake). The basic formula:

    r = p_object - p_ego
    v_rel = v_object - v_ego
    TTC = -(r . v_rel) / ||v_rel||^2      (only when this is > 0 and
                                            ||v_rel|| is not ~0)

This is NOT valid/meaningful for objects on a crossing path that will
never actually be at the same (x,y) even as ||r|| shrinks then grows
again -- e.g. a pedestrian crossing perpendicular to the ego's lane.
For those, use TIME-TO-CONFLICT instead (below).

Both formulas treat objects as points. We inflate the "collision"
condition into a distance threshold `combined_radius`, derived from
M3's `size` field (already precise, from LiDAR cluster extent) plus
the ego vehicle's own half-width -- this is the "bounding box / safety
radius" upgrade the project brief asks for, without needing full swept
polygon intersection.
"""

import math


def combined_radius(object_size_xyz, ego_half_width=1.0, safety_margin=0.3):
    """
    object_size_xyz: M3's [l, w, h] for the tracked object.
    ego_half_width: half the ego vehicle's width (meters).
    safety_margin: extra buffer added on top (meters) -- covers both
        sensor/prediction noise and a "don't cut it exactly to the mm" margin.

    Returns a single scalar radius (meters) used as the "collision"
    distance threshold between the two objects' centers.
    """
    obj_half_extent = max(object_size_xyz[0], object_size_xyz[1]) / 2.0
    return obj_half_extent + ego_half_width + safety_margin


def time_to_collision(ego_pos, ego_vel, object_pos, object_vel, radius, max_ttc=15.0):
    """
    Classic closing-velocity TTC, point-mass formula inflated by
    `radius`. Returns:
        - a positive float (seconds) if a collision is predicted
          within max_ttc under constant-velocity assumptions,
        - None if not applicable (diverging, ~parallel/zero relative
          velocity, or already resolved) -- callers should fall back
          to time_to_conflict() in that case.

    VALID when: r . v_rel < 0 (closing) and the relative-velocity
    direction actually passes within `radius` of a head-on point.
    NOT valid when: paths are crossing rather than converging on the
    same point (see time_to_conflict), or v_rel ~ 0 (objects moving
    together -- TTC is undefined/infinite, not a real risk from motion).
    """
    rx = object_pos[0] - ego_pos[0]
    ry = object_pos[1] - ego_pos[1]
    vrx = object_vel[0] - ego_vel[0]
    vry = object_vel[1] - ego_vel[1]

    v_rel_sq = vrx ** 2 + vry ** 2
    if v_rel_sq < 1e-6:
        return None  # not closing in any meaningful way

    closing_rate = -(rx * vrx + ry * vry) / v_rel_sq
    if closing_rate <= 0:
        return None  # diverging or already at closest point

    # closest approach distance under constant velocity
    t_closest = closing_rate
    closest_x = rx + vrx * t_closest
    closest_y = ry + vry * t_closest
    closest_dist = math.hypot(closest_x, closest_y)

    if closest_dist > radius:
        return None  # paths pass each other outside the collision radius

    if t_closest > max_ttc:
        return None

    return round(t_closest, 3)


def time_to_conflict(ego_pos, ego_heading, ego_speed, object_trajectory_points,
                      object_time_step, conflict_radius, ego_path_points=None,
                      ego_time_step=None, max_horizon=10.0):
    """
    For crossing objects where TTC (closing-velocity) doesn't apply:
    when do the ego and the object EACH predicted to enter the same
    conflict region, even if at different times?

    Practical implementation:
    1. Build a simple nominal ego path -- if ego_path_points isn't
       given, project ego forward at constant heading/speed (same
       spirit as motion_models.constant_velocity_rollout).
    2. For each predicted object point (indexed by its own time
       object_time_step * i), find the ego path point(s) that fall
       within conflict_radius of it.
    3. The object's arrival time at that point is i * object_time_step.
       The ego's arrival time is the matching ego path point's index *
       ego_time_step.
    4. time_to_conflict = |ego_arrival_time - object_arrival_time| at
       the FIRST (smallest object arrival time) spatial overlap found.

    Returns (time_to_conflict_seconds, object_arrival_time, ego_arrival_time)
    or None if no predicted spatial overlap is found within max_horizon.
    """
    if ego_time_step is None:
        ego_time_step = object_time_step

    if ego_path_points is None:
        n_steps = max(1, round(max_horizon / ego_time_step))
        ego_path_points = []
        vx = ego_speed * math.cos(ego_heading)
        vy = ego_speed * math.sin(ego_heading)
        for i in range(1, n_steps + 1):
            t = i * ego_time_step
            ego_path_points.append([ego_pos[0] + vx * t, ego_pos[1] + vy * t])

    for obj_i, obj_pt in enumerate(object_trajectory_points, start=1):
        object_arrival_t = obj_i * object_time_step
        for ego_i, ego_pt in enumerate(ego_path_points, start=1):
            ego_arrival_t = ego_i * ego_time_step
            dist = math.hypot(obj_pt[0] - ego_pt[0], obj_pt[1] - ego_pt[1])
            if dist <= conflict_radius:
                return round(abs(ego_arrival_t - object_arrival_t), 3), round(object_arrival_t, 3), round(ego_arrival_t, 3)

    return None