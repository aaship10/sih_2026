"""
collision.py
============
Practical estimate of P(collision | predictions), suitable for a
student project (Section 19): NOT full Monte-Carlo sampling of a
learned distribution -- instead, weight each of the object's 2-3
multimodal branches (multimodal.py) by whether it comes within the
combined collision radius (ttc.py) of the ego's nominal path at any
matching timestep, and sum the probabilities of the colliding branches.

    collision_probability = sum(branch.probability for branch in branches
                                 if branch comes within combined_radius
                                 of ego's nominal path at some matching t)

This is a discrete, explainable version of the "N trajectories / count
overlaps" method the brief describes -- here N=2 or 3 (the multimodal
branches) instead of hundreds of Monte-Carlo samples, which keeps it
fast enough for real-time and easy to justify in the report. It's a
reasonable simplification as long as the branch weights themselves are
reasonable (see multimodal.py) -- flag in the report that this trades
sampling density for speed/explainability.
"""

import math


def branch_min_distance_to_ego_path(branch_points, ego_path_points, time_step,
                                      ego_time_step=None):
    """
    Returns the minimum distance (meters) between the branch's
    predicted points and the ego's nominal path points at MATCHING
    timesteps (same index -> same predicted time), plus the timestep
    index at which that minimum occurs.
    """
    if ego_time_step is None:
        ego_time_step = time_step

    best_dist = float("inf")
    best_i = None
    n = min(len(branch_points), len(ego_path_points))
    for i in range(n):
        bx, by = branch_points[i]
        ex, ey = ego_path_points[i]
        d = math.hypot(bx - ex, by - ey)
        if d < best_dist:
            best_dist = d
            best_i = i
    return best_dist, best_i


def estimate_collision_probability(branches, ego_path_points, time_step, combined_radius,
                                    ego_time_step=None):
    """
    branches: output of multimodal.generate_branches()
    ego_path_points: nominal nominal nominal ego rollout (see ttc.py)
    combined_radius: from ttc.combined_radius()

    Returns: (collision_probability, per_branch_detail) where
    per_branch_detail is a list of {"mode", "probability", "min_distance",
    "collides": bool} for transparency/debugging/report figures.
    """
    total_prob = 0.0
    detail = []
    for branch in branches:
        min_dist, _ = branch_min_distance_to_ego_path(
            branch["points"], ego_path_points, time_step, ego_time_step
        )
        collides = min_dist <= combined_radius
        if collides:
            total_prob += branch["probability"]
        detail.append({
            "mode": branch.get("mode", "?"),
            "probability": branch["probability"],
            "min_distance": round(min_dist, 3) if min_dist != float("inf") else None,
            "collides": collides,
        })

    return round(min(total_prob, 1.0), 3), detail