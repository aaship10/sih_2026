"""
multimodal.py
==============
Section 13's simplest practical multi-modal implementation: 2-3 fixed
BRANCHES per object (straight / crossing / stop), each built with one
of motion_models.py's separate rollout functions, weighted heuristically
from the object's class (motion_models.CLASS_PARAMS) and its recent
buffered history (derive.py's lateral-velocity-trend signal) -- no
sampling network, no learned intent classifier, per the "don't
overengineer" instruction.

Branches produced (mode names match Section 13's example):
    "straight" -- motion_models.constant_velocity_rollout
    "crossing" -- motion_models.lateral_deviation_rollout, direction
                  picked by the object's own recent sideways motion
    "stop"     -- motion_models.stop_rollout

WEIGHTING:
    stop_probability and crossing_weight_base come straight from the
    class's CLASS_PARAMS entry (Section 12's per-class behavior).
    crossing_weight_base is then boosted when derive.py's
    estimate_lateral_velocity_trend() shows this object's sideways
    speed picking up RIGHT NOW -- the concrete "if lateral velocity
    component is growing over the last few frames, upweight the
    crossing branch" design note, and the mechanism behind Sections
    32/33 (sudden pedestrian crossing, cattle changing direction).
    Whatever's left goes to "straight". All three are renormalized to
    sum to exactly 1.0 (probabilities from CLASS_PARAMS + a boost can
    drift slightly off from rounding/clamping).
"""

STATIC_CLASSES = {
    "pothole",
    "speed_bumps",
    "barricade",
    "road_sign",
    "traffic_signal",
    "traffic_cones",
    "static_obstacle",
}

from motion_models import (
    get_class_params,
    constant_velocity_rollout,
    lateral_deviation_rollout,
    stop_rollout,
)

# Below this |vy| (m/s), we don't trust the sign of recent lateral
# motion enough to pick a swerve direction from it -- same spirit as
# derive.HEADING_SPEED_THRESHOLD, just applied to the lateral
# component specifically rather than full speed.
LATERAL_SIGN_SPEED_THRESHOLD = 0.15

# How much a growing lateral trend (m/s per second, from
# derive.estimate_lateral_velocity_trend) can boost the crossing
# branch's probability above its class baseline.
LATERAL_TREND_BOOST_THRESHOLD = 0.3   # m/s^2 -- below this, treat as noise, no boost
LATERAL_TREND_BOOST_GAIN = 0.6        # probability added per (m/s^2) of trend above threshold
LATERAL_TREND_BOOST_MAX = 0.35        # cap on how much the trend alone can add

# Keep at least this much probability mass on "straight" even for a
# class/situation that strongly favors crossing+stop -- a predictor
# that can put ~0 weight on "keeps doing what it's doing" is
# overconfident, and multiple branches with ~0 probability are
# useless to M5 downstream.
MIN_STRAIGHT_PROBABILITY = 0.10


def _lateral_sign(track_history):
    """
    Which of the two perpendicular directions (relative to current
    heading) the object has actually been drifting toward, scanning
    backward through the buffer for the most recent frame with a
    trustworthy sideways velocity component. Falls back to +1 if the
    object has no meaningful sideways history yet (an arbitrary but
    consistent choice -- with no evidence either way, direction is a
    coin flip, and the "crossing" branch's probability weight already
    reflects that this is a low-confidence branch, not a strong claim).
    """
    for frame in reversed(track_history):
        vy = frame["velocity"][1]
        if abs(vy) > LATERAL_SIGN_SPEED_THRESHOLD:
            return 1.0 if vy > 0 else -1.0
    return 1.0


def _branch_weights(cls, lateral_trend):
    """Returns (straight_prob, crossing_prob, stop_prob), summing to 1.0."""
    params = get_class_params(cls)
    stop_prob = params["stop_probability"]
    crossing_prob = params["crossing_weight_base"]

    if lateral_trend > LATERAL_TREND_BOOST_THRESHOLD:
        boost = min(
            LATERAL_TREND_BOOST_MAX,
            (lateral_trend - LATERAL_TREND_BOOST_THRESHOLD) * LATERAL_TREND_BOOST_GAIN,
        )
        crossing_prob += boost

    # Keep a minimum straight probability for moving objects.
    # Static/road-surface objects are allowed to put nearly all
    # probability on the stop branch because they do not move.
    if cls in STATIC_CLASSES:
        straight_prob = max(0.0, 1.0 - stop_prob - crossing_prob)
    else:
        straight_prob = max(
            MIN_STRAIGHT_PROBABILITY,
            1.0 - stop_prob - crossing_prob
        )

    total = straight_prob + crossing_prob + stop_prob
    return straight_prob / total, crossing_prob / total, stop_prob / total


def generate_branches(position_xy, velocity_xy, cls, track_history, horizon, time_step,
                       lateral_trend=0.0):
    """
    Returns a list of 3 branches: [{"mode", "points", "probability"}, ...].

    position_xy / velocity_xy: latest smoothed M3 state for this track.
    track_history: buffer.py's per-track deque (oldest -> newest) --
        used here only to pick the crossing branch's swerve direction
        (_lateral_sign); trend strength itself is passed in already
        computed (derive.estimate_lateral_velocity_trend), so it isn't
        recomputed here.
    """
    params = get_class_params(cls)
    straight_prob, crossing_prob, stop_prob = _branch_weights(cls, lateral_trend)

    straight_points = constant_velocity_rollout(position_xy, velocity_xy, horizon, time_step)

    sign = _lateral_sign(track_history)
    crossing_points = lateral_deviation_rollout(
        position_xy, velocity_xy, horizon, time_step,
        lateral_accel=sign * params["max_lateral_accel"],
    )

    stop_points = stop_rollout(
        position_xy, velocity_xy, horizon, time_step,
        deceleration=params["stop_decel"],
    )

    return [
        {"mode": "straight", "points": straight_points, "probability": round(straight_prob, 3)},
        {"mode": "crossing", "points": crossing_points, "probability": round(crossing_prob, 3)},
        {"mode": "stop", "points": stop_points, "probability": round(stop_prob, 3)},
    ]