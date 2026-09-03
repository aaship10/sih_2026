"""
motion_models.py
================
Baseline (Section 10): CONSTANT VELOCITY. M3's Kalman filter already
gives clean smoothed velocity, so the baseline just projects it
forward -- no re-derivation needed. This is `constant_velocity_rollout`,
used both as the CV branch for tracked objects (multimodal.py) and as
the ego's own nominal short-horizon projection (interface.py).

Everything past the baseline is built as SEPARATE ROLLOUT FUNCTIONS
(constant acceleration / lateral deviation / stop) rather than one
"smart" model, so each stays a few lines and is independently
testable -- multimodal.py picks which ones to call and blends their
probabilities.

CLASS-SPECIFIC BEHAVIOR is kept in one flat dict. Only parameters
used by the M4 prediction branches are retained.

class strings are exact ("pedestrian", "auto_rickshaw" with the
underscore, per the integration notes) -- get_class_params() falls
back to DEFAULT_CLASS_PARAMS for anything that doesn't match, instead
of raising, so a class-string typo degrades gracefully rather than
crashing M4 mid-scenario.
"""

import math

CLASS_PARAMS = {
    "pedestrian": {
        "max_lateral_accel": 1.5,    # m/s^2 -- can swerve/step aside quickly
        "stop_decel": 1.5,           # m/s^2 -- can stop mid-stride fast
        "stop_probability": 0.15,    # Section 12: "may stop"
        "crossing_weight_base": 0.30,  # Section 12: "may cross / suddenly enter road"
    },
    "bicycle": {
        "max_lateral_accel": 2.0,
        "stop_decel": 2.5,
        "stop_probability": 0.10,
        "crossing_weight_base": 0.25,
    },
    "motorcycle": {
        "max_lateral_accel": 3.0,    # Section 12: "can move laterally quickly / filter"
        "stop_decel": 4.0,
        "stop_probability": 0.08,
        "crossing_weight_base": 0.30,  # can overtake/filter -- frequent lateral moves
    },
    "rickshaw": {
        "max_lateral_accel": 1.8,    # Section 12: "can change lateral position / merge informally"
        "stop_decel": 3.5,           # Section 12: "can slow/stop suddenly"
        "stop_probability": 0.20,
        "crossing_weight_base": 0.20,
    },
    "tempo": {
        "max_lateral_accel": 1.0,
        "stop_decel": 3.0,
        "stop_probability": 0.08,
        "crossing_weight_base": 0.12,
    },
    "car": {
        "max_lateral_accel": 1.5,
        "stop_decel": 4.5,
        "stop_probability": 0.10,
        "crossing_weight_base": 0.15,  # Section 12: "more predictable than pedestrians"
    },
    "bus": {
        "max_lateral_accel": 0.6,    # Section 12: "slower lateral movement"
        "stop_decel": 2.5,
        "stop_probability": 0.05,
        "crossing_weight_base": 0.08,
    },
    "truck": {
        "max_lateral_accel": 0.6,
        "stop_decel": 2.5,
        "stop_probability": 0.05,
        "crossing_weight_base": 0.08,
    },
    "pushcart": {
        "max_lateral_accel": 1.0,
        "stop_decel": 1.5,
        "stop_probability": 0.15,
        "crossing_weight_base": 0.20,
    },
    "animal": {
        "max_lateral_accel": 2.5,    # Section 12/33: highly uncertain, sharp direction changes
        "stop_decel": 2.0,
        "stop_probability": 0.25,    # may stop mid-crossing
        "crossing_weight_base": 0.35,
    },
        "cattle": {
        "max_lateral_accel": 2.5,
        "stop_decel": 2.0,
        "stop_probability": 0.25,
        "crossing_weight_base": 0.35,
    },
    "pothole": {
        "max_lateral_accel": 0.0,
        "stop_decel": 0.0,
        "stop_probability": 0.98,
        "crossing_weight_base": 0.01,
    },

    "speed_bumps": {
        "max_lateral_accel": 0.0,
        "stop_decel": 0.0,
        "stop_probability": 0.98,
        "crossing_weight_base": 0.01,
    },

    "barricade": {
        "max_lateral_accel": 0.0,
        "stop_decel": 0.0,
        "stop_probability": 0.98,
        "crossing_weight_base": 0.01,
    },

    "road_sign": {
        "max_lateral_accel": 0.0,
        "stop_decel": 0.0,
        "stop_probability": 0.98,
        "crossing_weight_base": 0.01,
    },

    "traffic_signal": {
        "max_lateral_accel": 0.0,
        "stop_decel": 0.0,
        "stop_probability": 0.98,
        "crossing_weight_base": 0.01,
    },

    "traffic_cones": {
        "max_lateral_accel": 0.0,
        "stop_decel": 0.0,
        "stop_probability": 0.98,
        "crossing_weight_base": 0.01,
    },

    "static_obstacle": {
        "max_lateral_accel": 0.0,
        "stop_decel": 0.0,
        "stop_probability": 0.98,
        "crossing_weight_base": 0.01,
    },

}

# Fallback for any class string that doesn't exactly match CLASS_PARAMS
# (integration notes: "must match these exactly or objects silently
# fall back to a default") -- deliberately mid-range on every axis so
# an unrecognized class degrades to "moderately cautious", not to
# either extreme.
DEFAULT_CLASS_PARAMS = {
    "max_lateral_accel": 1.2,
    "stop_decel": 2.5,
    "stop_probability": 0.12,
    "crossing_weight_base": 0.18,
}


def get_class_params(cls):
    return CLASS_PARAMS.get(cls, DEFAULT_CLASS_PARAMS)


def constant_velocity_rollout(start_xy, velocity_xy, horizon, time_step):
    """
    The baseline (Section 10). Also used as the ego's own nominal
    short-horizon projection (interface.py._nominal_ego_path) and as
    Returns n_steps = round(horizon / time_step) points, at
    t = time_step, 2*time_step, ..., horizon (does NOT include t=0).
    """
    n_steps = max(1, round(horizon / time_step))
    x0, y0 = start_xy
    vx, vy = velocity_xy
    return [
        [round(x0 + vx * i * time_step, 3), round(y0 + vy * i * time_step, 3)]
        for i in range(1, n_steps + 1)
    ]


def lateral_deviation_rollout(start_xy, velocity_xy, horizon, time_step, lateral_accel):
    """
    Branch B (Section 13, "crossing"/turn mode): same forward progress
    as CV, plus a constant lateral acceleration applied PERPENDICULAR
    to the current velocity direction -- models a pedestrian veering
    into a crossing, a motorcycle filtering sideways, an animal cutting
    across, WITHOUT needing a full bicycle-model / curvature-based turn
    (Section 11 explicitly rules out building this around lane/Frenet
    geometry, and a full curvature model is more machinery than a
    heuristic 2-3-branch predictor needs).

    lateral_accel: SIGNED magnitude (m/s^2) -- sign picked by the
    caller (multimodal._lateral_sign, using the object's own recent
    velocity history) to decide which of the two perpendicular
    directions to veer toward. Magnitude should already be clamped to
    the class's max_lateral_accel bound by the caller.
    """
    n_steps = max(1, round(horizon / time_step))
    speed = math.hypot(velocity_xy[0], velocity_xy[1])

    if speed < 1e-3:
        # No established direction of travel -- "perpendicular to
        # velocity" is undefined for a near-stationary object. The
        # stop branch already models "stays roughly put" better than
        # an arbitrary swerve axis would, so just hold position here.
        return [[round(start_xy[0], 3), round(start_xy[1], 3)] for _ in range(n_steps)]

    heading = math.atan2(velocity_xy[1], velocity_xy[0])
    perp = heading + math.pi / 2.0

    x0, y0 = start_xy
    points = []
    for i in range(1, n_steps + 1):
        t = i * time_step
        forward_dist = speed * t
        lateral_dist = 0.5 * lateral_accel * t * t
        x = x0 + math.cos(heading) * forward_dist + math.cos(perp) * lateral_dist
        y = y0 + math.sin(heading) * forward_dist + math.sin(perp) * lateral_dist
        points.append([round(x, 3), round(y, 3)])
    return points


def stop_rollout(start_xy, velocity_xy, horizon, time_step, deceleration):
    """
    Branch C (Section 13, "stop" mode): decelerate along the current
    heading at a class-bounded rate until velocity reaches zero, then
    hold position -- covers "pedestrian stops mid-crossing",
    "auto-rickshaw stops suddenly", "animal stops" (Section 12).
    """
    n_steps = max(1, round(horizon / time_step))
    speed = math.hypot(velocity_xy[0], velocity_xy[1])
    x0, y0 = start_xy

    if speed < 1e-3:
        return [[round(x0, 3), round(y0, 3)] for _ in range(n_steps)]

    heading = math.atan2(velocity_xy[1], velocity_xy[0])
    time_to_stop = speed / deceleration if deceleration > 1e-6 else 0.0

    points = []
    for i in range(1, n_steps + 1):
        t = i * time_step
        if t <= time_to_stop:
            dist = speed * t - 0.5 * deceleration * t * t
        else:
            dist = speed * time_to_stop - 0.5 * deceleration * time_to_stop * time_to_stop
        x = x0 + math.cos(heading) * dist
        y = y0 + math.sin(heading) * dist
        points.append([round(x, 3), round(y, 3)])
    return points
