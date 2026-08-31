"""
interface.py
============
The M4 -> M5 contract, and the single predict() entrypoint M5 (or
main.py during standalone testing) calls per frame per object. M5
consumes this dict without knowing anything about Kalman filters,
branch generation, or covariance -- just "what will it do, how sure
are we, how dangerous is it" (Section 41).

PREDICTION_HORIZON / TIME_STEP (Section 9):
    horizon = 3.0s, time_step = 0.2s (matches M3's own DT, so M4's
    predicted points land on the same cadence M3 already ticks at --
    no resampling needed downstream).

    Why 3.0s: long enough to see a pedestrian actually finish crossing
    or a truck complete a slow merge, short enough that constant-
    velocity/heuristic branches (no learned intent model) don't drift
    into fantasy. Per-class reasoning:
        pedestrian/animal   -- 3s is enough to cross one lane; beyond
                                that, "will they even still be walking
                                the same direction" is pure guesswork
                                for a heuristic model.
        motorcycle/bicycle  -- fast enough that 3s covers a full
                                overtake or lane-filter maneuver.
        car/auto-rickshaw   -- 3s covers a merge or a stop decision.
        truck/bus           -- slower dynamics; 3s is on the short side
                                for a full lane change but sufficient
                                for M5's *local* replanning horizon,
                                which is the actual consumer here.
    time_step = 0.2s: matches M3's tracking rate (5 Hz) -- finer
    doesn't add real information given the motion models are
    heuristic, coarser would make TTC/time-to-conflict resolution too
    chunky for close-range pedestrian scenarios.

M4 does NOT need (Section 7's open question):
    - Drivable area / road boundaries / map: explicitly avoided --
      Section 11 rules out lane/Frenet-based prediction, and M4's
      motion models (CV + class-bounded lateral/stop) don't reference
      road geometry at all. M5 is the one that reasons about
      drivable space when turning predictions into a plan.
    - Static obstacles / goal: not M4's concern (M5 territory).
    M4 DOES need: ego_state (position/velocity/heading) purely to
      compute risk (TTC, time-to-conflict, collision probability)
      against a NOMINAL ego projection -- explicitly not M5's planned
      path, to avoid the circular M4<->M5 dependency (Section 23-24).
"""

import math

from derive import estimate_heading, estimate_acceleration, estimate_lateral_velocity_trend
from motion_models import constant_velocity_rollout
from multimodal import generate_branches
from uncertainty import trajectory_uncertainty, occupancy_radius, confidence_to_sigma0
from ttc import combined_radius, time_to_collision, time_to_conflict
from collision import estimate_collision_probability
from risk import compute_risk_score, risk_level_from_score

PREDICTION_HORIZON = 3.0
TIME_STEP = 0.2


def _normalize_ego_state(ego_state):
    """
    ego_state["velocity"] may arrive as either a scalar speed (paired
    with ego_state["heading"]) -- the shape dummy_ego.py produces and
    the shape M1/M5 are expected to send -- OR as a [vx, vy] vector,
    which is what some test/dummy data (and, notably, radar_processing.py's
    convention for TRACKED OBJECTS) uses instead. M3/radar's [vx, vy]
    convention is for tracked objects, not ego, but nothing stops a
    caller from handing ego_state in that same shape, and getting this
    wrong is exactly what broke earlier (TypeError multiplying a list
    by a float). Normalize once, here, so every line below this can
    assume scalar speed + heading without re-checking.

    Returns a new dict: {"position", "velocity": <scalar speed>,
    "heading": <radians>, "timestamp"} -- same keys predict() already
    expects, so nothing downstream needs to change.
    """
    velocity = ego_state["velocity"]

    if isinstance(velocity, (list, tuple)):
        vx, vy = velocity[0], velocity[1]
        speed = math.hypot(vx, vy)
        # prefer the vector's own direction; fall back to an explicit
        # "heading" field only when the vector is ~stationary (heading
        # from a near-zero vector is noise, same reasoning as
        # derive.estimate_heading's stopped-speed guard)
        if speed > 1e-6:
            heading = math.atan2(vy, vx)
        else:
            heading = ego_state.get("heading", 0.0)
    else:
        speed = velocity
        heading = ego_state.get("heading", 0.0)

    return {
        "position": ego_state["position"],
        "velocity": speed,
        "heading": heading,
        "timestamp": ego_state.get("timestamp"),
    }


def _nominal_ego_path(ego_state, horizon, time_step):
    x0, y0 = ego_state["position"][0], ego_state["position"][1]
    speed = ego_state["velocity"]
    heading = ego_state["heading"]
    vx, vy = speed * math.cos(heading), speed * math.sin(heading)
    return constant_velocity_rollout([x0, y0], [vx, vy], horizon, time_step)


def predict(track_id, cls, track_history, ego_state, confidence,
            horizon=PREDICTION_HORIZON, time_step=TIME_STEP,
            ego_half_width=1.0):
    """
    The M4 entrypoint. Call once per object per frame.

    track_history: this object's buffered M3 snapshots (buffer.py),
        oldest -> newest, most-recent-last.
    ego_state: {"position": [x,y], "velocity": ..., "heading": rad,
        "timestamp": t} -- from dummy_ego.py during standalone dev, or
        M1/M5's real ego state later. "velocity" may be a scalar speed
        (paired with "heading") or a [vx, vy] vector -- both are
        normalized internally by _normalize_ego_state(), so callers
        don't need to convert before calling predict(). Used ONLY for
        a nominal constant-velocity/heading ego projection, never M5's
        planned path (keeps the M4->M5 dependency one-directional).
    confidence: M3's track confidence (already smoothed by M3).

    Returns the M4 -> M5 prediction dict (see module docstring).
    """
    if not track_history:
        raise ValueError(f"predict() called with empty history for track {track_id}")

    ego_state = _normalize_ego_state(ego_state)

    latest = track_history[-1]
    position_xy = latest["position"][:2]
    velocity_xy = latest["velocity"][:2]
    size_xyz = latest.get("size") or [0.5, 0.5, 0.5]

    heading = estimate_heading(track_history)
    accel_xy = estimate_acceleration(track_history)
    lateral_trend = estimate_lateral_velocity_trend(track_history)

    # 1. multi-modal branches
    branches = generate_branches(
        position_xy, velocity_xy, cls, track_history, horizon, time_step,
        lateral_trend=lateral_trend,
    )

    # 2. uncertainty per predicted timestep (applies uniformly across branches
    #    for this v0 -- see uncertainty.py for why isotropic-growing was chosen)
    sigmas = trajectory_uncertainty(cls, confidence, horizon, time_step)
    mean_sigma = sum(sigmas) / len(sigmas)

    # 3. risk geometry: nominal ego path + inflated collision radius
    ego_path = _nominal_ego_path(ego_state, horizon, time_step)
    radius = combined_radius(size_xyz, ego_half_width=ego_half_width)

    # 4. TTC (closing-velocity) first; falls back to time-to-conflict
    #    for crossing objects where TTC doesn't apply (see ttc.py)
    ego_vel_xy = [
        ego_state["velocity"] * math.cos(ego_state["heading"]),
        ego_state["velocity"] * math.sin(ego_state["heading"]),
    ]
    ttc = time_to_collision(ego_state["position"], ego_vel_xy, position_xy, velocity_xy, radius)

    conflict_result = None
    if ttc is None:
        # use the highest-probability branch's points for conflict search
        primary_branch = max(branches, key=lambda b: b["probability"])
        conflict_result = time_to_conflict(
            ego_state["position"], ego_state["heading"], ego_state["velocity"],
            primary_branch["points"], time_step, radius,
            ego_path_points=ego_path, ego_time_step=time_step,
        )
    time_to_conflict_seconds = conflict_result[0] if conflict_result else None
    effective_ttc = ttc if ttc is not None else time_to_conflict_seconds

    # 5. collision probability across branches
    collision_probability, branch_detail = estimate_collision_probability(
        branches, ego_path, time_step, radius, ego_time_step=time_step
    )

    # 6. min distance across branches (for the distance risk term)
    min_distance = min(
        (d["min_distance"] for d in branch_detail if d["min_distance"] is not None),
        default=None,
    )

    # 7. risk score + level
    risk_score = compute_risk_score(
        collision_probability, effective_ttc, min_distance, mean_sigma, cls
    )
    risk_level = risk_level_from_score(risk_score)

    # 8. prediction confidence: floor at M3's own confidence (Section:
    #    "don't be more certain than M3 was"), reduced when the branches
    #    disagree a lot (high entropy = harder to say what will happen)
    branch_probs = [b["probability"] for b in branches]
    max_branch_prob = max(branch_probs)
    prediction_confidence = round(confidence * (0.5 + 0.5 * max_branch_prob), 3)

    return {
        "track_id": track_id,
        "class": cls,
        "trajectories": [
            {"mode": b["mode"], "points": b["points"], "probability": b["probability"]}
            for b in branches
        ],
        "prediction_horizon": horizon,
        "time_step": time_step,
        "uncertainty": {
            "sigma_per_step": sigmas,
            "occupancy_radius_per_step": [occupancy_radius(s) for s in sigmas],
        },
        "collision_probability": collision_probability,
        "minimum_ttc": ttc,
        "time_to_conflict": time_to_conflict_seconds,
        "min_predicted_distance_to_ego": round(min_distance, 3) if min_distance is not None else None,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "confidence": prediction_confidence,
        "debug": {
            "heading": round(heading, 3),
            "acceleration": [round(accel_xy[0], 3), round(accel_xy[1], 3)],
            "branch_detail": branch_detail,
        },
    }