"""Multi-modal prediction (section 13): the SIMPLEST practical implementation
-- rule-based branching into up to config.N_MODES=3 named trajectories per
object, not a learned mixture model. This is deliberately not more than 3
modes (section 52: don't overengineer):

  "nominal"  -- the class-specific CV/CA trajectory (models/class_specific.py),
                always generated.
  "stop"     -- decelerate to a stop at the class's stop_mode_decel_mps2,
                along the CURRENT heading, then hold position. Represents
                "it stops" (a pedestrian hesitating, a vehicle braking).
  "lateral"  -- heading rotates at up to max_turn_rate_dps while speed is
                held constant. Represents "it turns/crosses" -- the
                non-lane-based case section 11 is about.

Static objects (config.STATIC_CLASSES, or is_static=True) get ONLY
"nominal" (already the single stationary trajectory from
class_specific.static_trajectory) -- there is nothing to branch.

Lateral-mode turn direction: since M4 has no map/lane information to know
which way an object would "naturally" turn, the lateral mode's sign is
chosen to point TOWARD the ego's current position rather than away from it.
This is a deliberate, documented conservative bias, not an attempt at an
unbiased forecast: M4's purpose is to surface potential conflicts for M5
(section 6), and a lateral branch that curves away from the ego is, by
construction, never the dangerous case M5 needs to plan against. If a track
has a discernible recent turning trend in its history (heading changing
consistently in one direction over the last few measured samples), that
trend is used instead of the toward-ego heuristic -- an actual observed
trend is better evidence than the conservative default.

Probabilities: computed by multimodal/fuzzy.py's lightweight fuzzy-logic
system from three windowed features (multimodal/features.py) -- current
speed (relative to the class's own max plausible speed), recent heading
volatility ("is it curving/erratic"), and recent VELOCITY volatility ("has
its reported speed been jumping around" -- a track whose recent readings
disagree with each other is trusted less, regardless of its current
instantaneous value) -- plus a per-class lateral-affinity multiplier
(fuzzy.LATERAL_AFFINITY_BY_CLASS: e.g. a motorcycle is inherently weighted
more toward sudden lateral movement than a bus). Below
config.MIN_SAMPLES_FOR_FUZZY measured history samples, there isn't enough
data for a meaningful volatility estimate, so this falls back to the
original static per-class `profile.mode_probabilities` + a crisp
slow-speed (<0.3 m/s) reweight -- mirrors models/constant_acceleration.py's
own history-length fallback pattern. Probabilities are always renormalized
to sum to 1.0.
"""
from __future__ import annotations

import math

import numpy as np

import config
from models.class_specific import MAX_SPEED_MPS_BY_CLASS
from models.trajectory import Trajectory, sample_times
from multimodal import fuzzy
from multimodal.features import MotionFeatures
from tracking_input.history import HistorySample

SLOW_SPEED_THRESHOLD_MPS = 0.3
TURN_TREND_MIN_SAMPLES = 3
TURN_TREND_MIN_HEADING_CHANGE_RAD = math.radians(8.0)  # per sample, to count as a real trend and not noise


def _stop_trajectory(track_id: int, position: np.ndarray, velocity: np.ndarray,
                      decel_mps2: float, horizon_s: float, time_step_s: float) -> Trajectory:
    times = sample_times(horizon_s, time_step_s)
    points = np.zeros((len(times), 2))
    speed = float(np.linalg.norm(velocity))
    direction = velocity / speed if speed > 1e-3 else np.zeros(2)
    p = position.copy()
    prev_t = 0.0
    v = speed
    for i, t in enumerate(times):
        dt = t - prev_t
        v_new = max(0.0, v - decel_mps2 * dt)
        avg_v = (v + v_new) / 2.0
        p = p + direction * avg_v * dt
        v = v_new
        points[i] = p
        prev_t = t
    return Trajectory(track_id=track_id, mode="stop", probability=0.0, times=times, points=points)


def _recent_turn_rate_rad_s(history: list[HistorySample]) -> float | None:
    measured = [s for s in history if s.measured and np.linalg.norm(s.velocity) > SLOW_SPEED_THRESHOLD_MPS]
    if len(measured) < TURN_TREND_MIN_SAMPLES:
        return None
    headings = [math.atan2(s.velocity[1], s.velocity[0]) for s in measured[-TURN_TREND_MIN_SAMPLES:]]
    times = [s.timestamp for s in measured[-TURN_TREND_MIN_SAMPLES:]]
    diffs = []
    for i in range(1, len(headings)):
        dh = math.atan2(math.sin(headings[i] - headings[i - 1]), math.cos(headings[i] - headings[i - 1]))
        dt = times[i] - times[i - 1]
        if dt > 1e-3:
            diffs.append(dh / dt)
    if not diffs:
        return None
    mean_rate = sum(diffs) / len(diffs)
    if abs(mean_rate) * (times[-1] - times[0]) / max(1, len(diffs)) < TURN_TREND_MIN_HEADING_CHANGE_RAD:
        return None
    return mean_rate


def _lateral_trajectory(track_id: int, position: np.ndarray, velocity: np.ndarray,
                         ego_position: np.ndarray, max_turn_rate_dps: float,
                         horizon_s: float, time_step_s: float, history: list[HistorySample]) -> Trajectory:
    times = sample_times(horizon_s, time_step_s)
    speed = float(np.linalg.norm(velocity))
    heading0 = math.atan2(velocity[1], velocity[0]) if speed > 1e-3 else 0.0

    trend = _recent_turn_rate_rad_s(history)
    if trend is not None:
        turn_rate = max(-math.radians(max_turn_rate_dps), min(math.radians(max_turn_rate_dps), trend))
    else:
        # Conservative default: turn toward the ego's current position (see
        # module docstring) rather than an arbitrary/random side.
        to_ego = ego_position - position
        left_normal = np.array([-velocity[1], velocity[0]]) if speed > 1e-3 else np.array([0.0, 1.0])
        sign = 1.0 if np.dot(to_ego, left_normal) > 0 else -1.0
        turn_rate = sign * math.radians(max_turn_rate_dps)

    points = np.zeros((len(times), 2))
    p = position.copy()
    prev_t = 0.0
    heading = heading0
    for i, t in enumerate(times):
        dt = t - prev_t
        heading += turn_rate * dt
        p = p + speed * np.array([math.cos(heading), math.sin(heading)]) * dt
        points[i] = p
        prev_t = t
    return Trajectory(track_id=track_id, mode="lateral", probability=0.0, times=times, points=points)


def generate_modes(track_id: int, class_name: str, position: np.ndarray, velocity: np.ndarray,
                    nominal: Trajectory, profile: "config.ClassProfile", history: list[HistorySample],
                    ego_position: np.ndarray, is_static: bool, features: MotionFeatures) -> list[Trajectory]:
    if class_name in config.STATIC_CLASSES or is_static:
        nominal.probability = 1.0
        return [nominal]

    speed = float(np.linalg.norm(velocity))

    if features.n_samples_used >= config.MIN_SAMPLES_FOR_FUZZY:
        max_speed = MAX_SPEED_MPS_BY_CLASS.get(class_name, MAX_SPEED_MPS_BY_CLASS["unknown"])
        probs = fuzzy.compute_mode_weights(
            speed_ratio=speed / max_speed,
            heading_vol_rad=features.heading_volatility_rad,
            vel_vol=features.speed_volatility / max_speed,
            lateral_affinity=fuzzy.LATERAL_AFFINITY_BY_CLASS.get(class_name, fuzzy.DEFAULT_LATERAL_AFFINITY),
        )
    else:
        probs = dict(profile.mode_probabilities)
        if speed < SLOW_SPEED_THRESHOLD_MPS:
            probs = {"nominal": probs["nominal"] * 0.3, "stop": probs["stop"] + probs["nominal"] * 0.5 + probs["lateral"] * 0.5,
                     "lateral": probs["lateral"] * 0.3}
        total = sum(probs.values()) or 1.0
        probs = {k: v / total for k, v in probs.items()}

    nominal.probability = probs["nominal"]
    modes = [nominal]

    stop = _stop_trajectory(track_id, position, velocity, profile.stop_mode_decel_mps2, profile.horizon_s, profile.time_step_s)
    stop.probability = probs["stop"]
    modes.append(stop)

    lateral = _lateral_trajectory(track_id, position, velocity, ego_position, profile.max_turn_rate_dps,
                                   profile.horizon_s, profile.time_step_s, history)
    lateral.probability = probs["lateral"]
    modes.append(lateral)

    return modes
