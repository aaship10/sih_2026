"""Class-specific prediction (section 12): NOT a separate model per class --
one shared CV/CA implementation, driven by a per-class parameter table
(config.CLASS_PROFILES: horizon, timestep, uncertainty growth rates, mode
probabilities). This is deliberately the smallest possible
design that still behaves differently per class (section 52: don't
overengineer) -- a full learned per-class model would need per-class
training data M4 does not have.

Model choice: constant velocity is PRIMARY for every class (see
constant_velocity.py's docstring). Constant acceleration is used INSTEAD OF
CV only when both are true:
  1. the class is one whose real-world motion is dominated by intentional
     speed changes over a >1s horizon (cars/trucks/buses/tempos slowing for
     traffic, NOT pedestrians/animals whose direction changes are the
     dominant unpredictability, which CA's added position term does nothing
     useful for), and
  2. at least MIN_SAMPLES_FOR_ACCELERATION usable (measured) history samples
     exist, so the acceleration estimate isn't pure noise from 2 points.
Objects that fail check 2 silently fall back to CV -- there is no "half-CA"
state.

Static classes (config.STATIC_CLASSES: road_sign, traffic_signal,
speed_bumps, traffic_cones, pothole) get neither: `predict_class_specific`
returns a single zero-velocity "nominal" trajectory (the object's current
position, repeated) with probability 1.0 and no other modes -- there is
nothing to predict, and multimodal.modes.py checks CLASS-appropriateness
before adding stop/lateral branches to a track that is already stationary.
"""
from __future__ import annotations

import numpy as np

import config
from models import constant_acceleration, constant_velocity
from models.trajectory import Trajectory, sample_times
from tracking_input.history import HistorySample

USE_ACCELERATION_CLASSES = frozenset(["car", "tempo", "truck", "bus", "rickshaw"])
MAX_SPEED_MPS_BY_CLASS = {
    "pedestrian": 3.0, "animal": 4.0, "bicycle": 8.0, "motorcycle": 25.0,
    "rickshaw": 12.0, "car": 35.0, "tempo": 25.0, "truck": 22.0, "bus": 20.0, "unknown": 20.0,
}


def get_profile(class_name: str) -> config.ClassProfile:
    return config.CLASS_PROFILES.get(class_name, config.DEFAULT_CLASS_PROFILE)


def static_trajectory(track_id: int, position: np.ndarray, horizon_s: float, time_step_s: float) -> Trajectory:
    times = sample_times(horizon_s, time_step_s)
    points = np.repeat(position[None, :], len(times), axis=0)
    return Trajectory(track_id=track_id, mode="nominal", probability=1.0, times=times, points=points)


def predict_class_specific(track_id: int, class_name: str, position: np.ndarray, velocity: np.ndarray,
                            history: list[HistorySample], is_static: bool) -> Trajectory:
    profile = get_profile(class_name)

    if class_name in config.STATIC_CLASSES or is_static:
        return static_trajectory(track_id, position, profile.horizon_s, profile.time_step_s)

    max_speed = MAX_SPEED_MPS_BY_CLASS.get(class_name, MAX_SPEED_MPS_BY_CLASS["unknown"])

    if class_name in USE_ACCELERATION_CLASSES:
        accel = constant_acceleration.estimate_acceleration(history)
        if accel is not None:
            return constant_acceleration.predict(
                track_id, position, velocity, accel, profile.horizon_s, profile.time_step_s, max_speed_mps=max_speed,
            )

    return constant_velocity.predict(track_id, position, velocity, profile.horizon_s, profile.time_step_s)
