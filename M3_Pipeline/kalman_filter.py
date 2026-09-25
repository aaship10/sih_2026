"""A small constant-velocity Kalman filter for one track.

State: [x, y, z, vx, vy, vz]^T, tracked in the CARLA world frame (meters,
m/s) -- see tracker.py for why tracking happens in world frame rather than
the (accelerating, turning) ego frame.

Measurement: [x, y, z]^T (position only). The optional radar range-rate
measurement is folded in separately via `update_radar_los`, as a genuine
*linear* Kalman measurement update on the line-of-sight velocity component
rather than a full nonlinear (EKF) range-rate measurement model -- the
line-of-sight unit vector is derived from the current position estimate
(treated as fixed for this update, not part of the state), so h(x) =
los_unit . v is linear in the state and a real KF update applies (M3 mentor
notes: avoid an EKF unless a linear KF demonstrably isn't good enough).

An earlier version of this used a fixed-fraction blend instead of a proper
Kalman update; live CARLA testing showed that produced a velocity sawtooth
whenever radar association flickered on/off frame to frame (the blend always
yanked by the same fraction regardless of how confident the filter already
was). The Kalman-weighted update below fixes that by construction: the
correction naturally shrinks as the track's velocity estimate converges.
"""
from __future__ import annotations

import numpy as np

import config


class KalmanFilter6D:
    def __init__(self, initial_position: np.ndarray, initial_velocity: np.ndarray | None = None):
        self.x = np.zeros(6)
        self.x[:3] = initial_position
        if initial_velocity is not None:
            self.x[3:] = initial_velocity
        self.P = np.eye(6) * 5.0
        self.P[3:, 3:] *= 4.0  # velocity starts more uncertain than position

        self.H = np.zeros((3, 6))
        self.H[:3, :3] = np.eye(3)
        self.R = np.eye(3) * (config.POSITION_MEASUREMENT_STD ** 2)

    def predict(self, dt: float) -> None:
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        q = config.PROCESS_NOISE_STD ** 2
        dt2, dt3, dt4 = dt ** 2, dt ** 3, dt ** 4
        Q = np.zeros((6, 6))
        for i in range(3):
            v = i + 3
            Q[i, i] = dt4 / 4 * q
            Q[i, v] = Q[v, i] = dt3 / 2 * q
            Q[v, v] = dt2 * q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, measurement_position: np.ndarray, R: np.ndarray | None = None) -> None:
        """`R` defaults to self.R (a normal, trusted position measurement).
        tracker.py passes a much larger override R for a Pass-2 cascade
        match (see config.CASCADE_POSITION_MEASUREMENT_STD): a cascade match
        already failed the primary Mahalanobis gate, so it's a genuinely
        noisier/less-certain measurement, and Kalman gain should reflect
        that -- a large R shrinks the gain, so a mismatched cascade jump
        nudges position only slightly and barely touches velocity (which
        only moves through the small position-velocity cross-covariance
        terms), instead of forcing a full-trust jump the way Pass 1
        (correctly) would for a statistically-validated match.
        """
        R = self.R if R is None else R
        y = measurement_position - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def radar_los_innovation(self, sensor_position_world: np.ndarray, range_rate_mps: float) -> tuple[float, float] | None:
        """Return (y, S) -- the innovation and its variance -- for a
        candidate radar measurement, WITHOUT mutating state. Used by
        tracker.py to gate a radar association by this track's own velocity
        uncertainty before trusting it (see that gate's docstring for why:
        proximity-only radar association can glue a fast unrelated target
        onto a nearby, unrelated track; the Kalman-weighted update alone
        doesn't protect against a confidently *wrong* measurement, only a
        noisy right one)."""
        los = self.x[:3] - sensor_position_world
        norm = np.linalg.norm(los)
        if norm < 1e-3:
            return None
        los_unit = los / norm
        target_los_speed = -range_rate_mps
        H = np.zeros(6)
        H[3:] = los_unit
        R = config.RADAR_LOS_MEASUREMENT_STD ** 2
        y = target_los_speed - H.dot(self.x)
        S = H.dot(self.P).dot(H) + R
        return float(y), float(S)

    def update_radar_los(self, sensor_position_world: np.ndarray, range_rate_mps: float) -> None:
        """Scalar Kalman measurement update on the line-of-sight velocity
        component, using the radar's range-rate as the measurement.

        h(x) = los_unit . v is linear in the state (los_unit is computed
        from the current position estimate, held fixed for this update), so
        this is a proper linear KF update: the correction magnitude is
        weighted by the filter's own uncertainty (large early on, shrinking
        as the track converges) instead of a fixed blend fraction.
        """
        los = self.x[:3] - sensor_position_world
        norm = np.linalg.norm(los)
        if norm < 1e-3:
            return
        los_unit = los / norm

        # range_rate_mps > 0 means "approaching" (config.RADAR_VELOCITY_SIGN
        # convention); the object's velocity component along los_unit (which
        # points FROM the sensor TO the object) is then negative.
        target_los_speed = -range_rate_mps

        H = np.zeros(6)
        H[3:] = los_unit
        R = config.RADAR_LOS_MEASUREMENT_STD ** 2

        y = target_los_speed - H.dot(self.x)
        S = H.dot(self.P).dot(H) + R
        K = self.P.dot(H) / S
        self.x = self.x + K * y
        self.P = self.P - np.outer(K, H).dot(self.P)

    @property
    def position(self) -> np.ndarray:
        return self.x[:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:]
