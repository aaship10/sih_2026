"""Hand-computed unit tests for the constant-velocity/acceleration models."""
import numpy as np

from models.constant_acceleration import estimate_acceleration, predict as ca_predict
from models.constant_velocity import predict as cv_predict
from tracking_input.history import HistorySample


def test_constant_velocity_straight_line():
    p0 = np.array([10.0, 5.0])
    v0 = np.array([2.0, -1.0])
    traj = cv_predict(track_id=1, position=p0, velocity=v0, horizon_s=2.0, time_step_s=1.0)
    assert traj.times.tolist() == [0.0, 1.0, 2.0]
    np.testing.assert_allclose(traj.points[0], p0)
    np.testing.assert_allclose(traj.points[1], p0 + v0)
    np.testing.assert_allclose(traj.points[2], p0 + 2 * v0)


def test_constant_acceleration_matches_kinematics():
    p0 = np.array([0.0, 0.0])
    v0 = np.array([10.0, 0.0])
    a = np.array([-2.0, 0.0])  # decelerating
    traj = ca_predict(track_id=1, position=p0, velocity=v0, acceleration=a, horizon_s=2.0, time_step_s=2.0)
    # x(2) = 10*2 + 0.5*(-2)*4 = 20 - 4 = 16
    np.testing.assert_allclose(traj.points[-1], [16.0, 0.0], atol=1e-6)


def test_estimate_acceleration_from_linear_velocity_history():
    # velocity increasing by 1 m/s per second -> acceleration ~ (1, 0)
    history = [
        HistorySample(timestamp=0.0, position=np.zeros(2), velocity=np.array([0.0, 0.0]), measured=True, quality=1.0),
        HistorySample(timestamp=1.0, position=np.zeros(2), velocity=np.array([1.0, 0.0]), measured=True, quality=1.0),
        HistorySample(timestamp=2.0, position=np.zeros(2), velocity=np.array([2.0, 0.0]), measured=True, quality=1.0),
    ]
    a = estimate_acceleration(history)
    assert a is not None
    np.testing.assert_allclose(a, [1.0, 0.0], atol=1e-6)


def test_estimate_acceleration_ignores_coasted_samples():
    # only 2 MEASURED samples -- below MIN_SAMPLES_FOR_ACCELERATION=3 -- must return None
    history = [
        HistorySample(timestamp=0.0, position=np.zeros(2), velocity=np.array([0.0, 0.0]), measured=True, quality=1.0),
        HistorySample(timestamp=1.0, position=np.zeros(2), velocity=np.array([5.0, 0.0]), measured=False, quality=0.5),
        HistorySample(timestamp=2.0, position=np.zeros(2), velocity=np.array([1.0, 0.0]), measured=True, quality=1.0),
    ]
    assert estimate_acceleration(history) is None
