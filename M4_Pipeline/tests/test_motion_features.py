"""Hand-computed unit tests for multimodal/features.py's windowed motion
feature extraction (smoothing + volatility)."""
import numpy as np

from multimodal.features import compute_motion_features
from tracking_input.history import HistorySample


def _sample(t, v, measured=True, pos=(0.0, 0.0)):
    return HistorySample(timestamp=t, position=np.array(pos, dtype=float), velocity=np.array(v), measured=measured, quality=1.0)


def test_constant_velocity_history_smooths_to_the_same_value_zero_volatility():
    # position advances at exactly 5.0 m/s in x, matching the velocity readings.
    history = [
        _sample(0.0, [5.0, 0.0], pos=[0.0, 0.0]),
        _sample(0.1, [5.0, 0.0], pos=[0.5, 0.0]),
        _sample(0.2, [5.0, 0.0], pos=[1.0, 0.0]),
        _sample(0.3, [5.0, 0.0], pos=[1.5, 0.0]),
    ]
    f = compute_motion_features(history, current_velocity=np.array([5.0, 0.0]), window_n=5)
    np.testing.assert_allclose(f.smoothed_velocity, [5.0, 0.0], atol=1e-9)
    assert f.speed_volatility == 0.0
    assert f.heading_volatility_rad == 0.0
    assert f.n_samples_used == 4


def test_oscillating_velocity_readings_still_recover_true_speed_from_position_trend():
    # Reproduces the second live-data failure case (see module docstring):
    # M3's own velocity reading oscillates in sign frame-to-frame for a
    # genuinely, steadily-moving object. A mean of these raw readings would
    # cancel toward ~0; fitting velocity from the observed position trend
    # instead recovers the real ~3 m/s motion.
    history = [
        _sample(0.0, [-0.5, 0.0], pos=[0.0, 0.0]),
        _sample(0.1, [0.6, 0.0], pos=[0.3, 0.0]),
        _sample(0.2, [-0.4, 0.0], pos=[0.6, 0.0]),
        _sample(0.3, [0.5, 0.0], pos=[0.9, 0.0]),
    ]
    f = compute_motion_features(history, current_velocity=np.array([0.5, 0.0]), window_n=5)
    raw_mean = 0.05  # (-0.5+0.6-0.4+0.5)/4 -- what the old mean-of-readings approach produced
    assert f.smoothed_velocity[0] > 2.0
    assert abs(f.smoothed_velocity[0] - raw_mean) > 2.0


def test_jumping_speed_history_has_nonzero_speed_volatility():
    # Reproduces the live-data failure case: speed climbing frame to frame
    # while direction stays constant.
    history = [_sample(0.0, [0.8, 0.0]), _sample(0.1, [2.9, 0.0]), _sample(0.2, [5.8, 0.0])]
    f = compute_motion_features(history, current_velocity=np.array([5.8, 0.0]), window_n=5)
    assert f.speed_volatility > 1.0  # meters/sec std-dev, clearly nonzero
    assert f.heading_volatility_rad == 0.0  # direction never changed


def test_turning_history_has_nonzero_heading_volatility():
    history = [_sample(0.0, [5.0, 0.0]), _sample(0.1, [3.5, 3.5]), _sample(0.2, [0.0, 5.0])]
    f = compute_motion_features(history, current_velocity=np.array([0.0, 5.0]), window_n=5)
    assert f.heading_volatility_rad > 0.5  # roughly a 90-degree turn split over 2 steps


def test_window_n_limits_how_many_samples_are_used():
    history = [_sample(float(i), [1.0, 0.0]) for i in range(10)]
    f = compute_motion_features(history, current_velocity=np.array([1.0, 0.0]), window_n=3)
    assert f.n_samples_used == 3


def test_coasted_samples_are_excluded():
    history = [
        _sample(0.0, [5.0, 0.0], measured=True, pos=[0.0, 0.0]),
        _sample(0.1, [50.0, 50.0], measured=False, pos=[99.0, 99.0]),  # a coasted/dead-reckoned sample -- should be ignored
        _sample(0.2, [5.0, 0.0], measured=True, pos=[1.0, 0.0]),
    ]
    f = compute_motion_features(history, current_velocity=np.array([5.0, 0.0]), window_n=5)
    assert f.n_samples_used == 2  # only the 2 measured samples
    np.testing.assert_allclose(f.smoothed_velocity, [5.0, 0.0], atol=1e-9)


def test_too_few_samples_passes_current_velocity_through_unchanged():
    history = [_sample(0.0, [3.0, 4.0])]
    current = np.array([9.0, 9.0])
    f = compute_motion_features(history, current_velocity=current, window_n=5)
    np.testing.assert_allclose(f.smoothed_velocity, current)
    assert f.speed_volatility == 0.0
    assert f.n_samples_used == 1
