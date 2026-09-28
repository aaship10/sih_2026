"""Hand-computed unit tests for multimodal/fuzzy.py's lightweight
mode-probability weighting."""
import math

from multimodal.fuzzy import DEFAULT_LATERAL_AFFINITY, compute_mode_weights


def test_stationary_object_favors_stop():
    w = compute_mode_weights(speed_ratio=0.0, heading_vol_rad=0.0, vel_vol=0.0, lateral_affinity=DEFAULT_LATERAL_AFFINITY)
    assert w["stop"] > w["nominal"]
    assert w["stop"] > w["lateral"]
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_fast_straight_stable_object_favors_nominal():
    w = compute_mode_weights(speed_ratio=1.0, heading_vol_rad=0.0, vel_vol=0.0, lateral_affinity=DEFAULT_LATERAL_AFFINITY)
    assert w["nominal"] > w["stop"]
    assert w["nominal"] > w["lateral"]


def test_erratic_heading_boosts_lateral_relative_to_straight():
    straight = compute_mode_weights(speed_ratio=0.5, heading_vol_rad=0.0, vel_vol=0.0, lateral_affinity=DEFAULT_LATERAL_AFFINITY)
    erratic = compute_mode_weights(speed_ratio=0.5, heading_vol_rad=math.pi, vel_vol=0.0, lateral_affinity=DEFAULT_LATERAL_AFFINITY)
    assert erratic["lateral"] > straight["lateral"]


def test_unstable_velocity_suppresses_nominal_relative_to_stable():
    stable = compute_mode_weights(speed_ratio=0.5, heading_vol_rad=0.0, vel_vol=0.0, lateral_affinity=DEFAULT_LATERAL_AFFINITY)
    unstable = compute_mode_weights(speed_ratio=0.5, heading_vol_rad=0.0, vel_vol=1.0, lateral_affinity=DEFAULT_LATERAL_AFFINITY)
    assert unstable["nominal"] < stable["nominal"]
    assert abs(sum(unstable.values()) - 1.0) < 1e-9


def test_lateral_affinity_scales_lateral_weight_class_dependently():
    # Same kinematic situation (moderate speed, curving heading), only the
    # per-class multiplier differs -- a motorcycle-like affinity should end
    # up with a higher lateral weight than a bus-like one.
    bus_like = compute_mode_weights(speed_ratio=0.4, heading_vol_rad=0.4, vel_vol=0.1, lateral_affinity=0.5)
    motorcycle_like = compute_mode_weights(speed_ratio=0.4, heading_vol_rad=0.4, vel_vol=0.1, lateral_affinity=1.4)
    assert motorcycle_like["lateral"] > bus_like["lateral"]


def test_weights_always_sum_to_one_across_random_inputs():
    import random
    random.seed(0)
    for _ in range(200):
        w = compute_mode_weights(
            speed_ratio=random.uniform(-0.5, 2.0),   # deliberately out-of-range to exercise clipping
            heading_vol_rad=random.uniform(-1.0, 5.0),
            vel_vol=random.uniform(-0.5, 2.0),
            lateral_affinity=random.uniform(0.3, 2.0),
        )
        assert abs(sum(w.values()) - 1.0) < 1e-6
        assert all(v >= 0.0 for v in w.values())
