"""Hand-computed cases for ttc.py's relative-kinematics TTC."""
import math

from ttc import compute_ttc, min_ttc


def test_head_on_closing_gives_exact_ttc():
    # Ego at origin moving +x at 5 m/s, obstacle 50m ahead stationary.
    # Closing speed = 5 m/s -> TTC = 50/5 = 10s.
    t = compute_ttc((0.0, 0.0), (5.0, 0.0), (50.0, 0.0), (0.0, 0.0))
    assert t == 10.0


def test_moving_away_gives_no_ttc():
    t = compute_ttc((0.0, 0.0), (0.0, 0.0), (50.0, 0.0), (5.0, 0.0))
    assert t is None


def test_both_moving_same_direction_same_speed_not_closing():
    # No relative motion at all -> not closing -> None.
    t = compute_ttc((0.0, 0.0), (5.0, 0.0), (20.0, 0.0), (5.0, 0.0))
    assert t is None


def test_zero_relative_velocity_gives_no_ttc_even_when_laterally_offset():
    # Obstacle to the side, moving at the SAME velocity as ego (zero
    # relative motion) -- the straight-line sightline distance never
    # shrinks, so this reads as "no TTC" even though a later lane change
    # could bring them together. Known limitation (spec item E): this is
    # why planner.py ALSO checks predicted trajectories, not just this
    # formula, for a conflict TTC alone can't see.
    t = compute_ttc((0.0, 0.0), (5.0, 0.0), (0.0, 10.0), (5.0, 0.0))
    assert t is None


def test_already_overlapping_gives_zero():
    t = compute_ttc((10.0, 10.0), (1.0, 0.0), (10.0, 10.0), (0.0, 0.0))
    assert t == 0.0


def test_min_ttc_picks_the_smallest_across_multiple_obstacles():
    obstacles = [
        ((100.0, 0.0), (0.0, 0.0)),   # far, TTC = 100/5 = 20s
        ((20.0, 0.0), (0.0, 0.0)),    # close, TTC = 20/5 = 4s
        ((50.0, 0.0), (5.0, 0.0)),    # not closing -> excluded
    ]
    result = min_ttc((0.0, 0.0), (5.0, 0.0), obstacles)
    assert math.isclose(result, 4.0)


def test_min_ttc_returns_none_when_nothing_is_closing():
    obstacles = [((50.0, 0.0), (5.0, 0.0)), ((0.0, 30.0), (0.0, 5.0))]
    assert min_ttc((0.0, 0.0), (5.0, 0.0), obstacles) is None
