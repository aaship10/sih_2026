"""Verifies the quintic sampler actually satisfies all 6 boundary
conditions (the whole point of using a 5th-degree polynomial) and that
plan_quintic_path produces exactly N waypoints with consistent yaw/speed."""
import math

from quintic import plan_quintic_path, solve_quintic


def test_boundary_conditions_satisfied_exactly():
    q = solve_quintic(p0=0.0, v0=2.0, a0=0.5, pT=20.0, vT=5.0, aT=0.0, T=4.0)
    assert math.isclose(q.position(0.0), 0.0, abs_tol=1e-9)
    assert math.isclose(q.velocity(0.0), 2.0, abs_tol=1e-9)
    assert math.isclose(q.acceleration(0.0), 0.5, abs_tol=1e-9)
    assert math.isclose(q.position(4.0), 20.0, abs_tol=1e-6)
    assert math.isclose(q.velocity(4.0), 5.0, abs_tol=1e-6)
    assert math.isclose(q.acceleration(4.0), 0.0, abs_tol=1e-6)


def test_zero_duration_falls_back_without_crashing():
    q = solve_quintic(p0=1.0, v0=2.0, a0=0.0, pT=99.0, vT=99.0, aT=99.0, T=0.0)
    assert q.position(0.0) == 1.0
    assert q.velocity(0.0) == 2.0


def test_plan_quintic_path_shape_and_endpoints():
    waypoints = plan_quintic_path(
        start_xy=(0.0, 0.0), start_vel_xy=(5.0, 0.0), start_acc_xy=(0.0, 0.0),
        target_xy=(15.0, 0.0), target_vel_xy=(5.0, 0.0), target_acc_xy=(0.0, 0.0),
        horizon_s=3.0, n_points=16,
    )
    assert len(waypoints) == 16
    assert math.isclose(waypoints[0].x, 0.0, abs_tol=1e-6)
    assert math.isclose(waypoints[-1].x, 15.0, abs_tol=1e-6)
    assert math.isclose(waypoints[-1].t, 3.0, abs_tol=1e-9)
    # straight line along +x the whole way -> yaw should stay ~0
    assert all(abs(w.yaw) < 1e-6 for w in waypoints)


def test_lateral_offset_path_curves_and_returns():
    waypoints = plan_quintic_path(
        start_xy=(0.0, 0.0), start_vel_xy=(5.0, 0.0), start_acc_xy=(0.0, 0.0),
        target_xy=(15.0, 3.0), target_vel_xy=(5.0, 0.0), target_acc_xy=(0.0, 0.0),
        horizon_s=3.0, n_points=16,
    )
    ys = [w.y for w in waypoints]
    assert ys[0] < ys[len(ys) // 2] <= ys[-1] + 1e-6  # y increases toward the offset target
    assert math.isclose(waypoints[-1].y, 3.0, abs_tol=1e-6)
