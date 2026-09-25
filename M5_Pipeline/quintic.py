"""Quintic polynomial trajectory sampler (spec section 2). A 5th-degree
polynomial per axis is the lowest degree that can satisfy all 6 boundary
conditions (start/target position, velocity, acceleration) exactly, which
is what makes the resulting path C2-continuous (no velocity or
acceleration discontinuity at either end) -- i.e. jerk-limited by
construction, not by an added smoothing pass.

x and y are solved independently (two decoupled 1D quintics sharing the
same time horizon) -- this is the standard simplification for this kind of
sampler (Frenet-frame planners do the same split along/across the
reference path); a coupled 2D solve would need a curvilinear reference
frame this project has no map/lane source to build (spec section 4)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class QuinticCoeffs:
    c: np.ndarray  # c[0..5]

    def position(self, t: float) -> float:
        return float(sum(self.c[i] * t ** i for i in range(6)))

    def velocity(self, t: float) -> float:
        return float(sum(i * self.c[i] * t ** (i - 1) for i in range(1, 6)))

    def acceleration(self, t: float) -> float:
        return float(sum(i * (i - 1) * self.c[i] * t ** (i - 2) for i in range(2, 6)))


def solve_quintic(p0: float, v0: float, a0: float, pT: float, vT: float, aT: float, T: float) -> QuinticCoeffs:
    """Closed-form solve for the 6 coefficients of x(t) = sum(c_i * t^i)
    satisfying x(0)=p0, x'(0)=v0, x''(0)=a0, x(T)=pT, x'(T)=vT, x''(T)=aT."""
    c0, c1, c2 = p0, v0, a0 / 2.0
    if T < 1e-6:
        return QuinticCoeffs(np.array([c0, c1, c2, 0.0, 0.0, 0.0]))
    A = np.array([
        [T ** 3, T ** 4, T ** 5],
        [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
        [6 * T, 12 * T ** 2, 20 * T ** 3],
    ])
    b = np.array([
        pT - (c0 + c1 * T + c2 * T ** 2),
        vT - (c1 + 2 * c2 * T),
        aT - 2 * c2,
    ])
    c3, c4, c5 = np.linalg.solve(A, b)
    return QuinticCoeffs(np.array([c0, c1, c2, c3, c4, c5]))


@dataclass
class PlannedWaypoint:
    x: float
    y: float
    v_target: float
    yaw: float
    t: float


def plan_quintic_path(start_xy: tuple[float, float], start_vel_xy: tuple[float, float], start_acc_xy: tuple[float, float],
                       target_xy: tuple[float, float], target_vel_xy: tuple[float, float], target_acc_xy: tuple[float, float],
                       horizon_s: float, n_points: int) -> list[PlannedWaypoint]:
    """x(t) and y(t) each get their own quintic; yaw and v_target at each
    sample come from the resulting (vx(t), vy(t)) -- so the reported yaw is
    always consistent with the actual planned heading of motion, not a
    separately-interpolated angle that could disagree with where the path
    is really pointing."""
    qx = solve_quintic(start_xy[0], start_vel_xy[0], start_acc_xy[0], target_xy[0], target_vel_xy[0], target_acc_xy[0], horizon_s)
    qy = solve_quintic(start_xy[1], start_vel_xy[1], start_acc_xy[1], target_xy[1], target_vel_xy[1], target_acc_xy[1], horizon_s)

    waypoints = []
    for i in range(n_points):
        t = horizon_s * i / (n_points - 1) if n_points > 1 else 0.0
        x, y = qx.position(t), qy.position(t)
        vx, vy = qx.velocity(t), qy.velocity(t)
        speed = math.hypot(vx, vy)
        yaw = math.atan2(vy, vx) if speed > 1e-3 else math.atan2(start_vel_xy[1], start_vel_xy[0])
        waypoints.append(PlannedWaypoint(x=x, y=y, v_target=speed, yaw=yaw, t=t))
    return waypoints
