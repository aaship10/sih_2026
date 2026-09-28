"""Coordinate transforms shared by all M3 modules.

Frame conventions
------------------
- CARLA world frame: left-handed, Z-up, meters. This is the frame reported by
  `ego_position` / `ego_velocity` in the FramePacket M3 receives from M2.
- Ego-vehicle frame: origin at the ego actor's location, X-forward, Y-right,
  Z-up, axes aligned with the vehicle body (no roll/pitch assumed -- CARLA
  vehicles are treated as flat-riding here, consistent with scenario3.py,
  which never reads back vehicle pitch/roll). The ego actor's own origin is
  assumed to sit at ground level (config.GROUND_PLANE_EGO_Z).
- Sensor-local frame: X-forward, Y-right, Z-up, relative to the sensor's own
  mount transform (see config.py), which for LiDAR/radar in scenario3.py is a
  pure translation (no rotation) and for the camera includes a fixed pitch.
- Camera optical frame: X-right, Y-down, Z-forward (OpenCV/pinhole
  convention), used only for image-plane projection.

All rotation formulas below match CARLA's own `carla.Transform` definitions
(Transform.get_forward_vector / get_right_vector), so a rotation built here
from an actor's yaw/pitch/roll agrees with what the CARLA server itself would
report for the same angles.
"""
from __future__ import annotations

import math

import numpy as np

import config


def rotation_matrix(pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    """3x3 rotation matrix mapping a vector in the rotated local frame to its
    parent frame, matching CARLA's Transform rotation convention exactly
    (columns are the local frame's forward/right/up axes, expressed in the
    parent frame)."""
    pitch, yaw, roll = (math.radians(a) for a in (pitch_deg, yaw_deg, roll_deg))
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)

    forward = np.array([cp * cy, cp * sy, sp])
    right = np.array([
        cy * sr * sp - sy * cr,
        sy * sr * sp + cy * cr,
        -sr * cp,
    ])
    up = np.cross(forward, right)
    return np.column_stack([forward, right, up])


def local_to_parent(points_local: np.ndarray, transform: "config.Transform") -> np.ndarray:
    """Transform (N,3) points from a sensor/child frame into its parent frame."""
    if points_local.shape[0] == 0:
        return points_local
    R = rotation_matrix(transform.pitch_deg, transform.yaw_deg, transform.roll_deg)
    t = np.array([transform.x, transform.y, transform.z])
    return points_local @ R.T + t


def parent_to_local(points_parent: np.ndarray, transform: "config.Transform") -> np.ndarray:
    """Inverse of local_to_parent: parent-frame points into the child frame."""
    if points_parent.shape[0] == 0:
        return points_parent
    R = rotation_matrix(transform.pitch_deg, transform.yaw_deg, transform.roll_deg)
    t = np.array([transform.x, transform.y, transform.z])
    return (points_parent - t) @ R


def ego_to_world(points_ego: np.ndarray, ego_position: np.ndarray, ego_yaw_deg: float) -> np.ndarray:
    """Transform (N,3) ego-frame points into the CARLA world frame."""
    if points_ego.shape[0] == 0:
        return points_ego
    yaw = math.radians(ego_yaw_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    R = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return points_ego @ R.T + ego_position


def world_to_ego(points_world: np.ndarray, ego_position: np.ndarray, ego_yaw_deg: float) -> np.ndarray:
    """Transform (N,3) world-frame points into the ego-vehicle frame."""
    if points_world.shape[0] == 0:
        return points_world
    yaw = math.radians(ego_yaw_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    R = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return (points_world - ego_position) @ R


def vehicle_to_camera_optical(points_ego: np.ndarray, camera_transform: "config.Transform") -> np.ndarray:
    """Project ego-frame points into the camera's optical frame (X-right,
    Y-down, Z-forward), ready for pinhole projection."""
    local = parent_to_local(points_ego, camera_transform)  # camera sensor-local (X-fwd,Y-right,Z-up)
    if local.shape[0] == 0:
        return local
    return np.column_stack([local[:, 1], -local[:, 2], local[:, 0]])


def camera_pixel_to_ray_ego(u: float, v: float, camera_transform: "config.Transform") -> tuple[np.ndarray, np.ndarray]:
    """Return (origin_ego, direction_ego) for the camera ray through pixel (u, v)."""
    x_opt = (u - config.CAMERA_CX) / config.CAMERA_FX
    y_opt = (v - config.CAMERA_CY) / config.CAMERA_FY
    # optical (x_opt, y_opt, 1) -> sensor-local (forward, right, up) = (z_opt, x_opt, -y_opt)
    dir_local = np.array([[1.0, x_opt, -y_opt]])
    origin_local = np.zeros((1, 3))
    origin_ego = local_to_parent(origin_local, camera_transform)[0]
    dir_ego = local_to_parent(dir_local, camera_transform)[0] - origin_ego
    return origin_ego, dir_ego
