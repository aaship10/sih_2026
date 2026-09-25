"""Dummy camera/LiDAR/radar generators so M3 can be developed and unit
tested before CARLA/M2 data is available (M3 mentor notes, section 19).
NOT used anywhere in the live pipeline -- m3_server.py only ever consumes
real CARLA/M2 data; this module exists solely for offline development and
tests/test_pipeline_dummy.py.
"""
from __future__ import annotations

import numpy as np

import camera_lidar_fusion
import config

# Canonical ego-frame reference object dummy_camera_detection()'s default
# box represents -- matches the (x~15, y=0, z=0.8) center most callers in
# tests/test_pipeline_dummy.py also pass to dummy_lidar_bytes(), so the
# camera detection and LiDAR cluster correspond to the same real object by
# default (some tests deliberately pass a different, off-axis LiDAR center
# instead, to test a camera detection that has NO matching LiDAR support --
# that mismatch is intentional there, unrelated to this constant).
_DEFAULT_BOX_CENTER_EGO = (15.0, 0.0, 0.8)
_DEFAULT_BOX_HALF_EXTENT = (2.0, 0.9, 0.75)  # roughly car-sized


def _default_box() -> list[float]:
    """Computed via M3's OWN camera_lidar_fusion.project_to_image from the
    canonical reference object above, rather than a hardcoded pixel tuple --
    a hardcoded box silently goes spatially inconsistent with
    dummy_lidar_bytes's own object_center_ego the moment config.py's camera
    intrinsics (CAMERA_FOV_DEG, mount transform, ...) ever change; this
    can't drift out of sync since it's derived from the same intrinsics the
    production code under test actually uses."""
    cx, cy, cz = _DEFAULT_BOX_CENTER_EGO
    hx, hy, hz = _DEFAULT_BOX_HALF_EXTENT
    corners = np.array([
        [cx + sx * hx, cy + sy * hy, cz + sz * hz]
        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
    ])
    pixels, valid = camera_lidar_fusion.project_to_image(corners)
    pixels = pixels[valid]
    return [float(pixels[:, 0].min()), float(pixels[:, 1].min()), float(pixels[:, 0].max()), float(pixels[:, 1].max())]


def dummy_camera_detection(class_name="car", box=None, confidence=0.9, class_id=2):
    return {"class_id": class_id, "class_name": class_name, "confidence": confidence,
            "box": list(box) if box is not None else _default_box()}


def dummy_lidar_bytes(object_center_ego=(15.0, 0.0, 0.8), num_object_points=200, num_ground_points=2000, seed=0):
    """Synthetic LiDAR scan: a point cluster at `object_center_ego` (ego
    frame) plus a ground plane, encoded exactly like CARLA's ray_cast raw_data
    (float32, [x,y,z,intensity] per point, LiDAR-local frame)."""
    rng = np.random.default_rng(seed)
    mount = np.array([config.LIDAR_TRANSFORM.x, config.LIDAR_TRANSFORM.y, config.LIDAR_TRANSFORM.z])
    center_lidar = np.array(object_center_ego) - mount
    obj_pts = center_lidar + rng.normal(scale=0.3, size=(num_object_points, 3))

    ground_xy = rng.uniform(-20, 20, size=(num_ground_points, 2))
    ground_z_ego = config.GROUND_PLANE_EGO_Z + rng.normal(scale=0.03, size=(num_ground_points, 1))
    ground_z_lidar = ground_z_ego - mount[2]
    ground_pts = np.hstack([ground_xy, ground_z_lidar])

    points = np.vstack([obj_pts, ground_pts]).astype(np.float32)
    intensity = rng.uniform(0.2, 0.9, size=(points.shape[0], 1)).astype(np.float32)
    return np.hstack([points, intensity]).astype(np.float32).tobytes()


def dummy_radar_bytes(depth=15.0, azimuth_rad=0.0, altitude_rad=0.0, velocity_mps=-3.0):
    """Synthetic radar frame: one detection, encoded like CARLA's
    RadarMeasurement raw_data (float32 [velocity,azimuth,altitude,depth])."""
    detection = np.array([[velocity_mps, azimuth_rad, altitude_rad, depth]], dtype=np.float32)
    return detection.tobytes()
