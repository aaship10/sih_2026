"""Radar processing.

CARLA's `sensor.other.radar` raw_data is a flat float32 buffer, 4 values per
detection, in this fixed order: (velocity, azimuth, altitude, depth). This
matches CARLA's own RadarDetection layout (see PythonAPI examples such as
manual_control.py's radar callback, which projects each detection using
pitch=altitude, yaw=azimuth, depth=range from the sensor origin -- the same
formula `spherical_to_cartesian` below implements). scenario3.py listens with
`self.radar.listen(lambda x: self.radar_buf.put(x.frame, bytes(x.raw_data)))`.

Units: velocity in m/s (sign convention documented in config.RADAR_VELOCITY_SIGN
-- CARLA's own docs describe it as "towards the sensor", i.e. positive =
approaching; this should be spot-checked against a known closing/opening
scenario once live data is available), azimuth/altitude in radians, depth in
meters.
"""
from __future__ import annotations

import numpy as np

import config
from coordinate_transforms import local_to_parent


def decode_radar(raw_bytes: bytes) -> np.ndarray:
    """Return an (N,4) array of [velocity, azimuth, altitude, depth]."""
    if not raw_bytes:
        return np.zeros((0, 4), dtype=np.float32)
    values = np.frombuffer(raw_bytes, dtype=np.float32)
    usable = (values.size // 4) * 4
    return values[:usable].reshape(-1, 4)


def spherical_to_cartesian(detections: np.ndarray) -> np.ndarray:
    """[velocity,azimuth,altitude,depth] -> [x,y,z] in radar-local frame."""
    if detections.shape[0] == 0:
        return np.zeros((0, 3))
    azimuth, altitude, depth = detections[:, 1], detections[:, 2], detections[:, 3]
    x = depth * np.cos(altitude) * np.cos(azimuth)
    y = depth * np.cos(altitude) * np.sin(azimuth)
    z = depth * np.sin(altitude)
    return np.column_stack([x, y, z])


def process(raw_bytes: bytes) -> list[dict]:
    """Full radar pipeline: raw bytes -> targets with position + range-rate
    in the ego frame."""
    detections = decode_radar(raw_bytes)
    if detections.shape[0] == 0:
        return []
    xyz_local = spherical_to_cartesian(detections)
    xyz_ego = local_to_parent(xyz_local, config.RADAR_TRANSFORM)
    targets = []
    for i in range(detections.shape[0]):
        depth = float(detections[i, 3])
        if depth <= 0.0 or depth > config.RADAR_MAX_RANGE_M:
            continue
        targets.append({
            "position_ego": xyz_ego[i],
            "range_rate_mps": config.RADAR_VELOCITY_SIGN * float(detections[i, 0]),
            "azimuth_rad": float(detections[i, 1]),
            "altitude_rad": float(detections[i, 2]),
            "depth_m": depth,
        })
    return targets
