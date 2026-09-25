"""Direct camera <-> radar association (fallback path).

Primary radar association happens in sensor_fusion.py via proximity to a
LiDAR-derived 3D position (see that module's docstring): LiDAR gives accurate
geometry, so once an object's position is known, the closest radar target in
ego-frame Euclidean distance is almost always the right match, and needs no
re-projection.

This module covers the fallback case: a camera detection has no supporting
LiDAR cluster (e.g. a thin/fast-appearing object at range with sparse
returns), but a radar target's projected image position still falls inside
the camera's bounding box. Radar's angular resolution is coarse compared to
LiDAR/camera, so this is used only as a secondary signal, never the primary
association method.
"""
from __future__ import annotations

import numpy as np

from camera_lidar_fusion import project_to_image


def associate_unmatched(camera_detections: list[dict], camera_idxs: list[int], radar_targets: list[dict]) -> dict[int, int]:
    """Return {camera_idx: radar_target_idx} for camera detections (indices
    restricted to `camera_idxs`) whose bbox contains a projected radar
    target; picks the closest-to-center target when more than one lands
    inside the same box."""
    if not camera_idxs or not radar_targets:
        return {}
    positions = np.array([t["position_ego"] for t in radar_targets])
    pixels, valid = project_to_image(positions)
    result: dict[int, int] = {}
    for cam_idx in camera_idxs:
        box = camera_detections[cam_idx]["box"]
        best_j, best_dist = None, float("inf")
        for j in range(len(radar_targets)):
            if not valid[j]:
                continue
            u, v = pixels[j]
            if box[0] <= u <= box[2] and box[1] <= v <= box[3]:
                cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
                dist = (u - cx) ** 2 + (v - cy) ** 2
                if dist < best_dist:
                    best_dist, best_j = dist, j
        if best_j is not None:
            result[cam_idx] = best_j
    return result
