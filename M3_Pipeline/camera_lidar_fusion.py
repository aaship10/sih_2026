"""Camera <-> LiDAR association.

Method (M3 mentor notes, section 12): project each LiDAR cluster's points
into the camera image using the known camera intrinsics/extrinsics, take the
2D bounding box of the points that land inside the image, and match that
projected box against M2's camera detection boxes with IoU, solved as a
linear assignment problem. This is the "projection + IoU" approach -- no
learned cross-modal matching, deliberately the simplest reliable option.

`ground_plane_backproject` is the fallback for a camera detection that finds
no LiDAR support at all (e.g. thin/fast-appearing object with sparse
returns): it intersects the ray through the bbox's bottom-center pixel with
the assumed ground plane (config.GROUND_PLANE_EGO_Z) to get a rough 3D fix.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

import config
from coordinate_transforms import camera_pixel_to_ray_ego, vehicle_to_camera_optical


def project_to_image(points_ego: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project ego-frame points to pixel coordinates.

    Returns (pixels, valid_mask); pixels are only meaningful where valid.
    """
    if points_ego.shape[0] == 0:
        return np.zeros((0, 2)), np.zeros((0,), dtype=bool)
    optical = vehicle_to_camera_optical(points_ego, config.CAMERA_TRANSFORM)
    z = optical[:, 2]
    valid = z > 0.3
    u = np.full(z.shape, np.nan)
    v = np.full(z.shape, np.nan)
    u[valid] = config.CAMERA_FX * optical[valid, 0] / z[valid] + config.CAMERA_CX
    v[valid] = config.CAMERA_FY * optical[valid, 1] / z[valid] + config.CAMERA_CY
    in_bounds = valid & (u >= 0) & (u < config.CAMERA_WIDTH) & (v >= 0) & (v < config.CAMERA_HEIGHT)
    return np.column_stack([u, v]), in_bounds


def cluster_projected_bbox(cluster_points_ego: np.ndarray) -> np.ndarray | None:
    pixels, valid = project_to_image(cluster_points_ego)
    if valid.sum() < 3:
        return None
    pts = pixels[valid]
    return np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1, ya1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xa2, ya2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def associate(camera_detections: list[dict], lidar_clusters: list[dict]) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Return (matches, unmatched_camera_idx, unmatched_lidar_idx).

    matches: list of (camera_idx, lidar_idx) pairs.
    """
    if not camera_detections or not lidar_clusters:
        return [], list(range(len(camera_detections))), list(range(len(lidar_clusters)))

    projected = [cluster_projected_bbox(c["points_ego"]) for c in lidar_clusters]
    cost = np.ones((len(camera_detections), len(lidar_clusters)))
    for i, det in enumerate(camera_detections):
        for j, proj_box in enumerate(projected):
            if proj_box is None:
                continue
            iou = _iou(np.array(det["box"]), proj_box)
            cost[i, j] = 1.0 - iou

    row_idx, col_idx = linear_sum_assignment(cost)
    matches, used_cam, used_lidar = [], set(), set()
    for r, c in zip(row_idx, col_idx):
        if cost[r, c] <= 1.0 - config.CAM_LIDAR_MIN_IOU:
            matches.append((int(r), int(c)))
            used_cam.add(int(r))
            used_lidar.add(int(c))
    unmatched_cam = [i for i in range(len(camera_detections)) if i not in used_cam]
    unmatched_lidar = [j for j in range(len(lidar_clusters)) if j not in used_lidar]
    return matches, unmatched_cam, unmatched_lidar


def ground_plane_backproject(box: list[float]) -> np.ndarray | None:
    """Approximate an ego-frame 3D position for a camera-only detection by
    intersecting the ray through the bbox bottom-center with the assumed
    ground plane. Used only when no LiDAR cluster (and no radar target,
    see camera_radar_fusion.py) could be associated."""
    u = (box[0] + box[2]) / 2.0
    v = box[3]  # bottom edge of the box ~= where the object meets the ground
    origin_ego, direction_ego = camera_pixel_to_ray_ego(u, v, config.CAMERA_TRANSFORM)
    if abs(direction_ego[2]) < 1e-4:
        return None
    t = (config.GROUND_PLANE_EGO_Z - origin_ego[2]) / direction_ego[2]
    if t <= 0:
        return None
    return origin_ego + t * direction_ego
