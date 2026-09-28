"""LiDAR point-cloud processing.

CARLA's `sensor.lidar.ray_cast` raw_data is a flat float32 buffer, 4 values
per point: (x, y, z, intensity), in the LiDAR sensor's own local frame
(X-forward, Y-right, Z-up). See scenario3.py `_setup_sensors`, which listens
with `self.lidar.listen(lambda x: self.lidar_buf.put(x.frame, bytes(x.raw_data)))`.

Pipeline (Python, per M3 mentor notes section 9):
    decode -> transform to ego frame -> ROI filter -> ground removal (RANSAC
    plane) -> Euclidean clustering -> per-cluster 3D obstacle summary.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

import config
from coordinate_transforms import local_to_parent


def decode_lidar(raw_bytes: bytes) -> np.ndarray:
    """Return an (N,4) array of [x,y,z,intensity] in LiDAR-local frame."""
    if not raw_bytes:
        return np.zeros((0, 4), dtype=np.float32)
    points = np.frombuffer(raw_bytes, dtype=np.float32)
    usable = (points.size // 4) * 4
    return points[:usable].reshape(-1, 4)


def to_ego_frame(points_lidar_xyz: np.ndarray) -> np.ndarray:
    """Translate LiDAR-local points into the ego-vehicle frame."""
    return local_to_parent(points_lidar_xyz, config.LIDAR_TRANSFORM)


def apply_roi(points_ego: np.ndarray) -> np.ndarray:
    if points_ego.shape[0] == 0:
        return points_ego
    x, y, z = points_ego[:, 0], points_ego[:, 1], points_ego[:, 2]
    mask = (
        (x >= config.LIDAR_ROI_X_MIN) & (x <= config.LIDAR_ROI_X_MAX)
        & (y >= config.LIDAR_ROI_Y_MIN) & (y <= config.LIDAR_ROI_Y_MAX)
        & (z >= config.LIDAR_ROI_Z_MIN) & (z <= config.LIDAR_ROI_Z_MAX)
    )
    return points_ego[mask]


def remove_ground(points_ego: np.ndarray, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """RANSAC-fit a (roughly horizontal) ground plane and split points into
    (non_ground, ground). Falls back to a flat height threshold around the
    assumed ground plane if too few points remain to fit reliably (e.g. a
    near-empty scan)."""
    n = points_ego.shape[0]
    if n < 50:
        return points_ego, np.zeros((0, 3))

    rng = rng or np.random.default_rng(0)
    best_inliers = None
    best_count = -1
    xyz = points_ego[:, :3]
    for _ in range(config.LIDAR_GROUND_RANSAC_ITERATIONS):
        sample_idx = rng.choice(n, size=3, replace=False)
        p1, p2, p3 = xyz[sample_idx]
        normal = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm
        if abs(normal[2]) < 0.85:  # reject near-vertical planes (walls, vehicle sides)
            continue
        d = -normal.dot(p1)
        dist = np.abs(xyz.dot(normal) + d)
        inliers = dist < config.LIDAR_GROUND_DIST_THRESHOLD_M
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers

    if best_inliers is None or best_count < config.LIDAR_GROUND_MIN_INLIER_RATIO * n:
        ground_z = config.GROUND_PLANE_EGO_Z
        mask = xyz[:, 2] < ground_z + config.LIDAR_GROUND_DIST_THRESHOLD_M
        return points_ego[~mask], points_ego[mask]

    return points_ego[~best_inliers], points_ego[best_inliers]


def cluster_points(points_ego: np.ndarray) -> list[dict]:
    """Radius-connectivity clustering (a dependency-light stand-in for
    DBSCAN, using the same core idea: connect points within
    `LIDAR_CLUSTER_EPS_M`, then drop components smaller than
    `LIDAR_CLUSTER_MIN_POINTS` as noise)."""
    n = points_ego.shape[0]
    if n == 0:
        return []
    xyz = points_ego[:, :3]
    tree = cKDTree(xyz)
    pairs = tree.query_pairs(r=config.LIDAR_CLUSTER_EPS_M, output_type="ndarray")

    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i, j in pairs:
        union(int(i), int(j))

    roots = np.array([find(i) for i in range(n)])
    clusters = []
    for root in np.unique(roots):
        idx = np.where(roots == root)[0]
        if idx.size < config.LIDAR_CLUSTER_MIN_POINTS:
            continue
        pts = xyz[idx]
        centroid = pts.mean(axis=0)
        mins, maxs = pts.min(axis=0), pts.max(axis=0)
        size = np.maximum(maxs - mins, 0.15)  # floor so degenerate clusters still have volume
        clusters.append({
            "centroid_ego": centroid,
            "size": size,  # [length(x), width(y), height(z)]
            "num_points": int(idx.size),
            "points_ego": pts,
        })
    return clusters


def process(raw_bytes: bytes) -> list[dict]:
    """Full LiDAR pipeline: raw bytes -> obstacle clusters in ego frame."""
    raw = decode_lidar(raw_bytes)
    ego_pts = to_ego_frame(raw[:, :3])
    roi_pts = apply_roi(ego_pts)
    non_ground, _ground = remove_ground(roi_pts)
    return cluster_points(non_ground)
