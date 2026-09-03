"""
lidar_processing.py
====================
Turns a raw LiDAR point cloud (thousands of individual (x,y,z) dots --
one dot per laser beam that hit something) into a short list of
"clusters" -- one entry per real object nearby.

Pipeline:
    raw points
        -> remove ground points (the road surface)
        -> keep only points within a region of interest (ROI)
        -> group remaining points into clusters (DBSCAN)
        -> summarize each cluster as one 3D box (center + size)
"""

import numpy as np
from sklearn.cluster import DBSCAN


def remove_ground(points: np.ndarray, ground_z_threshold: float = 0.15) -> np.ndarray:
    """
    Deletes points that are basically part of the flat road surface.

    Simple approach (good enough for a student project): any point
    whose height (z) is below `ground_z_threshold` meters is treated as
    ground and removed. This works because our ego frame's z=0 is
    roughly road height, and the road is close to flat locally.

    (A more advanced approach -- fitting a plane with RANSAC -- handles
    slightly sloped roads better, but constant-height filtering is a
    reasonable, fast starting point.)
    """
    if len(points) == 0:
        return points
    mask = points[:, 2] > ground_z_threshold
    return points[mask]


def filter_roi(points: np.ndarray, max_forward: float = 50.0,
                max_sideways: float = 15.0, max_height: float = 3.0) -> np.ndarray:
    """
    Keeps only points inside a "region of interest" box around the car:
    up to `max_forward` meters ahead, `max_sideways` meters to either
    side, and below `max_height` meters tall. Anything outside this box
    is too far away to matter for immediate driving decisions, so we
    drop it to keep later steps fast and clean.
    """
    if len(points) == 0:
        return points
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    mask = (
        (x > 0) & (x < max_forward) &
        (np.abs(y) < max_sideways) &
        (z < max_height)
    )
    return points[mask]


def cluster_points(points: np.ndarray, eps: float = 0.5, min_samples: int = 5):
    """
    Groups nearby points into clusters using DBSCAN: any points within
    `eps` meters of each other are considered part of the same object,
    as long as a cluster has at least `min_samples` points (this
    filters out lone noisy points that aren't a real object).

    Returns: list of clusters, each cluster is a dict:
        {
            "center": [x, y, z],       # centroid of the cluster
            "size": [length, width, height],
            "num_points": int,
            "points": (M, 3) array     # kept for debugging/visualization
        }
    """
    if len(points) < min_samples:
        return []

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)

    clusters = []
    for label in set(labels):
        if label == -1:
            continue  # -1 = noise points DBSCAN couldn't group; discard
        cluster_pts = points[labels == label]
        center = cluster_pts.mean(axis=0)
        size = cluster_pts.max(axis=0) - cluster_pts.min(axis=0)
        clusters.append({
            "center": center.tolist(),
            "size": size.tolist(),
            "num_points": len(cluster_pts),
            "points": cluster_pts,
        })
    return clusters


def get_raw_roi_points(raw_points: np.ndarray, max_forward: float = 50.0,
                        max_sideways: float = 15.0, max_height: float = 3.0) -> np.ndarray:
    """
    Like process_lidar_frame(), but WITHOUT ground removal -- keeps
    points near road height. Needed for matching road-surface classes
    (pothole, speed_bump) in fusion.py, since remove_ground() would
    otherwise delete exactly the points that describe them.
    """
    return filter_roi(raw_points, max_forward, max_sideways, max_height)


def process_lidar_frame(raw_points: np.ndarray,
                         ground_z_threshold: float = 0.15,
                         max_forward: float = 50.0,
                         max_sideways: float = 15.0,
                         cluster_eps: float = 0.5,
                         cluster_min_samples: int = 5):
    """
    Convenience wrapper: runs the full LiDAR pipeline in one call.
    This is the function main.py actually calls each frame.
    """
    pts = remove_ground(raw_points, ground_z_threshold)
    pts = filter_roi(pts, max_forward, max_sideways)
    clusters = cluster_points(pts, cluster_eps, cluster_min_samples)
    return clusters