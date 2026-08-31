"""
fusion.py
=========
This is the heart of M3: combining three separate "opinions" about the
same object into one clean description.

    CAMERA says:  "pedestrian, inside this pixel box"
    LIDAR  says:  "solid cluster of points at this 3D location"
    RADAR  says:  "something moving at this velocity, near this location"

Method used (deliberately simple -- NOT a deep-learning fusion network,
per the project's "don't overengineer" guidance):

    For each camera detection:
        1. Project every LiDAR cluster's center point onto the camera
           image (using transforms.project_point_to_image).
        2. If a cluster's projected point falls INSIDE the camera's
           bounding box -> that cluster is this object's 3D position.
        3. Find the radar detection whose (x, y) position is CLOSEST
           to the matched LiDAR cluster's center (within a distance
           threshold) -> that's this object's velocity.
        4. Merge: class+confidence (camera) + position (LiDAR) +
           velocity (radar) = one fused object.

SPECIAL CASE -- ROAD-SURFACE CLASSES (pothole, speed_bumps):
    lidar_processing.remove_ground() deletes any point near road
    height, because normally that IS just the boring road surface.
    But a pothole or speed bump genuinely lives at that same height
    range -- ground removal would silently delete the very points that
    describe it, so the normal clustered-matching path above would
    never find anything for these classes.

    Fix: for these two classes specifically, we skip the clustered
    LiDAR path entirely and instead directly project the RAW,
    non-ground-removed point cloud (still ROI-filtered, so it's not
    the whole 30,000-point scan) into the camera's bounding box, and
    take whichever of those points fall inside it. This is handled by
    match_raw_points_to_camera() below.
"""

import numpy as np
from transforms import project_point_to_image

# Classes that live AT road-surface height. Normal ground-removal
# would delete the LiDAR points that describe these, so they need the
# raw-point matching path (match_raw_points_to_camera) instead of the
# clustered path (match_lidar_to_camera).
# Classes that live AT road-surface height. Normal ground-removal
# would delete the LiDAR points that describe these, so they need the
# raw-point matching path (match_raw_points_to_camera) instead of the
# clustered path (match_lidar_to_camera).
#
# NOTE: these names MUST exactly match M2's classes.yaml output. M2's
# real taxonomy uses "speed_bumps" (plural) and "pothole" (singular) --
# update this set immediately if M2's class list ever changes.
ROAD_SURFACE_CLASSES = {"pothole", "speed_bumps"}

# Expected (min_z, max_z) height band for each road-surface class,
# used to reject implausible points before matching -- see
# match_raw_points_to_camera's docstring for why this matters.
ROAD_SURFACE_HEIGHT_RANGES = {
    "pothole": (-0.3, -0.01),      # a dip BELOW road level
    "speed_bumps": (0.01, 0.2),    # a slight rise above road level
}


def bbox_contains(bbox, point_uv):
    """True if pixel point (u, v) falls inside bbox = [xmin, ymin, xmax, ymax]."""
    if point_uv is None:
        return False
    u, v = point_uv
    xmin, ymin, xmax, ymax = bbox
    return xmin <= u <= xmax and ymin <= v <= ymax


def match_lidar_to_camera(camera_det, lidar_clusters, camera_intrinsics, camera_extrinsics):
    """
    Finds which LiDAR cluster (if any) belongs to this camera detection,
    by projecting each cluster's center into the image and checking if
    it lands inside the camera's bounding box.

    Returns: the matching cluster dict, or None if no cluster matched
    (e.g. LiDAR didn't get a good return on this object this frame).
    """
    best_cluster = None
    best_distance_to_center = float("inf")
    bbox = camera_det["bbox"]
    bbox_center_u = (bbox[0] + bbox[2]) / 2
    bbox_center_v = (bbox[1] + bbox[3]) / 2

    for cluster in lidar_clusters:
        uv = project_point_to_image(cluster["center"], camera_intrinsics, camera_extrinsics)
        if bbox_contains(bbox, uv):
            # if multiple clusters land inside the same box (rare, but
            # possible with clutter), pick the one closest to the
            # bbox's center pixel -- usually the most relevant object
            u, v = uv
            dist = (u - bbox_center_u) ** 2 + (v - bbox_center_v) ** 2
            if dist < best_distance_to_center:
                best_distance_to_center = dist
                best_cluster = cluster

    return best_cluster


def match_raw_points_to_camera(camera_det, raw_points, camera_intrinsics, camera_extrinsics,
                                 min_points=3, height_range=(-0.3, 0.3)):
    """
    Alternative to match_lidar_to_camera(), used ONLY for road-surface
    classes (pothole, speed_bumps). Instead of matching against
    pre-computed clusters (which ground removal already stripped these
    points out of), this projects each RAW point directly into the
    image and keeps whichever ones fall inside the camera's bbox.

    height_range: (min_z, max_z) -- only points in this band are even
        considered. WHY THIS MATTERS: a camera image is 2D -- a tall
        object (e.g. a pedestrian) can project to the SAME pixel
        region as a distant pothole/speed bump, purely by angular
        coincidence, even at a completely different real-world
        distance. Since potholes/speed bumps are physically flat (a
        few cm at most), we can reject implausible points outright.
        Use a NEGATIVE range for pothole (it's a dip below road level)
        and a small POSITIVE range for speed_bumps (it's a slight
        rise) -- this is tighter than one symmetric cutoff and
        correctly excludes even a pedestrian's lowest (near-feet)
        points, which a single abs()-based threshold could still
        occasionally let through. See ROAD_SURFACE_HEIGHT_RANGES below.

    Returns a cluster-shaped dict (same shape match_lidar_to_camera
    returns, so fuse_frame doesn't need to know the difference), or
    None if fewer than min_points fall inside the box (too few points
    to trust -- probably just noise, not a real detection).
    """
    if len(raw_points) == 0:
        return None

    bbox = camera_det["bbox"]
    min_z, max_z = height_range
    inside_points = []

    for p in raw_points:
        if not (min_z <= p[2] <= max_z):
            continue  # outside the plausible height band for this class
        uv = project_point_to_image(p, camera_intrinsics, camera_extrinsics)
        if bbox_contains(bbox, uv):
            inside_points.append(p)

    if len(inside_points) < min_points:
        return None

    inside_points = np.array(inside_points)
    center = inside_points.mean(axis=0)
    size = inside_points.max(axis=0) - inside_points.min(axis=0)
    return {"center": center.tolist(), "size": size.tolist(), "num_points": len(inside_points)}




def match_radar_to_position(position_xy, radar_detections, max_distance_m=2.5):
    """
    Finds the radar detection closest (in x,y) to a given position,
    within max_distance_m. Returns the matching radar detection's
    velocity, or [0.0, 0.0] if nothing matched closely enough (better
    to report "no velocity info" as zero than to guess wildly wrong).
    """
    best_det = None
    best_dist = max_distance_m
    px, py = position_xy[0], position_xy[1]

    for det in radar_detections:
        rx, ry = det["position"]
        dist = np.hypot(px - rx, py - ry)
        if dist < best_dist:
            best_dist = dist
            best_det = det

    if best_det is None:
        return [0.0, 0.0], False   # False = "no radar match found"
    return best_det["velocity"], True


def fuse_frame(camera_detections, lidar_clusters, radar_detections,
                camera_intrinsics, camera_extrinsics, raw_roi_points=None):
    """
    Runs the full fusion step for one frame. Combines camera + LiDAR +
    radar into a list of fused objects (not yet tracked with IDs --
    that happens in tracker.py).

    raw_roi_points: the ROI-filtered but NOT ground-removed point
        cloud (see lidar_processing.filter_roi -- call it without
        remove_ground first). Only needed if any camera detections in
        this frame are pothole/speed_bumps; pass None or [] otherwise
        and normal classes are unaffected.

    Returns: list of dicts:
        {
            "class": ..., "confidence": ...,
            "position": [x, y, z],
            "velocity": [vx, vy],
            "size": [l, w, h],
            "has_lidar": bool, "has_radar": bool,
            "timestamp": ...
        }
    """
    fused_objects = []
    raw_roi_points = raw_roi_points if raw_roi_points is not None else []

    for cam_det in camera_detections:
        if cam_det["class"] in ROAD_SURFACE_CLASSES:
            # ground-level classes: use raw (non-ground-removed) points,
            # restricted to a class-specific plausible height band
            height_range = ROAD_SURFACE_HEIGHT_RANGES.get(cam_det["class"], (-0.3, 0.3))
            cluster = match_raw_points_to_camera(cam_det, raw_roi_points,
                                                  camera_intrinsics, camera_extrinsics,
                                                  height_range=height_range)
        else:
            # everything else: normal clustered-object matching
            cluster = match_lidar_to_camera(cam_det, lidar_clusters, camera_intrinsics, camera_extrinsics)

        if cluster is None:
            # No LiDAR confirmation this frame -- we still keep the
            # detection (camera-only), but flag it as lower confidence
            # since we don't have a real 3D position for it.
            fused_objects.append({
                "class": cam_det["class"],
                "confidence": cam_det["confidence"] * 0.5,  # penalize camera-only
                "position": None,
                "velocity": [0.0, 0.0],
                "size": None,
                "has_lidar": False,
                "has_radar": False,
                "timestamp": cam_det["timestamp"],
            })
            continue

        velocity, has_radar = match_radar_to_position(cluster["center"], radar_detections)

        fused_objects.append({
            "class": cam_det["class"],
            "confidence": cam_det["confidence"],
            "position": cluster["center"],
            "velocity": velocity,
            "size": cluster["size"],
            "has_lidar": True,
            "has_radar": has_radar,
            "timestamp": cam_det["timestamp"],
        })

    return fused_objects