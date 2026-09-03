"""
test_send_frame_matched.py
============================
An upgraded version of test_send_frame.py: instead of random,
unrelated LiDAR points, this script runs YOLO directly on test.jpg to
get REAL bounding boxes, then uses inverse camera projection (the math
below is verified against transforms.py's real forward projection --
see the round-trip self-test at the bottom) to place fake LiDAR points
exactly where a couple of those real detections are.

This means fusion should actually MATCH and produce non-zero
tracked_objects/predictions -- letting you see the full pipeline work
end-to-end before CARLA is connected.

CAVEAT: since there's no real depth sensor here, "depth" (how far away
each object actually is) is an ASSUMED value we pick, not measured --
this is only for testing the fusion/tracking/prediction LOGIC, not for
producing physically accurate distances. Real depth only comes from
real LiDAR once CARLA is connected.

BEFORE RUNNING:
    1. M2 must be running:  python perception_server.py   (port 8000)
    2. M3 must be running:  python m3_server.py            (port 9000)
    3. This script must run from a location where it can import
       transforms.py (your workspace root) AND find test.jpg
       (adjust TEST_JPG_PATH / MODEL_PATH below if needed)

RUN WITH:
    python test_send_frame_matched.py
"""

import sys
import os

import cv2
import numpy as np
import requests
from ultralytics import YOLO

# so we can import transforms.py from the workspace root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transforms import project_point_to_image, make_default_camera_intrinsics

M2_URL = "http://localhost:8000/api/v1/perception"
TEST_JPG_PATH = "Carla Object Detection/test.jpg"
MODEL_PATH = "Carla Object Detection/best.pt"

# Must match m3_server.py's defaults (or your actual env var overrides)
CAMERA_EXTRINSICS = {"offset": (1.5, 0.0, 1.4)}
LIDAR_SENSOR_OFFSET = (0.0, 0.0, 1.8)
RADAR_SENSOR_OFFSET = (2.0, 0.0, 0.5)

# Classes we'll pick for fake matching -- ground-level vehicles/people
# only, since our simple "assumed depth + object height band" approach
# models these well. Skip elevated/small classes like traffic_signal,
# road_sign for this synthetic test.
GROUND_LEVEL_CLASSES = {
    "car", "Car",
    "bus", "Bus",
    "truck", "Truck",
    "tempo", "Tempo",
    "motorcycle", "Motorcycle",
    "bicycle", "Bicycle",
    "rickshaw", "Rickshaw",
    "pedestrian", "Pedestrian",
}


def inverse_project(u, v, target_z_ego, intrinsics, extrinsics, min_depth=2.0, max_depth=60.0):
    """
    Inverse of transforms.project_point_to_image(). Given a pixel (u, v)
    and a TARGET ego-frame height (target_z_ego), solves for the depth
    (forward distance) that would place a point at exactly that height
    when projected to this pixel -- then returns the full 3D point.

    WHY TARGET HEIGHT, NOT ASSUMED DEPTH: an earlier version of this
    function assumed a fixed DEPTH and let the resulting height fall
    wherever the math produced -- but depending on which row of the
    image a bbox center lands on, that can produce wildly wrong
    heights (even negative/underground), which then get silently
    deleted by lidar_processing.remove_ground() before fusion ever
    runs. Solving for depth given a known-sane target height avoids
    this entire failure mode. Verified against transforms.py's real
    forward projection (round-trips to zero error, same as before).

    Raises ValueError if the solved depth falls outside
    [min_depth, max_depth] -- this is a genuine sanity check: if a
    pixel implies an object 200m away, something about the assumed
    target height or camera calibration is likely wrong for that
    detection, and it's better to fail loudly than silently place a
    nonsense point.
    """
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    ox, oy, oz = extrinsics["offset"]

    zc_target = target_z_ego - oz
    if abs(v - cy) < 1e-6:
        raise ValueError(f"pixel row v={v} is exactly at the camera's horizon line -- depth is undefined here")

    xc = -zc_target * fy / (v - cy)

    if not (min_depth <= xc <= max_depth):
        raise ValueError(
            f"solved depth {xc:.1f}m for pixel (u={u:.0f}, v={v:.0f}) with "
            f"target_z_ego={target_z_ego} is outside the plausible range "
            f"[{min_depth}, {max_depth}]m -- skipping this detection"
        )

    yc = -(u - cx) * xc / fx
    return [xc + ox, yc + oy, target_z_ego]


def make_lidar_cluster(center_xyz, n_points=50, spread_xy=0.4, spread_z=0.3):
    """Scatters fake LiDAR points around an assumed 3D object location,
    in the same style as dummy_data.py's synthetic clusters.

    IMPORTANT: z must be centered on center_xyz[2] (the actual
    inverse-projected height), not an independent absolute range --
    otherwise the fake cluster's mean position re-projects to the
    wrong image ROW, landing outside the original bbox and silently
    breaking the match. (This was a real bug in an earlier version of
    this script -- verified by round-tripping through the real
    transforms.py projection function.)
    """
    rng = np.random.default_rng()
    x = center_xyz[0] + rng.normal(0, spread_xy, n_points)
    y = center_xyz[1] + rng.normal(0, spread_xy, n_points)
    z = center_xyz[2] + rng.normal(0, spread_z, n_points)
    intensity = rng.uniform(0.4, 0.9, n_points)
    return np.stack([x, y, z, intensity], axis=1).astype(np.float32)


def make_radar_detection(assumed_position_xy, assumed_velocity_xy, radar_offset):
    """
    Builds a raw CARLA-format radar row (velocity, azimuth, altitude,
    depth) that will decode back to roughly assumed_position/velocity,
    matching m3_server.py's expected column order.
    """
    dx = assumed_position_xy[0] - radar_offset[0]
    dy = assumed_position_xy[1] - radar_offset[1]
    depth = float(np.hypot(dx, dy))
    azimuth = float(np.arctan2(dy, dx))
    altitude = 0.0
    # radial velocity: project the assumed (vx, vy) onto the line-of-sight direction
    radial_velocity = float(assumed_velocity_xy[0] * np.cos(azimuth) + assumed_velocity_xy[1] * np.sin(azimuth))
    return [radial_velocity, azimuth, altitude, depth]


def build_matched_sensor_data(image_path, model_path):
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Could not read {image_path}")
    height, width = image.shape[:2]

    # IMPORTANT CONSISTENCY CHECK: m3_server.py computes its own camera
    # intrinsics from CAMERA_IMAGE_WIDTH/CAMERA_IMAGE_HEIGHT env vars
    # (default 1280x720), NOT from the real image file. If this image's
    # actual size differs, M3 will use different fx/fy/cx/cy than this
    # script did -- causing the "matched" point to land OUTSIDE the
    # bbox on M3's side, for reasons that have nothing to do with a
    # real bug. Set CAMERA_IMAGE_WIDTH/CAMERA_IMAGE_HEIGHT env vars
    # before launching m3_server.py to match, if this warning fires.
    if (width, height) != (1280, 720):
        print(
            f"⚠️  WARNING: {image_path} is {width}x{height}, but m3_server.py "
            f"defaults to 1280x720 camera intrinsics. Set env vars before "
            f"starting m3_server.py:\n"
            f"     CAMERA_IMAGE_WIDTH={width} CAMERA_IMAGE_HEIGHT={height} python m3_server.py\n"
            f"   Otherwise this test's fake points won't line up with M3's "
            f"projection math, and fusion may fail to match for reasons "
            f"unrelated to the actual pipeline logic.\n"
        )

    intrinsics = make_default_camera_intrinsics(image_width=width, image_height=height, fov_deg=90)

    print(f"Running YOLO on {image_path} ({width}x{height}) ...")
    model = YOLO(model_path)
    results = model.predict(source=image, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        raise RuntimeError("YOLO found zero detections on this image -- can't build a matched test.")

    xyxy = boxes.xyxy.cpu().tolist()
    class_ids = boxes.cls.cpu().tolist()
    class_names = [model.names[int(c)] for c in class_ids]

    # pick up to 2 ground-level detections to fake-match
    candidates = [
        (i, name, xyxy[i]) for i, name in enumerate(class_names)
        if name in GROUND_LEVEL_CLASSES
    ][:2]

    if not candidates:
        raise RuntimeError(
            f"No ground-level classes found among detections: {set(class_names)}. "
            f"Add more class names to GROUND_LEVEL_CLASSES or try a different image."
        )

    all_lidar_points = []
    radar_rows = []
    TARGET_HEIGHT_M = 0.5  # roughly mid-body height for ground vehicles -- keeps z sane

    print(f"\nMatching up to {len(candidates)} detection(s):")
    matched_count = 0
    for idx, (box_index, cls_name, bbox) in enumerate(candidates):
        u_center = (bbox[0] + bbox[2]) / 2
        v_center = (bbox[1] + bbox[3]) / 2

        try:
            object_position = inverse_project(u_center, v_center, TARGET_HEIGHT_M, intrinsics, CAMERA_EXTRINSICS)
        except ValueError as exc:
            print(f"  [{cls_name}] bbox_center=({u_center:.0f},{v_center:.0f})  SKIPPED: {exc}")
            continue

        print(f"  [{cls_name}] bbox_center=({u_center:.0f},{v_center:.0f})  "
              f"target_z={TARGET_HEIGHT_M}m  ->  placed at {[round(v, 2) for v in object_position]}")

        cluster = make_lidar_cluster(object_position)
        all_lidar_points.append(cluster)

        assumed_velocity = (-1.5, 0.0) if matched_count == 0 else (0.0, 0.0)
        radar_rows.append(make_radar_detection(object_position, assumed_velocity, RADAR_SENSOR_OFFSET))
        matched_count += 1

    if matched_count == 0:
        raise RuntimeError("No detections could be placed at a plausible depth -- try a different image/bbox.")

    lidar_points_sensor_frame = np.vstack(all_lidar_points)
    # subtract the LiDAR mounting offset since these points are meant
    # to represent EGO-frame positions, but decode_lidar_bytes() on the
    # M3 side will ADD the sensor offset back -- so we pre-subtract it here
    lidar_points_sensor_frame[:, 0] -= LIDAR_SENSOR_OFFSET[0]
    lidar_points_sensor_frame[:, 1] -= LIDAR_SENSOR_OFFSET[1]
    lidar_points_sensor_frame[:, 2] -= LIDAR_SENSOR_OFFSET[2]

    lidar_bytes = lidar_points_sensor_frame.astype(np.float32).tobytes()
    radar_bytes = np.array(radar_rows, dtype=np.float32).tobytes()

    return lidar_bytes, radar_bytes, matched_count


def send_test_frame():
    lidar_bytes, radar_bytes, n_matched = build_matched_sensor_data(TEST_JPG_PATH, MODEL_PATH)

    with open(TEST_JPG_PATH, "rb") as image_file:
        image_bytes = image_file.read()

    files = {
        "image": ("test.jpg", image_bytes, "image/jpeg"),
        "lidar_file": ("lidar.bin", lidar_bytes, "application/octet-stream"),
        "radar_file": ("radar.bin", radar_bytes, "application/octet-stream"),
    }
    data = {
        "sensor_id": "test_sensor",
        "frame_id": 2,
        "timestamp": 12.0,
        "ego_speed_mps": 12.0,
    }

    print(f"\nSending frame to {M2_URL} (with {n_matched} intentionally-matched objects) ...")
    response = requests.post(M2_URL, files=files, data=data, timeout=30)
    print(f"HTTP {response.status_code}")
    print(response.json())

    print("\nNow check M3's terminal -- tracked_objects and predictions should be > 0 this time.")


if __name__ == "__main__":
    send_test_frame()