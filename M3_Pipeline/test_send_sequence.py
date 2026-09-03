"""
test_send_sequence.py
=======================
Sends a SEQUENCE of frames (not just one) through M2 -> M3 -> M4, with
the matched objects moving slightly each frame -- exactly what's
needed to properly exercise:
    - tracking confirmation (a track needs hits>=2, i.e. matched on at
      least 2 separate frames, before tracker.py reports it)
    - M4 prediction (needs history>=2 snapshots before predicting)

A single one-shot POST (see test_send_frame_matched.py) can only prove
fusion works -- it structurally CANNOT exercise tracking-over-time or
prediction, since both inherently require seeing the same object
across multiple frames. This script is the real end-to-end test.

Reuses the exact same verified inverse-projection and cluster-building
logic as test_send_frame_matched.py (see that file's docstrings for
why each step works the way it does) -- this file just adds a loop
that shifts each matched object's position frame-by-frame according to
its assumed velocity, then sends N frames in sequence.

BEFORE RUNNING:
    1. M2 must be running:  python perception_server.py   (port 8000)
    2. M3 must be running:  python m3_server.py            (port 9000)
       (restart M3 first if you want a clean tracker state --
       otherwise it picks up wherever previous test frames left off)

RUN WITH:
    python test_send_sequence.py
"""

import sys
import os
import time

import cv2
import numpy as np
import requests
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transforms import project_point_to_image, make_default_camera_intrinsics

M2_URL = "http://localhost:8000/api/v1/perception"
TEST_JPG_PATH = "Carla Object Detection/test.jpg"
MODEL_PATH = "Carla Object Detection/best.pt"

CAMERA_EXTRINSICS = {"offset": (1.5, 0.0, 1.4)}
LIDAR_SENSOR_OFFSET = (0.0, 0.0, 1.8)
RADAR_SENSOR_OFFSET = (2.0, 0.0, 0.5)

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

TARGET_HEIGHT_M = 0.5
N_FRAMES = 5
DT = 0.2  # seconds between frames, matches M3's usual rate
STARTING_TIMESTAMP = 12.0


def inverse_project(u, v, target_z_ego, intrinsics, extrinsics, min_depth=2.0, max_depth=60.0):
    """Same verified inverse-projection as test_send_frame_matched.py."""
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    ox, oy, oz = extrinsics["offset"]
    zc_target = target_z_ego - oz
    if abs(v - cy) < 1e-6:
        raise ValueError(f"pixel row v={v} is exactly at the camera's horizon line")
    xc = -zc_target * fy / (v - cy)
    if not (min_depth <= xc <= max_depth):
        raise ValueError(f"solved depth {xc:.1f}m outside plausible range [{min_depth},{max_depth}]m")
    yc = -(u - cx) * xc / fx
    return [xc + ox, yc + oy, target_z_ego]


def make_lidar_cluster(center_xyz, n_points=50, spread_xy=0.4, spread_z=0.3):
    """Same verified cluster generator as test_send_frame_matched.py."""
    rng = np.random.default_rng()
    x = center_xyz[0] + rng.normal(0, spread_xy, n_points)
    y = center_xyz[1] + rng.normal(0, spread_xy, n_points)
    z = center_xyz[2] + rng.normal(0, spread_z, n_points)
    intensity = rng.uniform(0.4, 0.9, n_points)
    return np.stack([x, y, z, intensity], axis=1).astype(np.float32)


def make_radar_detection(position_xy, velocity_xy, radar_offset):
    """Same verified radar builder as test_send_frame_matched.py."""
    dx = position_xy[0] - radar_offset[0]
    dy = position_xy[1] - radar_offset[1]
    depth = float(np.hypot(dx, dy))
    azimuth = float(np.arctan2(dy, dx))
    radial_velocity = float(velocity_xy[0] * np.cos(azimuth) + velocity_xy[1] * np.sin(azimuth))
    return [radial_velocity, azimuth, 0.0, depth]


def detect_base_objects(image_path, model_path):
    """Runs YOLO ONCE to get real bboxes, and computes each matched
    object's STARTING 3D position + an assumed velocity. Frame-to-frame
    motion is then simulated on top of this base position -- YOLO
    doesn't need to rerun per frame since the source image is static."""
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Could not read {image_path}")
    height, width = image.shape[:2]

    if (width, height) != (1280, 720):
        print(f"⚠️  WARNING: {image_path} is {width}x{height}, but m3_server.py defaults to "
              f"1280x720. Set CAMERA_IMAGE_WIDTH={width} CAMERA_IMAGE_HEIGHT={height} "
              f"before starting m3_server.py.\n")

    intrinsics = make_default_camera_intrinsics(image_width=width, image_height=height, fov_deg=90)

    print(f"Running YOLO on {image_path} ({width}x{height}) ...")
    model = YOLO(model_path)
    results = model.predict(source=image, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        raise RuntimeError("YOLO found zero detections -- can't build a matched test.")

    xyxy = boxes.xyxy.cpu().tolist()
    class_ids = boxes.cls.cpu().tolist()
    class_names = [model.names[int(c)] for c in class_ids]

    candidates = [
        (i, name, xyxy[i]) for i, name in enumerate(class_names)
        if name in GROUND_LEVEL_CLASSES
    ][:2]
    if not candidates:
        raise RuntimeError(f"No ground-level classes found among: {set(class_names)}")

    objects = []
    assumed_velocities = [(-1.5, 0.0), (0.0, 0.0)]  # first object approaches, second stays put

    print("\nBase objects (frame 0 positions):")
    for idx, (box_index, cls_name, bbox) in enumerate(candidates):
        u_center = (bbox[0] + bbox[2]) / 2
        v_center = (bbox[1] + bbox[3]) / 2
        try:
            base_position = inverse_project(u_center, v_center, TARGET_HEIGHT_M, intrinsics, CAMERA_EXTRINSICS)
        except ValueError as exc:
            print(f"  [{cls_name}] bbox_center=({u_center:.0f},{v_center:.0f})  SKIPPED: {exc}")
            continue

        velocity = assumed_velocities[len(objects) % len(assumed_velocities)]
        print(f"  [{cls_name}] base_position={[round(v, 2) for v in base_position]}  assumed_velocity={velocity}")
        objects.append({"class": cls_name, "base_position": base_position, "velocity": velocity})

    if not objects:
        raise RuntimeError("No detections could be placed at a plausible depth.")
    return objects


def build_frame_bytes(objects, elapsed_time):
    """Builds LiDAR/radar bytes for ONE frame, with each object's
    position advanced by elapsed_time * its assumed velocity --
    genuine simulated motion, so the Kalman filter has something real
    to estimate velocity from."""
    all_lidar_points = []
    radar_rows = []

    for obj in objects:
        current_position = [
            obj["base_position"][0] + obj["velocity"][0] * elapsed_time,
            obj["base_position"][1] + obj["velocity"][1] * elapsed_time,
            obj["base_position"][2],
        ]
        cluster = make_lidar_cluster(current_position)
        all_lidar_points.append(cluster)
        radar_rows.append(make_radar_detection(current_position[:2], obj["velocity"], RADAR_SENSOR_OFFSET))

    lidar_points_sensor_frame = np.vstack(all_lidar_points)
    lidar_points_sensor_frame[:, 0] -= LIDAR_SENSOR_OFFSET[0]
    lidar_points_sensor_frame[:, 1] -= LIDAR_SENSOR_OFFSET[1]
    lidar_points_sensor_frame[:, 2] -= LIDAR_SENSOR_OFFSET[2]

    lidar_bytes = lidar_points_sensor_frame.astype(np.float32).tobytes()
    radar_bytes = np.array(radar_rows, dtype=np.float32).tobytes()
    return lidar_bytes, radar_bytes


def send_sequence():
    objects = detect_base_objects(TEST_JPG_PATH, MODEL_PATH)

    with open(TEST_JPG_PATH, "rb") as image_file:
        image_bytes = image_file.read()

    print(f"\nSending {N_FRAMES} frames, {DT}s apart ...")
    print(f"{'frame':>6} | {'HTTP':>5} | {'tracked_objects':>15} | {'predictions':>11}")
    print("-" * 50)

    for frame_index in range(N_FRAMES):
        elapsed_time = frame_index * DT
        timestamp = STARTING_TIMESTAMP + elapsed_time
        frame_id = 100 + frame_index  # distinct from earlier single-frame tests

        lidar_bytes, radar_bytes = build_frame_bytes(objects, elapsed_time)

        files = {
            "image": ("test.jpg", image_bytes, "image/jpeg"),
            "lidar_file": ("lidar.bin", lidar_bytes, "application/octet-stream"),
            "radar_file": ("radar.bin", radar_bytes, "application/octet-stream"),
        }
        data = {
            "sensor_id": "test_sensor",
            "frame_id": frame_id,
            "timestamp": timestamp,
            "ego_speed_mps": 12.0,
        }

        response = requests.post(M2_URL, files=files, data=data, timeout=30)
        body = response.json() if response.status_code == 200 else {}

        print(f"{frame_id:6d} | {response.status_code:5d} | "
              f"{'sent -- check M3 log':>15} | {'':>11}")

        time.sleep(0.1)  # small gap so M3's terminal logs are easy to read in order

    print("-" * 50)
    print("\nDone. Check M3's terminal for the full per-frame log lines:")
    print('   "[frame N] tracked_objects=... predictions=..."')
    print(f"\nExpected pattern across these {N_FRAMES} frames:")
    print("  frame 1 (first appearance): tracked_objects likely 0-1 (not yet confirmed)")
    print("  frame 2 onward: tracked_objects should reach 2, predictions should reach 2")
    print("  (unless earlier test frames in this session already confirmed these tracks,")
    print("   in which case you may see 2/2 from frame 1 -- restart m3_server.py for a clean run)")


if __name__ == "__main__":
    send_sequence()