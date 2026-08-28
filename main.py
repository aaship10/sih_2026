"""
main.py
=======
Runs the complete M3 pipeline over a synthetic scenario, frame by
frame, and prints the tracked objects -- exactly what you'd hand to M4.

    dummy_data  ->  lidar_processing  ->  \
                    radar_processing  ->   fusion  ->  tracker  ->  M4
                    (M2 camera dets)  ->  /

HOW TO SWITCH TO REAL CARLA / REAL M2 DATA LATER
--------------------------------------------------
Only this file changes. Replace the calls to dummy_data.generate_scenario()
with your real CARLA sensor callbacks + M2's real detections, keeping the
exact same variable names (camera_dets, lidar_points, radar_dets) --
every other file (lidar_processing.py, radar_processing.py, fusion.py,
tracker.py) works unchanged, because they only care about the FORMAT of
the data, not where it came from.

RUN THIS WITH:
    python main.py
"""

import json

from dummy_data import generate_scenario
from lidar_processing import process_lidar_frame, get_raw_roi_points
from radar_processing import process_radar_frame
from fusion import fuse_frame
from tracker import MultiObjectTracker
from transforms import make_default_camera_intrinsics

# --- camera calibration (CARLA gives you these for free -- see transforms.py) ---
CAMERA_INTRINSICS = make_default_camera_intrinsics(image_width=1280, image_height=720, fov_deg=90)
CAMERA_EXTRINSICS = {"offset": (1.5, 0.0, 1.4)}  # camera mounted 1.5m fwd, 1.4m up from ego center

DT = 0.2  # seconds between frames (5 Hz) -- matches dummy_data's scenario generator


def run():
    tracker = MultiObjectTracker(
        max_match_distance_m=2.0,   # how close a new detection must be to an
                                     # existing track's predicted position to count
                                     # as "the same object"
        min_hits_to_confirm=2,      # need 2 matched frames before we trust a track
        max_age_without_update=5,   # forgive up to 5 missed frames (occlusion)
    )

    print(f"{'t':>5} | {'tracked_objects (id: class @ position, velocity)'}")
    print("-" * 80)

    for t, camera_dets, lidar_points, radar_dets in generate_scenario(n_frames=15, dt=DT):
        # Step 1: LiDAR processing -> clusters (candidate objects)
        lidar_clusters = process_lidar_frame(lidar_points)
        # ALSO keep raw (non-ground-removed) ROI points -- needed for
        # road-surface classes like pothole/speed_bump (see fusion.py)
        raw_roi_points = get_raw_roi_points(lidar_points)

        # Step 2: Radar processing -> clean (position, velocity) list
        radar_clean = process_radar_frame(radar_dets)

        # Step 3: Fusion -> match camera + lidar + radar into fused objects
        fused = fuse_frame(camera_dets, lidar_clusters, radar_clean,
                            CAMERA_INTRINSICS, CAMERA_EXTRINSICS, raw_roi_points)

        # Step 4: Tracking -> assign/maintain persistent IDs, smooth with Kalman filter
        tracked_objects = tracker.step(fused, dt=DT, timestamp=t)

        # This is exactly what you hand to M4 each frame:
        summary = ", ".join(
            f"#{o['track_id']}: {o['class']} @ {o['position'][:2]}, v={o['velocity'][:2]}"
            for o in tracked_objects
        )
        print(f"{t:5.2f} | {summary}")

    print("-" * 80)
    print("\nFinal tracked_objects list (this is the M3 -> M4 interface):\n")
    print(json.dumps(tracked_objects, indent=2))


if __name__ == "__main__":
    run()