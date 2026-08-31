"""
run_pipeline.py
===============

Real M3 -> M4 integration runner.

M3:
    dummy_data
        -> LiDAR processing
        -> Radar processing
        -> Fusion
        -> MultiObjectTracker
        -> tracked_objects

M4:
    tracked_objects
        -> TrackHistoryBuffer
        -> prediction / uncertainty / TTC / collision / risk

M3 and M4 standalone main.py files are intentionally left untouched.
"""

import json
import os
import sys


# ============================================================
# 1. Make both M3 root and M4_Pipeline imports available
# ============================================================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
M4_DIR = os.path.join(ROOT_DIR, "M4_Pipeline")

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

if M4_DIR not in sys.path:
    sys.path.insert(0, M4_DIR)


# ============================================================
# 2. M3 imports
# ============================================================

from dummy_data import generate_scenario
from lidar_processing import process_lidar_frame, get_raw_roi_points
from radar_processing import process_radar_frame
from fusion import fuse_frame
from tracker import MultiObjectTracker
from transforms import make_default_camera_intrinsics


# ============================================================
# 3. M4 imports
# ============================================================

from buffer import TrackHistoryBuffer
from dummy_ego import generate_dummy_ego
from interface import predict


# ============================================================
# 4. Configuration
# ============================================================

CAMERA_INTRINSICS = make_default_camera_intrinsics(
    image_width=1280,
    image_height=720,
    fov_deg=90
)

CAMERA_EXTRINSICS = {
    "offset": (1.5, 0.0, 1.4)
}

DT = 0.2                  # M3 runs at 5 Hz
MIN_HISTORY_TO_PREDICT = 2


# ============================================================
# 5. Integrated pipeline
# ============================================================

def run():

    # --------------------------------------------------------
    # M3 tracker
    # --------------------------------------------------------

    tracker = MultiObjectTracker(
        max_match_distance_m=2.0,
        min_hits_to_confirm=2,
        max_age_without_update=5,
    )

    # --------------------------------------------------------
    # M4 history
    # --------------------------------------------------------

    history = TrackHistoryBuffer(max_len=15)

    # Store latest prediction for each track
    last_predictions = {}

    # M3 dummy clock starts at 12.0 seconds.
    # M4 dummy ego needs elapsed scenario time.
    t0 = None

    print()
    print("=" * 110)
    print("M3 -> M4 INTEGRATED PIPELINE")
    print("=" * 110)
    print(
        f"{'t':>6} | "
        f"{'M3 tracked objects':>18} | "
        f"{'M4 predictions':>16} | "
        f"{'Risk summary'}"
    )
    print("-" * 110)

    # ========================================================
    # M3 produces one tracked_objects list per frame
    # ========================================================

    for t, camera_dets, lidar_points, radar_dets in generate_scenario(
        n_frames=15,
        dt=DT
    ):

        if t0 is None:
            t0 = t

        # ----------------------------------------------------
        # M3 STEP 1: LiDAR
        # ----------------------------------------------------

        lidar_clusters = process_lidar_frame(lidar_points)

        # Keep raw ROI points for pothole/speed_bump detection.
        raw_roi_points = get_raw_roi_points(lidar_points)

        # ----------------------------------------------------
        # M3 STEP 2: Radar
        # ----------------------------------------------------

        radar_clean = process_radar_frame(radar_dets)

        # ----------------------------------------------------
        # M3 STEP 3: Sensor fusion
        # ----------------------------------------------------

        fused = fuse_frame(
            camera_dets,
            lidar_clusters,
            radar_clean,
            CAMERA_INTRINSICS,
            CAMERA_EXTRINSICS,
            raw_roi_points
        )

        # ----------------------------------------------------
        # M3 STEP 4: Tracking
        # ----------------------------------------------------

        tracked_objects = tracker.step(
            fused,
            dt=DT,
            timestamp=t
        )

        # ====================================================
        # M4 STEP 1: Store M3 snapshots in history
        # ====================================================

        history.update(tracked_objects)

        # ----------------------------------------------------
        # M4 ego state
        # ----------------------------------------------------

        ego_state = generate_dummy_ego(t - t0)

        # Keep M3's actual simulation timestamp
        ego_state["timestamp"] = round(t, 2)

        # ----------------------------------------------------
        # M4 predictions for every confirmed M3 track
        # ----------------------------------------------------

        frame_predictions = []

        for obj in tracked_objects:

            track_id = obj["track_id"]

            track_history = history.get(track_id)

            # M4 needs at least two snapshots to derive
            # meaningful acceleration / motion trends.
            if len(track_history) < MIN_HISTORY_TO_PREDICT:
                continue

            prediction = predict(
                track_id=track_id,
                cls=obj["class"],
                track_history=track_history,
                ego_state=ego_state,
                confidence=obj["confidence"],
            )

            frame_predictions.append(prediction)

            last_predictions[track_id] = prediction

        # ====================================================
        # Print frame summary
        # ====================================================

        risk_summary = ", ".join(
            f"#{p['track_id']} {p['class']}={p['risk_level']}"
            for p in frame_predictions
        )

        if not risk_summary:
            risk_summary = "warming up history..."

        print(
            f"{t:6.2f} | "
            f"{len(tracked_objects):18d} | "
            f"{len(frame_predictions):16d} | "
            f"{risk_summary}"
        )

    # ========================================================
    # Final output
    # ========================================================

    print("-" * 110)

    print()
    print("FINAL M4 PREDICTIONS")
    print("=" * 110)

    if not last_predictions:
        print("No predictions were generated.")
        return

    for track_id, prediction in last_predictions.items():

        printable = {
            key: value
            for key, value in prediction.items()
            if key not in ("trajectories", "debug")
        }

        # Don't print all trajectory points to keep console readable.
        printable["trajectories"] = [
            {
                "mode": branch["mode"],
                "probability": branch["probability"],
                "n_points": len(branch["points"]),
            }
            for branch in prediction["trajectories"]
        ]

        print()
        print(f"--- Track #{track_id} ---")
        print(json.dumps(printable, indent=2))


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    run()