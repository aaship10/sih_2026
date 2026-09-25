"""Standalone sanity test for the M3 pipeline using dummy data (no CARLA/M2
required). Run with: `python tests/test_pipeline_dummy.py` from inside
M3_Pipeline, or `python -m pytest tests -q`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import dummy_data
import sensor_fusion
from tracker import Tracker


def test_single_object_tracked_across_frames():
    tracker = Tracker()
    ego_position = np.array([0.0, 0.0, 0.0])
    timestamp = 0.0
    for i in range(5):
        camera_dets = [dummy_data.dummy_camera_detection()]
        lidar_bytes = dummy_data.dummy_lidar_bytes(object_center_ego=(15.0 + i * 0.3, 0.0, 0.8), seed=i)
        radar_bytes = dummy_data.dummy_radar_bytes(depth=15.0 + i * 0.3)
        fused = sensor_fusion.fuse_frame(camera_dets, lidar_bytes, radar_bytes)
        assert len(fused) >= 1, f"frame {i}: expected at least one fused object"
        detections_world = [{**obj, "position_world": obj["position_ego"] + ego_position} for obj in fused]
        tracker.predict(0.1)
        tracker.update(detections_world, timestamp, ego_position)
        timestamp += 0.1

    confirmed = tracker.confirmed_tracks()
    assert len(confirmed) >= 1, "expected the object to be confirmed as a track after 5 consistent frames"
    assert confirmed[0].hits >= 3
    assert confirmed[0].class_name == "car"
    assert confirmed[0].last_match_pass == "primary"


def test_lidar_only_cluster_becomes_unknown_track():
    tracker = Tracker()
    ego_position = np.array([0.0, 0.0, 0.0])
    timestamp = 0.0
    for i in range(5):
        lidar_bytes = dummy_data.dummy_lidar_bytes(object_center_ego=(10.0, 3.0, 0.8), seed=100 + i)
        radar_bytes = dummy_data.dummy_radar_bytes(depth=0.0)  # no radar target nearby
        fused = sensor_fusion.fuse_frame([], lidar_bytes, radar_bytes)
        assert len(fused) >= 1
        assert fused[0]["class_name"] == "unknown"
        detections_world = [{**obj, "position_world": obj["position_ego"] + ego_position} for obj in fused]
        tracker.predict(0.1)
        tracker.update(detections_world, timestamp, ego_position)
        timestamp += 0.1

    confirmed = tracker.confirmed_tracks()
    assert len(confirmed) >= 1
    assert confirmed[0].class_name == "unknown"


def test_static_track_absorbs_clustering_jitter_via_cascade():
    """Regression test for the live-CARLA finding: 790 track_ids minted over
    265 frames, 92% stuck "unknown", because static clutter's LiDAR cluster
    shape jitters frame to frame and kept failing re-association. A track
    classified `is_static` should absorb a large one-off jump instead of
    spawning a new track_id (see tracker.py's two-pass update)."""
    tracker = Tracker()
    ego_position = np.array([0.0, 0.0, 0.0])
    timestamp = 0.0

    # Establish a confirmed, static LiDAR-only ("unknown") track: an object
    # that sits still for a few frames, like a curb, pothole prop, or parked
    # vehicle edge.
    for i in range(3):
        lidar_bytes = dummy_data.dummy_lidar_bytes(object_center_ego=(20.0, 5.0, 0.8), seed=i)
        radar_bytes = dummy_data.dummy_radar_bytes(depth=0.0)
        fused = sensor_fusion.fuse_frame([], lidar_bytes, radar_bytes)
        detections_world = [{**obj, "position_world": obj["position_ego"] + ego_position} for obj in fused]
        tracker.predict(0.1)
        tracker.update(detections_world, timestamp, ego_position)
        timestamp += 0.1

    assert len(tracker.tracks) == 1
    track = tracker.tracks[0]
    assert track.status == "confirmed"
    assert track.is_static
    original_id = track.track_id

    # A large clustering jitter for the SAME physical clutter (e.g. a
    # different subset of points on an extended static object returned this
    # sweep) -- big enough that a tightly-converged primary Mahalanobis gate
    # would plausibly reject it, but well within the static cascade's radius.
    lidar_bytes = dummy_data.dummy_lidar_bytes(object_center_ego=(24.5, 5.0, 0.8), seed=42)
    radar_bytes = dummy_data.dummy_radar_bytes(depth=0.0)
    fused = sensor_fusion.fuse_frame([], lidar_bytes, radar_bytes)
    detections_world = [{**obj, "position_world": obj["position_ego"] + ego_position} for obj in fused]
    tracker.predict(0.1)
    tracker.update(detections_world, timestamp, ego_position)

    assert len(tracker.tracks) == 1, "clustering jitter on a static object should not spawn a new track"
    assert tracker.tracks[0].track_id == original_id
    assert tracker.tracks[0].last_match_pass == "cascade"


def test_cascade_match_does_not_hijack_velocity():
    """Regression test for the exact live-CARLA (Run #3) finding: a pure
    LiDAR "unknown" track jumped 0.29 -> 14.82 -> 29.8 m/s across 3 frames
    because Pass 2 (the cascade) matches on flat Euclidean distance alone,
    with no Mahalanobis/velocity check -- so a fast, unrelated cluster that
    happened to land within the cascade radius of an old slow/static track
    got a full-trust Kalman position update, which forced a huge velocity
    change too. Constructs detections directly (bypassing sensor_fusion) for
    precise control over the geometry."""
    tracker = Tracker()
    ego_position = np.array([0.0, 0.0, 0.0])
    timestamp = 0.0

    def unknown_det(position_world):
        return {
            "position_world": np.array(position_world),
            "size": np.array([0.5, 0.5, 1.0]),
            "class_name": "unknown",
            "confidence": 0.4,
            "sensor_sources": ["lidar"],
        }

    # Establish a confirmed, slow/near-static "unknown" track.
    for i in range(5):
        tracker.predict(0.1)
        tracker.update([unknown_det([20.0 + i * 0.05, 5.0, 0.8])], timestamp, ego_position)
        timestamp += 0.1

    track = tracker.confirmed_tracks()[0]
    speed_before = np.linalg.norm(track.kf.velocity)
    assert speed_before < 1.0

    # A fast, unrelated LiDAR cluster lands 8m away in a single frame -- far
    # enough to fail Pass 1's Mahalanobis gate, but within the (Euclidean)
    # cascade radius, and class "unknown" on both sides so it's cascade-eligible.
    timestamp += 0.1
    tracker.predict(0.1)
    tracker.update([unknown_det([28.0, 5.0, 0.8])], timestamp, ego_position)

    track = tracker.confirmed_tracks()[0]
    speed_after_one_hit = np.linalg.norm(track.kf.velocity)
    assert speed_after_one_hit < 5.0, (
        "a single cascade match to an unrelated fast-looking cluster must not hijack this track's velocity "
        f"(got {speed_after_one_hit:.2f} m/s; the pre-fix behavior produced ~11-15 m/s here)"
    )


def test_single_frame_radar_blip_does_not_move_velocity():
    """Regression test for the live-CARLA finding: radar association
    flickering on/off produced a velocity sawtooth (mean |delta speed| 1.27
    m/s on radar-touched frames vs 0.28 m/s baseline). A single flickering
    radar hit should not move a stable track's velocity; two consecutive
    hits (config.RADAR_MIN_STREAK) should."""
    tracker = Tracker()
    ego_position = np.array([0.0, 0.0, 0.0])
    timestamp = 0.0

    for i in range(4):
        camera_dets = [dummy_data.dummy_camera_detection()]
        lidar_bytes = dummy_data.dummy_lidar_bytes(object_center_ego=(15.0 + i * 0.1, 0.0, 0.8), seed=i)
        radar_bytes = dummy_data.dummy_radar_bytes(depth=0.0)  # no radar yet
        fused = sensor_fusion.fuse_frame(camera_dets, lidar_bytes, radar_bytes)
        detections_world = [{**obj, "position_world": obj["position_ego"] + ego_position} for obj in fused]
        tracker.predict(0.1)
        tracker.update(detections_world, timestamp, ego_position)
        timestamp += 0.1

    track = tracker.confirmed_tracks()[0]
    speed_before = np.linalg.norm(track.kf.velocity)

    # One-off radar blip with an inconsistent range-rate -- mild enough to
    # pass the LOS consistency gate (radar_los_innovation) as "plausible
    # given this track's current uncertainty" on its own (so radar_streak
    # can start), but the RADAR_MIN_STREAK requirement should still hold off
    # actually applying it after just one occurrence. (A more extreme value,
    # e.g. -8.0, gets rejected by the consistency gate outright on the very
    # first frame -- a stronger, separate protection also worth having, but
    # not what this streak-gating test targets.)
    camera_dets = [dummy_data.dummy_camera_detection()]
    lidar_bytes = dummy_data.dummy_lidar_bytes(object_center_ego=(15.4, 0.0, 0.8), seed=99)
    radar_bytes = dummy_data.dummy_radar_bytes(depth=15.4, velocity_mps=-5.0)
    fused = sensor_fusion.fuse_frame(camera_dets, lidar_bytes, radar_bytes)
    detections_world = [{**obj, "position_world": obj["position_ego"] + ego_position} for obj in fused]
    timestamp += 0.1
    tracker.predict(0.1)
    tracker.update(detections_world, timestamp, ego_position)

    track = tracker.confirmed_tracks()[0]
    assert track.radar_streak == 1
    speed_after_one_blip = np.linalg.norm(track.kf.velocity)
    assert abs(speed_after_one_blip - speed_before) < 0.5, "a single radar blip should not have moved the velocity yet"

    # Two more consecutive frames with the same reading -- streak now
    # satisfies RADAR_MIN_STREAK, so it should take effect.
    for k in range(2):
        depth = 15.4 + 0.1 * (k + 1)
        camera_dets = [dummy_data.dummy_camera_detection()]
        lidar_bytes = dummy_data.dummy_lidar_bytes(object_center_ego=(depth, 0.0, 0.8), seed=200 + k)
        radar_bytes = dummy_data.dummy_radar_bytes(depth=depth, velocity_mps=-5.0)
        fused = sensor_fusion.fuse_frame(camera_dets, lidar_bytes, radar_bytes)
        detections_world = [{**obj, "position_world": obj["position_ego"] + ego_position} for obj in fused]
        timestamp += 0.1
        tracker.predict(0.1)
        tracker.update(detections_world, timestamp, ego_position)

    track = tracker.confirmed_tracks()[0]
    assert track.radar_streak >= 2
    speed_after_streak = np.linalg.norm(track.kf.velocity)
    assert abs(speed_after_streak - speed_before) > 0.2, "two consecutive radar hits should measurably influence velocity"


def test_radar_consistency_gate_rejects_wildly_inconsistent_reading():
    """Regression test for the exact live-CARLA bug: track_id 59 jumped from
    0.15 m/s to 40.27 m/s (144 km/h) in one 0.1s step, right as
    RADAR_MIN_STREAK first passed -- a fast, unrelated target (a racer bike)
    happened to be the nearest radar return to a slow/static LiDAR cluster.
    A reading this inconsistent with a track's own converged velocity
    estimate should now be rejected by radar_los_innovation before it can
    even start a streak, however many times it recurs."""
    tracker = Tracker()
    ego_position = np.array([0.0, 0.0, 0.0])
    timestamp = 0.0

    for i in range(4):
        camera_dets = [dummy_data.dummy_camera_detection()]
        lidar_bytes = dummy_data.dummy_lidar_bytes(object_center_ego=(15.0 + i * 0.02, 0.0, 0.8), seed=i)
        radar_bytes = dummy_data.dummy_radar_bytes(depth=0.0)
        fused = sensor_fusion.fuse_frame(camera_dets, lidar_bytes, radar_bytes)
        detections_world = [{**obj, "position_world": obj["position_ego"] + ego_position} for obj in fused]
        tracker.predict(0.1)
        tracker.update(detections_world, timestamp, ego_position)
        timestamp += 0.1

    track = tracker.confirmed_tracks()[0]
    speed_before = np.linalg.norm(track.kf.velocity)
    assert speed_before < 1.0  # a near-static object, like the reported case

    # A racer-bike-speed radar return (~20 m/s closing) recurring for 3
    # frames on this otherwise near-static object's cluster.
    for k in range(3):
        depth = 15.1 + k * 0.02
        camera_dets = [dummy_data.dummy_camera_detection()]
        lidar_bytes = dummy_data.dummy_lidar_bytes(object_center_ego=(depth, 0.0, 0.8), seed=300 + k)
        radar_bytes = dummy_data.dummy_radar_bytes(depth=depth, velocity_mps=-20.0)
        fused = sensor_fusion.fuse_frame(camera_dets, lidar_bytes, radar_bytes)
        detections_world = [{**obj, "position_world": obj["position_ego"] + ego_position} for obj in fused]
        timestamp += 0.1
        tracker.predict(0.1)
        tracker.update(detections_world, timestamp, ego_position)
        track = tracker.confirmed_tracks()[0]
        assert track.radar_streak == 0, f"frame {k}: an inconsistent reading should never start a streak"

    speed_after = np.linalg.norm(track.kf.velocity)
    assert speed_after < 5.0, "a wildly inconsistent radar target must not be able to hijack this track's velocity"


if __name__ == "__main__":
    test_single_object_tracked_across_frames()
    test_lidar_only_cluster_becomes_unknown_track()
    test_static_track_absorbs_clustering_jitter_via_cascade()
    test_cascade_match_does_not_hijack_velocity()
    test_single_frame_radar_blip_does_not_move_velocity()
    test_radar_consistency_gate_rejects_wildly_inconsistent_reading()
    print("OK: dummy pipeline tests passed")
