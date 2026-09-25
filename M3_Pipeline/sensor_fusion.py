"""Frame-level sensor fusion: combine M2's camera detections with this
frame's LiDAR and radar processing into a list of fused object candidates,
ready for the tracker.

Fusion order (M3 mentor notes, sections 11/14):
  1. Camera <-> LiDAR via projection + IoU (camera gives class, LiDAR gives
     accurate 3D position/size).
  2. Camera-only detections that found no LiDAR support fall back to a
     direct camera-radar projection match, then a ground-plane
     back-projection for a rough position -- better than dropping them
     outright (e.g. a sudden cattle/pedestrian appearance with only 1-2
     frames of sparse LiDAR returns so far).
  3. LiDAR-only clusters (no camera match) are kept as "unknown" obstacles --
     still important for collision avoidance even without a class label.
  4. Radar is associated last, to whichever candidate (from 1-3) is
     geometrically closest in the ego frame, and only ever contributes a
     velocity estimate. This is deliberately done at the fusion stage,
     *before* tracking: radar's job here is to seed the Kalman filter's
     velocity state with a real measurement instead of waiting for it to
     converge from position differences alone (this matters most exactly
     where it's most useful -- a fast highway-merge closing speed, or a
     racer bike appearing from behind).
"""
from __future__ import annotations

import numpy as np

import camera_lidar_fusion
import camera_radar_fusion
import config
import lidar_processing
import radar_processing


def _nearest_radar(position_ego: np.ndarray, radar_targets: list[dict]) -> dict | None:
    if not radar_targets:
        return None
    dists = [np.linalg.norm(t["position_ego"] - position_ego) for t in radar_targets]
    j = int(np.argmin(dists))
    if dists[j] <= config.RADAR_ASSOC_MAX_DIST_M:
        return radar_targets[j]
    return None


def fuse_frame(camera_detections: list[dict], lidar_bytes: bytes, radar_bytes: bytes) -> list[dict]:
    camera_detections = [d for d in camera_detections if d["confidence"] >= config.CAMERA_MIN_CONFIDENCE]

    lidar_clusters = lidar_processing.process(lidar_bytes)
    radar_targets = radar_processing.process(radar_bytes)

    matches, unmatched_cam, unmatched_lidar = camera_lidar_fusion.associate(camera_detections, lidar_clusters)

    fused: list[dict] = []

    for cam_idx, lidar_idx in matches:
        det, cluster = camera_detections[cam_idx], lidar_clusters[lidar_idx]
        fused.append({
            "position_ego": cluster["centroid_ego"],
            "size": cluster["size"],
            "class_name": det["class_name"],
            "confidence": det["confidence"],
            "sensor_sources": ["camera", "lidar"],
            "num_lidar_points": cluster["num_points"],
        })

    # Camera-only detections: try direct radar association, then ground-plane back-projection.
    radar_cam_matches = camera_radar_fusion.associate_unmatched(camera_detections, unmatched_cam, radar_targets)
    for cam_idx in unmatched_cam:
        det = camera_detections[cam_idx]
        sources = ["camera"]
        if cam_idx in radar_cam_matches:
            position = radar_targets[radar_cam_matches[cam_idx]]["position_ego"]
            sources.append("radar")
        else:
            position = camera_lidar_fusion.ground_plane_backproject(det["box"])
        if position is None:
            continue  # cannot localize this detection in 3D; drop it
        fused.append({
            "position_ego": position,
            "size": np.array([0.6, 0.6, 1.0]),  # unknown extent; generic placeholder
            "class_name": det["class_name"],
            "confidence": det["confidence"] * 0.6,  # lower confidence: no direct 3D support
            "sensor_sources": sources,
            "num_lidar_points": 0,
        })

    # LiDAR-only clusters: kept as unclassified obstacles.
    for lidar_idx in unmatched_lidar:
        cluster = lidar_clusters[lidar_idx]
        fused.append({
            "position_ego": cluster["centroid_ego"],
            "size": cluster["size"],
            "class_name": "unknown",
            "confidence": 0.4,
            "sensor_sources": ["lidar"],
            "num_lidar_points": cluster["num_points"],
        })

    # Attach the nearest radar target (if any) to every candidate as a velocity cue.
    for obj in fused:
        if "radar" in obj["sensor_sources"]:
            continue
        target = _nearest_radar(obj["position_ego"], radar_targets)
        if target is not None:
            obj["sensor_sources"] = obj["sensor_sources"] + ["radar"]
            obj["radar_range_rate_mps"] = target["range_rate_mps"]
            obj["radar_target"] = target

    return fused
