"""Builds the exact JSON payload M5_Pipeline's UDP input contract expects
(M5_Pipeline/README.md section 1 / the spec it was built from), from the
raw M3 wire packet (position/velocity/size/confidence -- current state) and
M4's own PredictionPacket (trajectories -- predicted state). Two different
sources because M4's internal ObjectPrediction (output/schema.py) never
carried "current state" fields at all (M5 didn't exist yet when that schema
was proposed) -- rebuilding them here would duplicate TrackedObject, so this
reads them straight from the same raw dict m4_server.py already validated.

Resolves the open items the M5 spec explicitly left for M4 to decide:

  (A) Static tracks: M3's `is_static=True` objects are split into their own
      top-level `statics` list (track_id/class/confidence/pos_x/pos_y/
      width/length -- no velocity or trajectories, there is nothing to
      predict for something that isn't moving) instead of living in
      `obstacles` next to real dynamic predictions. Keeps M5's costmap-
      inflation-only path for statics separate from its TTC/trajectory-
      avoidance path for movers. `confidence` was added after a live
      incident: M5's sensor-artifact filter (config.py there) needs it to
      recognize a LiDAR-only-confidence "unknown" static right next to the
      ego as a likely self-detection artifact, not a real fixture.
  (B) Uncertainty: ADDED to the spec's minimal trajectory shape as a
      sibling `sigma_m` array (one combined per-point radius, computed from
      M4's existing along/lateral sigma as sqrt(along^2+lateral^2)/2 --
      a simple, conservative circular radius, not the full oriented
      ellipse M4 uses internally) alongside `points`. M5's costmap can
      inflate an obstacle's risk zone by this radius directly; a receiver
      that ignores the field still gets exactly the spec's original shape.
  (C)/(D) Frames and units: WORLD frame, meters, radians -- matching what
      M3/M4 already use internally end to end (position/velocity are
      passed straight through, unconverted). M5_Pipeline's own output back
      to M1 keeps the same convention for the same reason (one frame/unit
      system across the whole M1-M5 loop means zero silent conversion bugs
      at any hop).
"""
from __future__ import annotations

import math
from typing import Any


def _point_and_sigma(points: list[list[float]], times_s: list[float],
                      along_sigma_m: list[float], lateral_sigma_m: list[float]) -> tuple[list[dict], list[float]]:
    pts = [{"x": p[0], "y": p[1], "t": t} for p, t in zip(points, times_s)]
    sigma = [math.hypot(a, l) / 2.0 for a, l in zip(along_sigma_m, lateral_sigma_m)]
    return pts, sigma


def build_obstacle_packet(raw_frame_packet: dict[str, Any], prediction_packet_dict: dict[str, Any]) -> dict[str, Any]:
    """Both arguments are plain dicts: `raw_frame_packet` is the JSON body
    M3 posted (already validated as a FramePacket by the caller),
    `prediction_packet_dict` is PredictionPacket.model_dump(). Joined by
    track_id -- every object in predictions came from the same frame's
    tracked_objects, so this always finds a match."""
    raw_by_id = {o["track_id"]: o for o in raw_frame_packet.get("tracked_objects", [])}

    obstacles: list[dict] = []
    statics: list[dict] = []

    for pred in prediction_packet_dict.get("predictions", []):
        track_id = pred["track_id"]
        raw = raw_by_id.get(track_id)
        if raw is None:
            continue  # should not happen -- every prediction comes from a tracked_objects entry in this same frame
        length, width = raw["size"][0], raw["size"][1]

        if pred["is_static"]:
            statics.append({
                "track_id": track_id,
                "class": pred["class_name"],
                "confidence": raw["confidence"],  # M5's own sensor-artifact filter needs this for statics too -- see M5_Pipeline/config.py
                "pos_x": raw["position"][0],
                "pos_y": raw["position"][1],
                "width": width,
                "length": length,
            })
            continue

        time_step_s = pred["time_step_s"]
        trajectories = []
        for traj in pred["trajectories"]:
            times_s = [time_step_s * i for i in range(len(traj["points"]))]
            pts, sigma = _point_and_sigma(traj["points"], times_s, traj["along_sigma_m"], traj["lateral_sigma_m"])
            trajectories.append({"mode": traj["mode"], "probability": traj["probability"], "points": pts, "sigma_m": sigma})

        obstacles.append({
            "track_id": track_id,
            "class": pred["class_name"],
            "confidence": raw["confidence"],  # M2's classification confidence (raw, "Current State") -- NOT pred["confidence"] (M4's own predicted-path confidence, a different concept, see predictor.py's docstring)
            "pos_x": raw["position"][0],
            "pos_y": raw["position"][1],
            "vel_x": raw["velocity"][0],
            "vel_y": raw["velocity"][1],
            "width": width,
            "length": length,
            "trajectories": trajectories,
        })

    return {
        "timestamp": prediction_packet_dict["timestamp"],
        "frame_id": prediction_packet_dict["frame_id"],
        "ego_position": prediction_packet_dict["ego_position"],
        "ego_velocity": prediction_packet_dict["ego_velocity"],
        "ego_yaw_deg": prediction_packet_dict["ego_yaw_deg"],
        "obstacles": obstacles,
        "statics": statics,
    }
