"""Offline replay of the real M5_Pipeline code (UNMODIFIED) against the
recorded M4 input/prediction logs, to trace internal state that isn't
normally logged (which obstacle _nearest_risky_obstacle_direction picked,
per-track TTC, etc). Read-only investigation script -- calls existing
functions as-is, does its own bookkeeping alongside them."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\Lenovo\temp\M4_Pipeline")
sys.path.insert(0, r"C:\Users\Lenovo\temp\M4_Pipeline\output")
sys.path.insert(0, r"C:\Users\Lenovo\temp\M5_Pipeline")

import config as m5_config
m5_config.USE_STATEFLOW_FSM = False  # fast Python FSM for replay -- parity-verified identical to Stateflow

from m5_udp_schema import build_obstacle_packet
from schema import parse_obstacle_packet
from planner import M5Engine, _nearest_risky_obstacle_direction, _required_avoid_offset_m
from ttc import compute_ttc

raw_recs = {r["frame_id"]: r for r in (json.loads(l) for l in open(r"C:\Users\Lenovo\temp\M4_Pipeline\logs\m4_input_packets.jsonl"))}
pred_recs = {r["frame_id"]: r for r in (json.loads(l) for l in open(r"C:\Users\Lenovo\temp\M4_Pipeline\logs\m4_predictions.jsonl"))}
frame_ids = sorted(set(raw_recs) & set(pred_recs))
print("total frames available:", len(frame_ids), "range", frame_ids[0], frame_ids[-1])

WINDOW_LO, WINDOW_HI = 6600, 6800
window_frames = [f for f in frame_ids if WINDOW_LO <= f <= WINDOW_HI]
print("frames in window:", window_frames)
print()

engine = M5Engine()  # ONE persistent engine across the whole session -- correct artifact-history/FSM continuity

CONE_CANDIDATES = set()  # track_ids seen as class in {'unknown'} with small size, to help identify the cone later

for fid in frame_ids:
    raw = raw_recs[fid]
    pred = pred_recs[fid]
    obstacle_packet_dict = build_obstacle_packet(raw, pred)
    packet = parse_obstacle_packet(obstacle_packet_dict)
    ego_x, ego_y = packet.ego_position[0], packet.ego_position[1]
    ego_vx, ego_vy = packet.ego_velocity[0], packet.ego_velocity[1]
    ego_yaw_deg = packet.ego_yaw_deg
    heading = (math.cos(math.radians(ego_yaw_deg)), math.sin(math.radians(ego_yaw_deg)))

    decision = engine.step(packet)  # advances state; packet.obstacles/.statics mutated in-place to POST-FILTER

    if fid not in window_frames:
        continue

    # -- which track is "nearest" (re-derive, matching _nearest_risky_obstacle_direction's own search) --
    best_d, best_obj, best_kind = math.inf, None, None
    for o in packet.obstacles:
        d = math.hypot(o.pos_x - ego_x, o.pos_y - ego_y)
        if d < best_d:
            best_d, best_obj, best_kind = d, o, "obstacle"
    for s in packet.statics:
        d = math.hypot(s.pos_x - ego_x, s.pos_y - ego_y)
        if d < best_d:
            best_d, best_obj, best_kind = d, s, "static"

    nearest_result = _nearest_risky_obstacle_direction(ego_x, ego_y, packet)

    # -- which track produces min TTC (obstacles only, per ttc.py's own scope) --
    best_ttc, best_ttc_track = math.inf, None
    for o in packet.obstacles:
        t = compute_ttc((ego_x, ego_y), (ego_vx, ego_vy), (o.pos_x, o.pos_y), (o.vel_x, o.vel_y))
        if t is not None and t < best_ttc:
            best_ttc, best_ttc_track = t, o.track_id
    if best_ttc_track is None:
        best_ttc = None

    bearing_deg = None
    required = None
    if nearest_result is not None:
        direction, half_width = nearest_result
        lateral_component = direction[0] * -heading[1] + direction[1] * heading[0]
        bearing_deg = math.degrees(math.asin(max(-1.0, min(1.0, lateral_component))))
        required = _required_avoid_offset_m(half_width)

    traj_dx = decision.trajectory[-1].x - decision.trajectory[0].x
    traj_dy = decision.trajectory[-1].y - decision.trajectory[0].y
    lateral_offset_commanded = -heading[1] * (0) + 0  # placeholder, computed below properly
    # Recover the actual commanded lateral offset from the trajectory target
    # (target = ego + heading*dist - perp(heading)*offset); solve via projection.
    perp = (-heading[1], heading[0])
    offset_from_traj = -(traj_dx * perp[0] + traj_dy * perp[1])

    same_track = None
    nearest_track_id = None
    nearest_class = None
    nearest_pos = None
    nearest_size = None
    if best_obj is not None:
        nearest_track_id = best_obj.track_id
        nearest_class = best_obj.class_name
        nearest_pos = (round(best_obj.pos_x, 2), round(best_obj.pos_y, 2))
        nearest_size = (best_obj.width, best_obj.length)
        if best_kind == "obstacle" and best_ttc_track is not None:
            same_track = (nearest_track_id == best_ttc_track)

    print(f"fid={fid} state={decision.behavior_state} min_ttc={engine.last_diagnostics.get('min_ttc')} "
          f"now={engine.last_diagnostics.get('now_risk')} fut={engine.last_diagnostics.get('future_risk')} "
          f"n_obs={len(packet.obstacles)} n_stat={len(packet.statics)}")
    print(f"    nearest: kind={best_kind} track_id={nearest_track_id} class={nearest_class} "
          f"pos={nearest_pos} size={nearest_size} dist={best_d:.2f} bearing_deg={bearing_deg}")
    print(f"    ttc_threat_track={best_ttc_track} ttc={best_ttc} same_as_nearest={same_track}")
    print(f"    required_offset={required} traj_derived_offset={offset_from_traj:.3f}")
    print()
