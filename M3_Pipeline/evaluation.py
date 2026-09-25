"""Offline evaluation: compares M3's tracked_objects (logs/m3_tracks.jsonl)
against CARLA ground truth.

Ground truth is NEVER fed into the perception/fusion/tracking pipeline (M3
mentor notes, section 24) -- it is only used here, after a run, for scoring.
scenario3.py (M1) already prints a `[GROUND TRUTH][frame N]` block every tick
with each actor's world position/velocity; redirect that stdout to a file and
pass it here alongside the M3 tracks log:

    python scenario3.py > logs/scenario_stdout.log
    python evaluation.py logs/scenario_stdout.log logs/m3_tracks.jsonl

Matching is nearest-position-in-world-frame per frame (M3 tracks don't know
CARLA's actor IDs, so identity can't be compared directly -- only position/
count-based metrics are computed here).
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict

GT_FRAME_RE = re.compile(r"\[GROUND TRUTH\]\[frame (\d+)\]")
GT_ACTOR_RE = re.compile(
    r"^\s*(\w+) actor=(\d+).*?pos=\(([-\d.]+),([-\d.]+)\).*?vel=\(([-\d.]+),([-\d.]+)\)"
)

MATCH_DIST_THRESHOLD_M = 5.0


def parse_ground_truth(path: str) -> dict[int, list[dict]]:
    frames: dict[int, list[dict]] = defaultdict(list)
    current_frame = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = GT_FRAME_RE.search(line)
            if m:
                current_frame = int(m.group(1))
                continue
            m = GT_ACTOR_RE.match(line)
            if m and current_frame is not None:
                kind, actor_id, x, y, vx, vy = m.groups()
                if kind == "EGO":
                    continue
                frames[current_frame].append({
                    "actor_id": int(actor_id), "kind": kind,
                    "x": float(x), "y": float(y), "vx": float(vx), "vy": float(vy),
                })
    return frames


def parse_tracks(path: str) -> dict[int, list[dict]]:
    frames: dict[int, list[dict]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            frames[record["frame_id"]] = record["tracked_objects"]
    return frames


def evaluate(gt_frames: dict[int, list[dict]], track_frames: dict[int, list[dict]]) -> dict:
    pos_errors: list[float] = []
    matched = total_gt = false_tracks = 0
    common_frames = sorted(set(gt_frames) & set(track_frames))

    for frame in common_frames:
        gts = gt_frames[frame]
        tracks = track_frames[frame]
        total_gt += len(gts)
        used_tracks: set[int] = set()
        for gt in gts:
            best_i, best_d = None, float("inf")
            for i, t in enumerate(tracks):
                if i in used_tracks:
                    continue
                d = ((t["position"][0] - gt["x"]) ** 2 + (t["position"][1] - gt["y"]) ** 2) ** 0.5
                if d < best_d:
                    best_d, best_i = d, i
            if best_i is not None and best_d < MATCH_DIST_THRESHOLD_M:
                matched += 1
                used_tracks.add(best_i)
                pos_errors.append(best_d)
        false_tracks += len(tracks) - len(used_tracks)

    return {
        "frames_compared": len(common_frames),
        "ground_truth_objects": total_gt,
        "matched": matched,
        "match_rate": (matched / total_gt) if total_gt else 0.0,
        "mean_position_error_m": (sum(pos_errors) / len(pos_errors)) if pos_errors else None,
        "false_track_instances": false_tracks,
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluation.py <scenario_stdout.log> <m3_tracks.jsonl>")
        raise SystemExit(1)
    gt = parse_ground_truth(sys.argv[1])
    tracks = parse_tracks(sys.argv[2])
    print(json.dumps(evaluate(gt, tracks), indent=2))
