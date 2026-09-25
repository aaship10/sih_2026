"""Phase 5 runner: scores logs/m4_predictions.jsonl against a scenario's
GROUND TRUTH stdout capture using metrics.py + ground_truth.py.

Usage: python -m evaluation.run_eval [predictions.jsonl] [ground_truth.log]
"""
from __future__ import annotations

import json
import sys

from evaluation.ground_truth import parse_ground_truth
from evaluation.metrics import aggregate, evaluate_matched_trajectory, match_actors_unique


def main() -> None:
    pred_path = sys.argv[1] if len(sys.argv) > 1 else "logs/m4_predictions.jsonl"
    gt_path = sys.argv[2] if len(sys.argv) > 2 else "logs/scenario3_ground_truth.log"

    gt_frames = parse_ground_truth(gt_path)
    print(f"Parsed {len(gt_frames)} ground-truth frames from {gt_path}")

    QUALITY_BUCKETS = [("low_quality(<0.5)", lambda q: q < 0.5), ("high_quality(>=0.5)", lambda q: q >= 0.5)]

    # errors_by_bucket[(mode, bucket_name)] -> list[TrajectoryError]
    errors_by_bucket: dict[tuple[str, str], list] = {}
    n_packets = 0
    n_predictions = 0
    n_unmatched = 0

    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            packet = json.loads(line)
            n_packets += 1
            start_frame_id = packet["frame_id"]
            start_timestamp = packet["timestamp"]
            n_predictions += len(packet["predictions"])

            start_frame = gt_frames.get(start_frame_id)
            if start_frame is None:
                n_unmatched += len(packet["predictions"])
                continue

            # One match per OBJECT per frame (all its trajectory modes share
            # the same t=0 point), resolved uniquely against this frame's
            # actors -- see metrics.match_actors_unique()'s docstring for
            # why matching independently per trajectory-mode double- and
            # triple-counts (and can duplicate-claim) the same identity.
            candidates = [(obj["track_id"], obj["trajectories"][0]["points"][0][0],
                           obj["trajectories"][0]["points"][0][1]) for obj in packet["predictions"]]
            matches = match_actors_unique(candidates, start_frame["actors"])

            for obj in packet["predictions"]:
                actor_id = matches.get(obj["track_id"])
                if actor_id is None:
                    n_unmatched += 1
                    continue
                time_step_s = obj["time_step_s"]
                quality = obj["input_quality"]
                bucket_name = next(name for name, pred in QUALITY_BUCKETS if pred(quality))
                per_mode_errs = []
                for traj in obj["trajectories"]:
                    points = traj["points"]
                    # points[0] is t=0 (current position, see
                    # models/trajectory.py's sample_times) -- NOT one step
                    # into the future.
                    times_s = [time_step_s * i for i in range(len(points))]
                    err = evaluate_matched_trajectory(
                        points, times_s, start_frame_id, start_timestamp,
                        gt_frames, obj["track_id"], actor_id,
                    )
                    errors_by_bucket.setdefault((traj["mode"], bucket_name), []).append(err)
                    per_mode_errs.append(err)

                # M4 is a MULTIMODAL predictor (nominal/stop/lateral): the
                # standard metric for that (minADE_k/minFDE_k/MissRate_k in
                # the trajectory-prediction literature, e.g. nuScenes'
                # prediction challenge) scores the BEST of the k candidate
                # modes per object, not each mode judged in isolation --
                # a single mode "missing" is expected and by design (that's
                # why the other modes exist), so per-mode-only numbers
                # overstate how often M4 as a whole fails to cover reality.
                scored = [e for e in per_mode_errs if e.n_points_scored > 0]
                if scored:
                    best = min(scored, key=lambda e: e.fde_m)
                    errors_by_bucket.setdefault(("BEST-OF-3", bucket_name), []).append(best)

    print(f"Scanned {n_packets} prediction packets, {n_predictions} object-predictions "
          f"({n_unmatched} unmatched to any ground-truth actor)\n")

    for (mode, bucket_name), errors in sorted(errors_by_bucket.items()):
        agg = aggregate(errors)
        n_missed_matches = sum(1 for e in errors if e.n_points_scored == 0)
        print(f"[{mode} | {bucket_name}] n_scored={agg['n']} (+{n_missed_matches} matched-but-unscored)")
        if agg["n"]:
            print(f"  ADE:  {agg['ade_m']:.3f} m")
            print(f"  FDE:  {agg['fde_m']:.3f} m")
            print(f"  Miss rate (FDE > 2.0m): {agg['miss_rate']:.1%}")
        print()


if __name__ == "__main__":
    main()
