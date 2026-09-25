"""Prediction metrics (section 30: ADE, FDE, Miss Rate), scored against
CARLA ground truth (evaluation/ground_truth.py), never against M4's own
predictions -- the whole point is to check M4 against reality.

(Risk metrics -- collision-prediction precision/recall, TTC error -- are
not implemented here: M4 no longer computes risk/TTC/collision-probability,
see README.md's "Status" section. If a later stage adds that back, this is
the natural place to re-add the matching evaluation.)

Track identity problem: M4's track_ids don't correspond to CARLA actor ids
(M3 doesn't carry them through -- same limitation M3_Pipeline/evaluation.py
already documents). Each prediction is matched to the ground-truth actor
nearest to it (within MATCH_DIST_THRESHOLD_M) AT THE PREDICTION's OWN start
frame, then that specific actor_id is followed forward through ground truth
to score the prediction's future points -- so a mid-horizon M3 track_id
swap (M3 README 7B.7: known, unsolved) does not, by itself, break a single
prediction's evaluation, only cross-frame trajectory continuity.

Future ground-truth lookup: `timestamp(frame_id) = anchor_timestamp +
(frame_id - anchor_frame_id) * DT` is EXACT, not approximate, given any one
KNOWN (frame_id, timestamp) pair -- CARLA's synchronous-mode fixed_delta_
seconds means SIMULATION time advances by exactly DT per tick regardless of
real (wall-clock) pacing between packets (M3 README 7B: "packet spacing is
irregular" refers to WALL-CLOCK arrival, not simulation timestamps). This
lets ADE/FDE look up "where was this actor at sim-time t0+k*step" from
ground truth frame numbers alone, without needing an exact timestamp in the
stdout log.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

MATCH_DIST_THRESHOLD_M = 5.0
CARLA_DT_S = 0.05  # scenario3.py / M1_Pipeline/common.py fixed_delta_seconds


def frame_id_for_time(anchor_frame_id: int, anchor_timestamp: float, t: float, dt: float = CARLA_DT_S) -> int:
    return anchor_frame_id + round((t - anchor_timestamp) / dt)


def match_actor(x: float, y: float, gt_actors: list[dict]) -> dict | None:
    best, best_d = None, float("inf")
    for a in gt_actors:
        d = math.hypot(a["x"] - x, a["y"] - y)
        if d < best_d:
            best_d, best = d, a
    return best if best_d <= MATCH_DIST_THRESHOLD_M else None


def match_actors_unique(candidates: list[tuple[object, float, float]], gt_actors: list[dict]) -> dict[object, int]:
    """Match many (key, x, y) candidates against gt_actors from the SAME
    frame, one ground-truth actor to at most one candidate.

    match_actor() alone is only safe when candidates are sparse relative to
    MATCH_DIST_THRESHOLD_M. In a dense frame (checked on a real recorded
    run: ~28 simultaneous actors, ~2 OTHER actors within the 5m threshold
    of each one on average) calling match_actor() independently per
    candidate lets two different tracks both claim the same nearest actor
    -- confirmed happening in practice (two different M4 track_ids matched
    to the same CARLA actor_id in the same frame), which silently scores at
    least one of them against the wrong object's future ground truth.

    Greedy nearest-pair-first assignment: sort every (candidate, actor)
    pair within threshold by distance, then walk it claiming both sides
    exactly once. Not a global optimum (that's an assignment problem), but
    removes the specific duplicate-claim failure mode, is O(n*m log(n*m))
    on frame-sized n/m, and always prefers the closest available pair
    first."""
    pairs = []
    for ci, (_, x, y) in enumerate(candidates):
        for ai, a in enumerate(gt_actors):
            d = math.hypot(a["x"] - x, a["y"] - y)
            if d <= MATCH_DIST_THRESHOLD_M:
                pairs.append((d, ci, ai))
    pairs.sort(key=lambda p: p[0])

    claimed_candidates: set[int] = set()
    claimed_actors: set[int] = set()
    result: dict[object, int] = {}
    for _, ci, ai in pairs:
        if ci in claimed_candidates or ai in claimed_actors:
            continue
        claimed_candidates.add(ci)
        claimed_actors.add(ai)
        result[candidates[ci][0]] = gt_actors[ai]["actor_id"]
    return result


@dataclass
class TrajectoryError:
    track_id: int
    actor_id: int
    ade_m: float               # average displacement error over the whole horizon
    fde_m: float                # final displacement error (last predicted point only)
    n_points_scored: int
    missed: bool                 # True if the actor could not be found in ground truth at ANY future frame checked


def evaluate_trajectory(predicted_points: list[list[float]], times_s: list[float],
                         start_frame_id: int, start_timestamp: float,
                         gt_frames: dict[int, dict], track_id: int, miss_threshold_m: float = 2.0) -> TrajectoryError | None:
    """Score ONE predicted trajectory (predicted_points[i] is where the
    object was predicted to be at times_s[i] seconds after the prediction
    was made) against ground truth. Returns None if no ground-truth actor
    can be matched at the prediction's own start frame at all.

    Matches independently via match_actor() -- fine for a sparse frame, but
    see match_actors_unique()'s docstring for the dense-frame duplicate-
    claim failure mode this does NOT protect against. Prefer
    evaluate_matched_trajectory() (pass an actor_id resolved once per
    OBJECT per frame via match_actors_unique(), shared across that
    object's nominal/stop/lateral modes) when scoring a full frame of
    predictions at once."""
    start_frame = gt_frames.get(start_frame_id)
    if start_frame is None:
        return None
    matched = match_actor(predicted_points[0][0], predicted_points[0][1], start_frame["actors"])
    if matched is None:
        return None
    actor_id = matched["actor_id"]
    return evaluate_matched_trajectory(predicted_points, times_s, start_frame_id, start_timestamp,
                                        gt_frames, track_id, actor_id, miss_threshold_m)


def evaluate_matched_trajectory(predicted_points: list[list[float]], times_s: list[float],
                                 start_frame_id: int, start_timestamp: float,
                                 gt_frames: dict[int, dict], track_id: int, actor_id: int,
                                 miss_threshold_m: float = 2.0) -> TrajectoryError:
    """Same scoring as evaluate_trajectory(), but the actor_id is already
    resolved (see match_actors_unique()) instead of matched here."""
    errors = []
    for t, (px, py) in zip(times_s, predicted_points):
        fid = frame_id_for_time(start_frame_id, start_timestamp, start_timestamp + t)
        frame = gt_frames.get(fid)
        if frame is None:
            continue
        actual = next((a for a in frame["actors"] if a["actor_id"] == actor_id), None)
        if actual is None:
            continue
        errors.append(math.hypot(actual["x"] - px, actual["y"] - py))

    if not errors:
        return TrajectoryError(track_id, actor_id, ade_m=float("nan"), fde_m=float("nan"), n_points_scored=0, missed=True)

    ade = sum(errors) / len(errors)
    fde = errors[-1]
    return TrajectoryError(track_id, actor_id, ade_m=ade, fde_m=fde, n_points_scored=len(errors),
                            missed=fde > miss_threshold_m)


def aggregate(errors: list[TrajectoryError]) -> dict:
    scored = [e for e in errors if e.n_points_scored > 0]
    if not scored:
        return {"n": 0, "ade_m": None, "fde_m": None, "miss_rate": None}
    return {
        "n": len(scored),
        "ade_m": sum(e.ade_m for e in scored) / len(scored),
        "fde_m": sum(e.fde_m for e in scored) / len(scored),
        "miss_rate": sum(1 for e in scored if e.missed) / len(scored),
    }
