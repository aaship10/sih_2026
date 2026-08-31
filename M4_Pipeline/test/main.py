"""
main.py
=======
Runs M4 standalone against the dummy M3 stream (dummy_m3_stream.py) +
dummy ego (dummy_ego.py), printing predictions per frame -- mirrors
M3's own main.py so the two are easy to run side by side.

TO SWITCH TO REAL M3 LATER:
    Replace the `from dummy_m3_stream import generate_dummy_tracked_objects`
    loop with a loop over M3's MultiObjectTracker.step() output (see
    M3_Pipeline/main.py for the exact pattern) -- everything below the
    `history.update(tracked_objects)` line is unchanged, because
    buffer.py/interface.py only care about the FORMAT M3 already
    guarantees via to_output_dict(), not where it came from.

    Replace dummy_ego.generate_dummy_ego() with M1/M5's real ego_state
    once available -- interface.predict() only needs the same
    {position, velocity, heading, timestamp} shape.

RUN WITH:
    python main.py
"""

import json

from buffer import TrackHistoryBuffer
from dummy_m3_stream import generate_dummy_tracked_objects
from dummy_ego import generate_dummy_ego
from interface import predict

MIN_HISTORY_TO_PREDICT = 2  # need at least 2 frames to derive heading/accel meaningfully


def run():
    history = TrackHistoryBuffer(max_len=15)

    print(f"{'t':>5} | predictions (id: class -> risk_level [score] | TTC/conflict)")
    print("-" * 90)

    last_predictions = {}
    t0 = None
    for t, tracked_objects in generate_dummy_tracked_objects(n_frames=15, dt=0.2):
        if t0 is None:
            t0 = t  # dummy_m3_stream's clock starts at 12.0, not 0.0 --
                     # dummy_ego needs elapsed time since scenario start,
                     # not M3's absolute simulation timestamp, or the ego
                     # starts the "scenario" already 96m down the road.
        history.update(tracked_objects)
        ego_state = generate_dummy_ego(t - t0)
        ego_state["timestamp"] = round(t, 2)  # keep the real timestamp for downstream consumers

        frame_predictions = []
        for obj in tracked_objects:
            track_hist = history.get(obj["track_id"])
            if len(track_hist) < MIN_HISTORY_TO_PREDICT:
                continue  # not enough history yet -- skip this track this frame

            pred = predict(
                track_id=obj["track_id"],
                cls=obj["class"],
                track_history=track_hist,
                ego_state=ego_state,
                confidence=obj["confidence"],
            )
            frame_predictions.append(pred)
            last_predictions[obj["track_id"]] = pred

        summary = ", ".join(
            f"#{p['track_id']}: {p['class']} -> {p['risk_level']} [{p['risk_score']}] "
            f"| TTC={p['minimum_ttc']} conflict={p['time_to_conflict']}"
            for p in frame_predictions
        )
        print(f"{t:5.2f} | {summary}")

    print("-" * 90)
    print("\nFinal predictions (this is the M4 -> M5 interface, one entry per track):\n")
    for tid, pred in last_predictions.items():
        # drop the bulky per-branch point lists for a readable console dump;
        # the real dict handed to M5 includes full trajectory points
        printable = {k: v for k, v in pred.items() if k not in ("trajectories", "debug")}
        printable["trajectories"] = [
            {"mode": b["mode"], "probability": b["probability"], "n_points": len(b["points"])}
            for b in pred["trajectories"]
        ]
        print(f"--- track {tid} ---")
        print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    run()