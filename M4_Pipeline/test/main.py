"""Standalone M4 prediction test against the dummy M3 stream."""

import json
import os
import sys

M4_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, M4_DIR)

from buffer import TrackHistoryBuffer
from dummy_m3_stream import generate_dummy_tracked_objects
from interface import predict

MIN_HISTORY_TO_PREDICT = 2


def run():
    history = TrackHistoryBuffer(max_len=15)
    last_predictions = {}

    for t, tracked_objects in generate_dummy_tracked_objects(n_frames=15, dt=0.2):
        history.update(tracked_objects)
        frame_predictions = []

        for obj in tracked_objects:
            track_hist = history.get(obj["track_id"])
            if len(track_hist) < MIN_HISTORY_TO_PREDICT:
                continue
            pred = predict(
                track_id=obj["track_id"],
                cls=obj["class"],
                track_history=track_hist,
                confidence=obj["confidence"],
            )
            frame_predictions.append(pred)
            last_predictions[obj["track_id"]] = pred

        print(f"{t:5.2f} | predictions={len(frame_predictions)}")

    print("\nFinal M4 predictions:\n")
    for tid, pred in last_predictions.items():
        print(f"--- track {tid} ---")
        print(json.dumps(pred, indent=2))


if __name__ == "__main__":
    run()
