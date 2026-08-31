"""
run_dummy_test_cases.py
========================
Runs m4_dummy_test_cases.ALL_TEST_CASES against interface.predict().

No adapter needed anymore -- predict() now normalizes ego_state
internally (interface._normalize_ego_state()), so this file's
[vx, vy]-vector ego_state passes straight through unchanged.

Lives in M4_Pipeline/test/ alongside the dummy test cases.

RUN WITH (from anywhere):
    python run_dummy_test_cases.py
"""

import json
import os
import sys

# Makes this importable regardless of the working directory it's run
# from -- interface.py and m4_dummy_test_cases.py both live right next
# to this file.
M4_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, M4_DIR)

from interface import predict
from m4_dummy_test_cases import ALL_TEST_CASES


def run():
    for case in ALL_TEST_CASES:
        print("=" * 70)
        print(case["name"])
        print("=" * 70)

        # history + the current frame, oldest -> newest, as buffer.py expects
        track_history = case["history"] + [case["tracked_object"]]

        # NOTE on TEST 11 ("zero history"): case["history"] is [], but
        # track_history above always appends tracked_object, so predict()
        # actually still gets exactly 1 frame here -- it does NOT raise.
        # heading/acceleration just fall back to their single-frame
        # defaults (derive.py needs >=2 frames for a real finite-
        # difference estimate). True zero-history (empty track_history)
        # is only what main.py's own MIN_HISTORY_TO_PREDICT guard skips
        # in the live loop -- this test file never produces that case.
        try:
            pred = predict(
                track_id=case["tracked_object"]["track_id"],
                cls=case["tracked_object"]["class"],
                track_history=track_history,
                ego_state=case["ego_state"],
                confidence=case["tracked_object"]["confidence"],
            )
        except ValueError as e:
            print(f"predict() raised: {e}")
            print()
            continue

        summary = {
            "risk_level": pred["risk_level"],
            "risk_score": pred["risk_score"],
            "collision_probability": pred["collision_probability"],
            "minimum_ttc": pred["minimum_ttc"],
            "time_to_conflict": pred["time_to_conflict"],
            "min_predicted_distance_to_ego": pred["min_predicted_distance_to_ego"],
            "confidence": pred["confidence"],
        }
        print(json.dumps(summary, indent=2))
        print()


if __name__ == "__main__":
    run()