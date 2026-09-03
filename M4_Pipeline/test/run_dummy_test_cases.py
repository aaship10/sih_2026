"""Validate the M4 -> M5 prediction fields against dummy test cases."""

import json
import os
import sys

M4_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, M4_DIR)

from interface import predict
from m4_dummy_test_cases import ALL_TEST_CASES


def run():
    required = {
        "track_id", "class", "confidence", "position", "velocity",
        "size", "trajectories",
    }
    forbidden = {
        "minimum_ttc", "time_to_conflict", "collision_probability",
        "min_predicted_distance_to_ego", "risk_score", "risk_level",
    }

    for case in ALL_TEST_CASES:
        track = case["tracked_object"]
        history = case["history"] + [track]
        pred = predict(
            track_id=track["track_id"],
            cls=track["class"],
            track_history=history,
            confidence=track["confidence"],
        )

        missing = required - pred.keys()
        present_forbidden = forbidden & pred.keys()
        assert not missing, f"{case['name']}: missing {missing}"
        assert not present_forbidden, f"{case['name']}: obsolete fields {present_forbidden}"

        assert len(pred["trajectories"]) == 3
        assert abs(sum(b["probability"] for b in pred["trajectories"]) - 1.0) < 0.01

    print(json.dumps({"status": "PASS", "cases": len(ALL_TEST_CASES)}, indent=2))


if __name__ == "__main__":
    run()
