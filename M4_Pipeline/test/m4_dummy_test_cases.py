"""
m4_dummy_test_cases.py
=======================
Just the dummy INPUTS for M4 -- fake versions of what M3 will really
hand over, so M4 can run their own predict/risk code against these and
look at the results themselves. No built-in checker, no "correct
answer" logic -- just clean test data.

Each test case is a dict:
    {
        "name": short description,
        "tracked_object": the CURRENT frame's object -- exactly M3's
                           per-object output format,
        "history": a list of that SAME object's past few frames
                   (oldest -> newest) -- also exactly M3's format,
        "ego_state": the car's own current position/velocity
    }

HOW TO USE THIS:
    from m4_dummy_test_cases import ALL_TEST_CASES

    for case in ALL_TEST_CASES:
        result = my_predict_function(case["tracked_object"], case["history"], case["ego_state"])
        print(case["name"])
        print(result)

Coordinate frame (matches M3's output): x = forward (meters),
y = left/right (meters, positive = left), all in the ego (car) frame.
"""

EGO_STATE = {
    "position": [0.0, 0.0],
    "velocity": [12.0, 0.0],
    "heading": 0.0,
    "timestamp": 10.0,
}


# ---------------------------------------------------------------------
# TEST 1: Constant straight-moving car (adjacent lane, no real danger)
# ---------------------------------------------------------------------
TEST_1_CONSTANT_VELOCITY_CAR = {
    "name": "TEST 1: Constant straight-moving car",
    "tracked_object": {
        "track_id": 101, "class": "car",
        "position": [40.0, -3.5, 0.0], "velocity": [8.0, 0.0, 0.0],
        "heading": 0.0, "size": [4.5, 1.8, 1.5],
        "confidence": 0.9, "timestamp": 10.8,
    },
    "history": [
        {"timestamp": 10.0, "position": [38.4, -3.5], "velocity": [8.0, 0.0]},
        {"timestamp": 10.2, "position": [39.04, -3.5], "velocity": [8.0, 0.0]},
        {"timestamp": 10.4, "position": [39.44, -3.5], "velocity": [8.0, 0.0]},
        {"timestamp": 10.6, "position": [39.76, -3.5], "velocity": [8.0, 0.0]},
        {"timestamp": 10.8, "position": [40.0, -3.5], "velocity": [8.0, 0.0]},
    ],
    "ego_state": EGO_STATE,
}


# ---------------------------------------------------------------------
# TEST 2: Accelerating vehicle
# ---------------------------------------------------------------------
TEST_2_ACCELERATING_VEHICLE = {
    "name": "TEST 2: Accelerating vehicle",
    "tracked_object": {
        "track_id": 102, "class": "car",
        "position": [30.6, -3.5, 0.0], "velocity": [6.6, 0.0, 0.0],
        "heading": 0.0, "size": [4.5, 1.8, 1.5],
        "confidence": 0.9, "timestamp": 10.8,
    },
    "history": [
        {"timestamp": 10.0, "position": [25.0, -3.5], "velocity": [5.0, 0.0]},
        {"timestamp": 10.2, "position": [26.04, -3.5], "velocity": [5.4, 0.0]},
        {"timestamp": 10.4, "position": [27.16, -3.5], "velocity": [5.8, 0.0]},
        {"timestamp": 10.6, "position": [28.36, -3.5], "velocity": [6.2, 0.0]},
        {"timestamp": 10.8, "position": [29.64, -3.5], "velocity": [6.6, 0.0]},
    ],
    "ego_state": EGO_STATE,
}


# ---------------------------------------------------------------------
# TEST 3: Decelerating vehicle, directly in ego's own lane
# ---------------------------------------------------------------------
TEST_3_DECELERATING_VEHICLE = {
    "name": "TEST 3: Decelerating vehicle, same lane as ego",
    "tracked_object": {
        "track_id": 103, "class": "car",
        "position": [27.04, 0.0, 0.0], "velocity": [7.6, 0.0, 0.0],
        "heading": 0.0, "size": [4.5, 1.8, 1.5],
        "confidence": 0.9, "timestamp": 10.8,
    },
    "history": [
        {"timestamp": 10.0, "position": [20.0, 0.0], "velocity": [10.0, 0.0]},
        {"timestamp": 10.2, "position": [21.94, 0.0], "velocity": [9.4, 0.0]},
        {"timestamp": 10.4, "position": [23.52, 0.0], "velocity": [8.8, 0.0]},
        {"timestamp": 10.6, "position": [24.74, 0.0], "velocity": [8.2, 0.0]},
        {"timestamp": 10.8, "position": [25.6, 0.0], "velocity": [7.6, 0.0]},
    ],
    "ego_state": EGO_STATE,
}


# ---------------------------------------------------------------------
# TEST 4: Pedestrian crossing (clearly angling into the road)
# ---------------------------------------------------------------------
TEST_4_PEDESTRIAN_CROSSING = {
    "name": "TEST 4: Pedestrian crossing",
    "tracked_object": {
        "track_id": 104, "class": "pedestrian",
        "position": [15.2, 1.8, 0.0], "velocity": [0.5, -3.0, 0.0],
        "heading": -1.4, "size": [0.5, 0.5, 1.7],
        "confidence": 0.9, "timestamp": 10.8,
    },
    "history": [
        {"timestamp": 10.0, "position": [15.0, 4.0], "velocity": [0.0, -2.5]},
        {"timestamp": 10.2, "position": [15.0, 3.5], "velocity": [0.0, -2.5]},
        {"timestamp": 10.4, "position": [15.05, 3.0], "velocity": [0.2, -2.6]},
        {"timestamp": 10.6, "position": [15.1, 2.4], "velocity": [0.3, -2.8]},
        {"timestamp": 10.8, "position": [15.2, 1.8], "velocity": [0.5, -3.0]},
    ],
    "ego_state": EGO_STATE,
}


# ---------------------------------------------------------------------
# TEST 5: Motorcycle lateral movement (weaving through traffic)
# ---------------------------------------------------------------------
TEST_5_MOTORCYCLE_LATERAL = {
    "name": "TEST 5: Motorcycle lateral movement",
    "tracked_object": {
        "track_id": 105, "class": "motorcycle",
        "position": [22.0, -2.6, 0.0], "velocity": [15.0, 3.0, 0.0],
        "heading": 0.2, "size": [2.0, 0.8, 1.3],
        "confidence": 0.85, "timestamp": 10.8,
    },
    "history": [
        {"timestamp": 10.0, "position": [10.0, -5.0], "velocity": [15.0, 3.0]},
        {"timestamp": 10.2, "position": [13.0, -4.4], "velocity": [15.0, 3.0]},
        {"timestamp": 10.4, "position": [16.0, -3.8], "velocity": [15.0, 3.0]},
        {"timestamp": 10.6, "position": [19.0, -3.2], "velocity": [15.0, 3.0]},
        {"timestamp": 10.8, "position": [22.0, -2.6], "velocity": [15.0, 3.0]},
    ],
    "ego_state": EGO_STATE,
}


# ---------------------------------------------------------------------
# TEST 6: Auto-rickshaw sudden direction change
# ---------------------------------------------------------------------
TEST_6_AUTORICKSHAW_DIRECTION_CHANGE = {
    "name": "TEST 6: Auto-rickshaw sudden direction change",
    "tracked_object": {
        "track_id": 106, "class": "auto_rickshaw",
        "position": [28.8, -2.0, 0.0], "velocity": [8.5, 4.5, 0.0],
        "heading": 0.48, "size": [2.6, 1.4, 1.8],
        "confidence": 0.87, "timestamp": 11.0,
    },
    "history": [
        {"timestamp": 10.0, "position": [20.0, -4.0], "velocity": [10.0, 0.0]},
        {"timestamp": 10.2, "position": [22.0, -4.0], "velocity": [10.0, 0.0]},
        {"timestamp": 10.4, "position": [24.0, -4.0], "velocity": [10.0, 0.0]},
        {"timestamp": 10.6, "position": [25.8, -3.7], "velocity": [9.5, 2.0]},
        {"timestamp": 10.8, "position": [27.4, -3.0], "velocity": [9.0, 3.5]},
        {"timestamp": 11.0, "position": [28.8, -2.0], "velocity": [8.5, 4.5]},
    ],
    "ego_state": EGO_STATE,
}


# ---------------------------------------------------------------------
# TEST 7: Animal changing direction unpredictably (cattle crossing)
# ---------------------------------------------------------------------
TEST_7_ANIMAL_ERRATIC = {
    "name": "TEST 7: Animal changing direction (cattle)",
    "tracked_object": {
        "track_id": 107, "class": "animal",
        "position": [17.6, 1.8, 0.0], "velocity": [-1.0, -4.2, 0.0],
        "heading": -1.8, "size": [2.2, 0.9, 1.4],
        "confidence": 0.75, "timestamp": 10.8,
    },
    "history": [
        {"timestamp": 10.0, "position": [18.0, 5.0], "velocity": [-2.5, -4.0]},
        {"timestamp": 10.2, "position": [17.5, 4.2], "velocity": [1.5, -4.5]},
        {"timestamp": 10.4, "position": [17.8, 3.3], "velocity": [-3.0, -3.5]},
        {"timestamp": 10.6, "position": [17.2, 2.6], "velocity": [2.0, -4.0]},
        {"timestamp": 10.8, "position": [17.6, 1.8], "velocity": [-1.0, -4.2]},
    ],
    "ego_state": EGO_STATE,
}


# ---------------------------------------------------------------------
# TEST 8: Vehicle merging into ego's lane, close ahead
# ---------------------------------------------------------------------
TEST_8_VEHICLE_MERGING = {
    "name": "TEST 8: Vehicle merging",
    "tracked_object": {
        "track_id": 108, "class": "car",
        "position": [11.0, -1.8, 0.0], "velocity": [12.0, 2.0, 0.0],
        "heading": 0.165, "size": [4.5, 1.8, 1.5],
        "confidence": 0.9, "timestamp": 11.0,
    },
    "history": [
        {"timestamp": 10.0, "position": [9.0, -3.8], "velocity": [12.0, 2.0]},
        {"timestamp": 10.2, "position": [9.4, -3.4], "velocity": [12.0, 2.0]},
        {"timestamp": 10.4, "position": [9.8, -3.0], "velocity": [12.0, 2.0]},
        {"timestamp": 10.6, "position": [10.2, -2.6], "velocity": [12.0, 2.0]},
        {"timestamp": 10.8, "position": [10.6, -2.2], "velocity": [12.0, 2.0]},
        {"timestamp": 11.0, "position": [11.0, -1.8], "velocity": [12.0, 2.0]},
    ],
    "ego_state": EGO_STATE,
}


# ---------------------------------------------------------------------
# TEST 9: Multiple vehicles at an intersection (2 separate objects)
# ---------------------------------------------------------------------
TEST_9_INTERSECTION_AGENT_A = {
    "name": "TEST 9a: Intersection -- car crossing top to bottom",
    "tracked_object": {
        "track_id": 201, "class": "car",
        "position": [5.0, 2.0, 0.0], "velocity": [0.0, -10.0, 0.0],
        "heading": -1.57, "size": [4.5, 1.8, 1.5],
        "confidence": 0.9, "timestamp": 10.6,
    },
    "history": [
        {"timestamp": 10.0, "position": [5.0, 8.0], "velocity": [0.0, -10.0]},
        {"timestamp": 10.2, "position": [5.0, 6.0], "velocity": [0.0, -10.0]},
        {"timestamp": 10.4, "position": [5.0, 4.0], "velocity": [0.0, -10.0]},
        {"timestamp": 10.6, "position": [5.0, 2.0], "velocity": [0.0, -10.0]},
    ],
    "ego_state": EGO_STATE,
}

TEST_9_INTERSECTION_AGENT_B = {
    "name": "TEST 9b: Intersection -- motorcycle crossing right to left",
    "tracked_object": {
        "track_id": 202, "class": "motorcycle",
        "position": [9.0, -6.0, 0.0], "velocity": [10.0, 0.0, 0.0],
        "heading": 0.0, "size": [2.0, 0.8, 1.3],
        "confidence": 0.87, "timestamp": 10.6,
    },
    "history": [
        {"timestamp": 10.0, "position": [3.0, -6.0], "velocity": [10.0, 0.0]},
        {"timestamp": 10.2, "position": [5.0, -6.0], "velocity": [10.0, 0.0]},
        {"timestamp": 10.4, "position": [7.0, -6.0], "velocity": [10.0, 0.0]},
        {"timestamp": 10.6, "position": [9.0, -6.0], "velocity": [10.0, 0.0]},
    ],
    "ego_state": EGO_STATE,
}


# ---------------------------------------------------------------------
# TEST 10: Sudden obstacle appearance -- only 1 history sample so far
# ---------------------------------------------------------------------
TEST_10_SUDDEN_APPEARANCE = {
    "name": "TEST 10: Sudden appearance (minimal history)",
    "tracked_object": {
        "track_id": 301, "class": "animal",
        "position": [12.0, -1.0, 0.0], "velocity": [0.0, 0.0, 0.0],
        "heading": 0.0, "size": [2.0, 0.8, 1.3],
        "confidence": 0.6, "timestamp": 10.0,
    },
    "history": [
        {"timestamp": 10.0, "position": [12.0, -1.0], "velocity": [0.0, 0.0]},
    ],
    "ego_state": EGO_STATE,
}


# ---------------------------------------------------------------------
# TEST 11: Zero history at all (extreme edge case -- brand new track)
# ---------------------------------------------------------------------
TEST_11_EMPTY_HISTORY = {
    "name": "TEST 11: Zero history samples",
    "tracked_object": {
        "track_id": 302, "class": "pedestrian",
        "position": [10.0, 1.0, 0.0], "velocity": [0.0, 0.0, 0.0],
        "heading": 0.0, "size": [0.5, 0.5, 1.7],
        "confidence": 0.5, "timestamp": 10.0,
    },
    "history": [],
    "ego_state": EGO_STATE,
}


# Every single-object test case, in one list, for easy looping
ALL_TEST_CASES = [
    TEST_1_CONSTANT_VELOCITY_CAR,
    TEST_2_ACCELERATING_VEHICLE,
    TEST_3_DECELERATING_VEHICLE,
    TEST_4_PEDESTRIAN_CROSSING,
    TEST_5_MOTORCYCLE_LATERAL,
    TEST_6_AUTORICKSHAW_DIRECTION_CHANGE,
    TEST_7_ANIMAL_ERRATIC,
    TEST_8_VEHICLE_MERGING,
    TEST_9_INTERSECTION_AGENT_A,
    TEST_9_INTERSECTION_AGENT_B,
    TEST_10_SUDDEN_APPEARANCE,
    TEST_11_EMPTY_HISTORY,
]


if __name__ == "__main__":
    # Just prints every test case so you can see exactly what's inside --
    # doesn't run any prediction code, since this file has none.
    import json
    for case in ALL_TEST_CASES:
        print("=" * 70)
        print(case["name"])
        print("=" * 70)
        print(json.dumps(case, indent=2))
        print()