from interfaces.m4_adapter import M4Adapter


def main():
    sample_message = {
        "timestamp": 123.456,

        "ego": {
            "speed_mps": 6.0,
            "position": [0.0, 0.0],
            "velocity": [6.0, 0.0],
            "yaw_deg": 0.0
        },
        
        "obstacles": [
            {
                "track_id": 101,
                "class": "cattle",
                "confidence": 0.92,
                "pos_x": 15.5,
                "pos_y": 2.0,
                "vel_x": 0.0,
                "vel_y": -0.5,
                "width": 1.2,
                "length": 2.0,
                "trajectories": [
                    {
                        "mode": "straight",
                        "probability": 0.7,
                        "points": [
                            {"x": 15.5, "y": 1.9, "t": 0.2},
                            {"x": 15.5, "y": 1.8, "t": 0.4},
                        ],
                    }
                ],
            }
        ],
    }

    predictions = M4Adapter.parse_predictions(sample_message)

    print("Number of predictions:", len(predictions))

    for pred in predictions:
        print("Track ID:", pred.track_id)

        for traj in pred.trajectories:
            print("Probability:", traj.probability)
            print("Waypoints:", traj.waypoints)

        print("Risk:", pred.risk_level)
        print("Collision probability:", pred.collision_probability)

    ego = M4Adapter.parse_ego_state(sample_message)

    print("Ego:", ego)


if __name__ == "__main__":
    main()