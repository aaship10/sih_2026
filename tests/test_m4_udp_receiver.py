from interfaces.m4_adapter import M4Adapter


def main():
    predictions = M4Adapter.receive_predictions()

    print("\n--- Parsed M4 Predictions ---")

    for pred in predictions:
        print(f"Track ID: {pred.track_id}")

        for traj in pred.trajectories:
            print(f"  Probability: {traj.probability}")
            print(f"  Waypoints: {traj.waypoints}")

        print(f"  Risk: {pred.risk_level}")
        print(f"  Collision Probability: {pred.collision_probability}")


if __name__ == "__main__":
    main()