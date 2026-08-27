"""
Mock Scenario Generator for testing M5 path planning independently.
Generates Bird's-Eye-View (BEV) cost maps for all 5 Indian road scenarios.
"""

import sys
import os
import numpy as np

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
# Try interactive macOS backend, fallback to Agg
try:
    matplotlib.use("MacOSX")
except Exception:
    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from interfaces.data_types import EgoState, TrackedObject, DynamicPrediction, PredictedTrajectory, DrivableSpace
from costmap.local_costmap import LocalCostMap


def get_scenario_data(scenario_id: int):
    ego = EgoState(x=0.0, y=0.0, vx=6.0, vy=0.0, yaw=0.0, accel=0.0, steering=0.0, timestamp=0.0)
    drivable = DrivableSpace()
    static_objs = []
    dynamic_preds = []

    if scenario_id == 1:
        # Scenario 1: Unmarked Village Road (Irregular boundaries + static rock)
        drivable.occupancy_mask[:, :20] = 0   # Left unpaved edge
        drivable.occupancy_mask[:, 80:] = 0   # Right unpaved edge
        static_objs.append(TrackedObject(1, "rock", (15.0, 0.5), (0, 0), 0, (0.8, 0.8, 0.5), 0.95))

    elif scenario_id == 2:
        # Scenario 2: Unsignalized Urban Intersection (Auto crossing path)
        traj_auto = [(12.0, 6.0 - 1.5 * t, t) for t in np.linspace(0, 3, 10)]
        dynamic_preds.append(DynamicPrediction(
            track_id=10,
            trajectories=[PredictedTrajectory(waypoints=traj_auto, probability=0.85)],
            risk_level="HIGH",
            collision_probability=0.75
        ))

    elif scenario_id == 3:
        # Scenario 3: Highway Slow-Vehicle Merge (Slow tractor ahead)
        static_objs.append(TrackedObject(2, "tractor", (18.0, 0.0), (2.0, 0), 0, (4.0, 2.0, 2.5), 0.99))

    elif scenario_id == 4:
        # Scenario 4: Dense Market (Static pushcarts + pedestrian crossing)
        static_objs.append(TrackedObject(3, "pushcart", (8.0, -1.8), (0, 0), 0, (1.8, 1.0, 1.2), 0.9))
        static_objs.append(TrackedObject(4, "pushcart", (14.0, 1.5), (0, 0), 0, (1.8, 1.0, 1.2), 0.9))
        traj_ped = [(10.0, -3.0 + 1.2 * t, t) for t in np.linspace(0, 3, 10)]
        dynamic_preds.append(DynamicPrediction(
            track_id=12,
            trajectories=[PredictedTrajectory(waypoints=traj_ped, probability=0.9)],
            risk_level="MEDIUM",
            collision_probability=0.55
        ))

    elif scenario_id == 5:
        # Scenario 5: Sudden Cattle Crossing (High collision risk)
        traj_cattle = [(10.0, 4.0 - 2.0 * t, t) for t in np.linspace(0, 2, 8)]
        dynamic_preds.append(DynamicPrediction(
            track_id=20,
            trajectories=[PredictedTrajectory(waypoints=traj_cattle, probability=0.95)],
            risk_level="CRITICAL",
            collision_probability=0.90
        ))

    return ego, drivable, static_objs, dynamic_preds


def run_visualizer():
    costmap_gen = LocalCostMap()
    scenarios = [
        "1. Village Road", "2. Unsignalized Intersection",
        "3. Highway Slow Vehicle", "4. Dense Market", "5. Cattle Crossing"
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for idx, name in enumerate(scenarios):
        ego, drivable, statics, dynamics = get_scenario_data(idx + 1)
        grid = costmap_gen.build_costmap(drivable, statics, dynamics, time_horizon_step=1.0)

        ax = axes[idx]
        ax.imshow(grid.T, origin="lower", cmap="inferno", extent=[-10, 30, -10, 10])
        ax.plot(0, 0, 'go', markersize=8, label="Ego")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("X (m)")
        if idx == 0:
            ax.set_ylabel("Y (m)")

    plt.tight_layout()
    
    # Save the output image locally and show
    output_path = "mock_scenarios_output.png"
    plt.savefig(output_path, dpi=150)
    print(f"Visualization saved to {output_path}")
    
    try:
        plt.show(block=True)
    except Exception:
        # If GUI backend fails on macOS terminal, open the saved image directly
        os.system(f"open {output_path}")


if __name__ == "__main__":
    run_visualizer()