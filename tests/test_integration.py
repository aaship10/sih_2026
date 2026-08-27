"""
Integration test: Connects the Costmap, Trajectory Sampler, and Cost Evaluator.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
try:
    matplotlib.use("MacOSX")
except Exception:
    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from interfaces.data_types import EgoState, TrackedObject, DrivableSpace
from costmap.local_costmap import LocalCostMap
from trajectory_generator.polynomial_sampler import PolynomialTrajectorySampler
from trajectory_generator.cost_evaluator import CostEvaluator

def run_integration_test():
    # 1. Setup Ego State & Map
    ego = EgoState(x=0.0, y=0.0, vx=6.0, vy=0.0, yaw=0.0, accel=0.0, steering=0.0, timestamp=0.0)
    costmap_gen = LocalCostMap()
    drivable = DrivableSpace()
    
    # 2. Place an obstacle directly in the ego vehicle's path
    static_objs = [
        TrackedObject(1, "pushcart", (12.0, 0.0), (0, 0), 0, (1.8, 1.2, 1.2), 0.95)
    ]
    
    # 3. Build Costmap
    grid = costmap_gen.build_costmap(drivable, static_objs, [])
    
    # 4. Generate Candidates
    sampler = PolynomialTrajectorySampler(time_horizon=3.0, dt=0.2)
    candidates = sampler.generate_candidates(ego, target_speed=6.0)
    
    # 5. Evaluate and Select Best Path
    evaluator = CostEvaluator(costmap_gen)
    best_path, best_cost = evaluator.evaluate_paths(candidates, grid)
    
    # 6. Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(grid.T, origin="lower", cmap="inferno", extent=[-10, 30, -10, 10])
    
    # Plot all candidates in gray
    for path in candidates:
        xs = [pt.x for pt in path]
        ys = [pt.y for pt in path]
        ax.plot(xs, ys, color='gray', alpha=0.5, linestyle='--')
        
    # Plot best path in bright green
    if best_path:
        xs = [pt.x for pt in best_path]
        ys = [pt.y for pt in best_path]
        ax.plot(xs, ys, color='lime', linewidth=3, label=f"Best Path (Cost: {best_cost:.1f})")
    else:
        ax.set_title("EMERGENCY: No valid paths found!")
        
    ax.plot(0, 0, 'go', markersize=8, label="Ego")
    ax.set_title("M5 Planner Integration: Dynamic Obstacle Avoidance")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    
    output_path = "integration_test_output.png"
    plt.savefig(output_path, dpi=150)
    print(f"Integration visualization saved to {output_path}")
    
    try:
        plt.show(block=True)
    except Exception:
        os.system(f"open {output_path}")

if __name__ == "__main__":
    run_integration_test()