"""
Test script to visualize the generated candidate trajectories and vehicle footprint.
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
import numpy as np
from trajectory_generator.polynomial_sampler import PolynomialTrajectorySampler
from interfaces.data_types import EgoState

def plot_vehicle_footprint(ax, x, y, yaw):
    # 3-circle approximation of an SUV/Sedan
    circle_radius = 1.05
    offsets = [0.8, 2.2, 3.6] # Distance from rear axle
    
    for offset in offsets:
        cx = x + offset * np.cos(yaw)
        cy = y + offset * np.sin(yaw)
        circle = plt.Circle((cx, cy), circle_radius, color='blue', fill=False, linestyle='--')
        ax.add_patch(circle)

def run_planner_test():
    ego = EgoState(x=0.0, y=0.0, vx=8.0, vy=0.0, yaw=0.0, accel=0.0, steering=0.0, timestamp=0.0)
    sampler = PolynomialTrajectorySampler(time_horizon=3.0, dt=0.2)
    
    candidates = sampler.generate_candidates(ego, target_speed=8.0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each candidate path
    colors = ['red', 'orange', 'green', 'orange', 'red']
    for idx, path in enumerate(candidates):
        xs = [pt.x for pt in path]
        ys = [pt.y for pt in path]
        ax.plot(xs, ys, color=colors[idx], label=f'Offset: {sampler.lateral_offsets[idx]}m')
        
    # Plot vehicle footprints at the end of the green (center) path
    end_pt = candidates[2][-1]
    plot_vehicle_footprint(ax, ego.x, ego.y, ego.yaw)      # Start footprint
    plot_vehicle_footprint(ax, end_pt.x, end_pt.y, end_pt.yaw) # End footprint
    
    ax.set_title("Quintic Polynomial Candidate Trajectories & 3-Circle Footprint")
    ax.set_xlabel("X (m) - Forward")
    ax.set_ylabel("Y (m) - Lateral")
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    
    output_path = "planner_test_output.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved visualization to {output_path}")
    
    try:
        plt.show(block=True)
    except Exception:
        os.system(f"open {output_path}")

if __name__ == "__main__":
    run_planner_test()