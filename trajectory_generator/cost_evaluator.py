"""
Evaluates candidate trajectories against the local cost map.
Performs 3-circle collision checking and calculates trajectory cost.
"""

import numpy as np
from typing import List, Tuple, Optional
from interfaces.data_types import TrajectoryPoint
from costmap.local_costmap import LocalCostMap

class CostEvaluator:
    def __init__(self, costmap_gen: LocalCostMap, weights: dict = None):
        self.costmap_gen = costmap_gen
        # w1: Risk, w2: Smoothness, w3: Lateral deviation
        self.weights = weights or {'risk': 1.5, 'smoothness': 15.0, 'centerline': 5.0}
        self.circle_offsets = [0.8, 2.2, 3.6]  # 3-circle vehicle footprint
        self.collision_threshold = 250         # 255 is solid obstacle/boundary

    def evaluate_paths(self, candidates: List[List[TrajectoryPoint]], costmap_grid: np.ndarray) -> Tuple[Optional[List[TrajectoryPoint]], float]:
        best_path = None
        min_cost = float('inf')

        for path in candidates:
            path_cost = 0.0
            is_valid = True

            for idx, pt in enumerate(path):
                # 1. 3-Circle Collision Check
                for offset in self.circle_offsets:
                    cx = pt.x + offset * np.cos(pt.yaw)
                    cy = pt.y + offset * np.sin(pt.yaw)
                    gx, gy = self.costmap_gen.world_to_grid(cx, cy)

                    if 0 <= gx < self.costmap_gen.nx and 0 <= gy < self.costmap_gen.ny:
                        cell_cost = costmap_grid[gx, gy]
                        if cell_cost >= self.collision_threshold:
                            is_valid = False
                            break
                        # Accumulate risk from the cost map (e.g., Gaussian predictions)
                        path_cost += cell_cost * self.weights['risk']
                    else:
                        # Hitting the edge of the known map is treated as a collision
                        is_valid = False
                        break
                
                if not is_valid:
                    break

                # 2. Smoothness Cost (Penalize sharp steering/yaw changes)
                if idx > 0:
                    prev_pt = path[idx-1]
                    yaw_diff = abs(pt.yaw - prev_pt.yaw)
                    path_cost += (yaw_diff ** 2) * self.weights['smoothness']

            # 3. Centerline Preference (Prefer paths that don't swerve unnecessarily)
            if is_valid:
                lateral_deviation = abs(path[-1].y)
                path_cost += lateral_deviation * self.weights['centerline']

            # Update best path
            if is_valid and path_cost < min_cost:
                min_cost = path_cost
                best_path = path

        return best_path, min_cost