"""
Quintic Polynomial Trajectory Sampler.
Generates smooth candidate paths for unstructured road navigation.
"""

import numpy as np
from typing import List
from interfaces.data_types import EgoState, TrajectoryPoint

class PolynomialTrajectorySampler:
    def __init__(self, time_horizon: float = 3.0, dt: float = 0.2):
        self.time_horizon = time_horizon
        self.dt = dt
        # Different lateral offsets to sample (e.g., -2.5m is a left swerve, +2.5m is right)
        self.lateral_offsets = [-3.0, -1.5, 0.0, 1.5, 3.0] 

    def generate_candidates(self, ego: EgoState, target_speed: float) -> List[List[TrajectoryPoint]]:
        candidates = []
        t_steps = np.arange(0, self.time_horizon + self.dt, self.dt)

        for offset in self.lateral_offsets:
            path = []
            for t in t_steps:
                # 1. Constant velocity longitudinal prediction
                x = ego.x + (ego.vx * t)
                
                # 2. Smooth quintic polynomial lateral transition
                # Ensures smooth steering without sudden jerks
                tau = t / self.time_horizon if self.time_horizon > 0 else 0
                y_shift = offset * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)
                y = ego.y + y_shift
                
                # 3. Calculate heading (yaw) along the curve
                dx = ego.vx
                dy = offset * (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / self.time_horizon
                yaw = np.arctan2(dy, dx)
                
                path.append(TrajectoryPoint(x=x, y=y, v_target=target_speed, yaw=yaw, t=t))
            
            candidates.append(path)
            
        return candidates