"""
Local Cost Map Generator for unstructured roads.
Handles drivable bounds, static obstacle inflation, and dynamic Gaussian prediction splatting.
"""

from typing import List, Tuple
import numpy as np
from scipy.ndimage import distance_transform_edt
from interfaces.data_types import DrivableSpace, TrackedObject, DynamicPrediction


class LocalCostMap:
    def __init__(self, resolution: float = 0.2, length_m: float = 40.0, width_m: float = 20.0,
                 ego_offset_x: float = 10.0, ego_offset_y: float = 10.0):
        self.resolution = resolution
        self.nx = int(length_m / resolution)  # 200 cells
        self.ny = int(width_m / resolution)   # 100 cells
        self.origin_x = ego_offset_x
        self.origin_y = ego_offset_y

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int((x + self.origin_x) / self.resolution)
        gy = int((y + self.origin_y) / self.resolution)
        return gx, gy

    def build_costmap(self, drivable: DrivableSpace, static_objects: List[TrackedObject],
                      dynamic_preds: List[DynamicPrediction], time_horizon_step: float = 0.0) -> np.ndarray:
        # Base grid: 0 (free) to 255 (obstacle/non-drivable)
        costmap = np.zeros((self.nx, self.ny), dtype=np.float32)

        # 1. Non-drivable boundaries
        if drivable.occupancy_mask.shape == (self.nx, self.ny):
            costmap[drivable.occupancy_mask == 0] = 255.0

        # 2. Static obstacle inflation
        static_mask = np.zeros((self.nx, self.ny), dtype=bool)
        for obj in static_objects:
            gx, gy = self.world_to_grid(obj.position[0], obj.position[1])
            l_cells = int((obj.dimensions[0] / 2) / self.resolution)
            w_cells = int((obj.dimensions[1] / 2) / self.resolution)
            x_min, x_max = max(0, gx - l_cells), min(self.nx, gx + l_cells + 1)
            y_min, y_max = max(0, gy - w_cells), min(self.ny, gy + w_cells + 1)
            static_mask[x_min:x_max, y_min:y_max] = True

        if np.any(static_mask):
            dist = distance_transform_edt(~static_mask) * self.resolution
            inflation_radius = 1.8  # Safety margin in meters
            inflation_cost = np.clip((1.0 - (dist / inflation_radius)) * 200.0, 0, 200)
            costmap = np.maximum(costmap, inflation_cost)
            costmap[static_mask] = 255.0

        # 3. Dynamic predicted obstacle footprint (Gaussian splat at target time step)
        for pred in dynamic_preds:
            for traj in pred.trajectories:
                # Find waypoint closest to the target time step
                closest_pt = min(traj.waypoints, key=lambda pt: abs(pt[2] - time_horizon_step))
                gx, gy = self.world_to_grid(closest_pt[0], closest_pt[1])

                if 0 <= gx < self.nx and 0 <= gy < self.ny:
                    sigma_x, sigma_y = 1.5 / self.resolution, 1.0 / self.resolution
                    kx = np.arange(max(0, gx - 15), min(self.nx, gx + 16))
                    ky = np.arange(max(0, gy - 15), min(self.ny, gy + 16))
                    KX, KY = np.meshgrid(kx, ky, indexing='ij')

                    gaussian = traj.probability * 255.0 * np.exp(
                        -(((KX - gx) ** 2) / (2 * sigma_x ** 2) + ((KY - gy) ** 2) / (2 * sigma_y ** 2))
                    )
                    costmap[kx[0]:kx[-1] + 1, ky[0]:ky[-1] + 1] = np.maximum(
                        costmap[kx[0]:kx[-1] + 1, ky[0]:ky[-1] + 1], gaussian
                    )

        return np.clip(costmap, 0, 255).astype(np.uint8)