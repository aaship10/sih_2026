import numpy as np
from utils.coordinate_transforms import normalize_angle

class StanleyController:
    def __init__(self, k_gain=1.5, k_soft=1.0, max_steer=1.0):
        self.k = k_gain       # Control gain
        self.k_soft = k_soft  # Softening constant to prevent singularity at 0 speed
        self.max_steer = max_steer
        self.prev_steer = 0.0
        self.max_steer_rate = 0.1 # Limit steering change per tick for smoothness

    def calculate_steering(self, ego_x, ego_y, ego_yaw, ego_v, target_point):
        # 1. Heading Error
        yaw_target = target_point.yaw  # Changed from .heading
        yaw_error = normalize_angle(yaw_target - ego_yaw)

        # 2. Cross-Track Error
        # Vector from ego to target point
        dx = target_point.x - ego_x
        dy = target_point.y - ego_y
        
        # Cross track error is the projection of the distance vector onto the normal of the path
        # Using cross product-like 2D formula
        crosstrack_error = dx * np.sin(yaw_target) - dy * np.cos(yaw_target)

        # 3. Stanley Steering Calculation
        crosstrack_steer = np.arctan2(self.k * crosstrack_error, ego_v + self.k_soft)
        
        raw_steer = yaw_error + crosstrack_steer
        
        # 4. Smooth and Clamp
        steer = np.clip(raw_steer, -self.max_steer, self.max_steer)
        steer = np.clip(steer, self.prev_steer - self.max_steer_rate, self.prev_steer + self.max_steer_rate)
        
        self.prev_steer = steer
        return steer, crosstrack_error