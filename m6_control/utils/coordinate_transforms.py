import numpy as np
import math

def normalize_angle(angle):
    """Normalize angle to [-pi, pi]."""
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return angle

def calc_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def get_closest_waypoint_index(ego_x, ego_y, trajectory_points):
    """Finds the index of the closest trajectory point to the vehicle."""
    min_dist = float('inf')
    closest_idx = 0
    for i, pt in enumerate(trajectory_points):
        dist = calc_distance(ego_x, ego_y, pt.x, pt.y)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    return closest_idx