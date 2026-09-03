"""
Data interfaces for M5 (Behavior Decision & Path Planning).
Defines strict schemas for upstream inputs (M3/M4) and downstream outputs (M6).
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class EgoState:
    x: float             # Global/Local X (m)
    y: float             # Global/Local Y (m)
    vx: float            # Forward velocity (m/s)
    vy: float            # Lateral velocity (m/s)
    yaw: float           # Heading angle (rad)
    accel: float         # Longitudinal acceleration (m/s^2)
    steering: float      # Current steering angle (rad)
    timestamp: float     # Simulation time (s)


@dataclass
class TrackedObject:
    track_id: int
    class_name: str      # 'car', 'auto', 'bike', 'pedestrian', 'cattle', 'pushcart'
    position: Tuple[float, float]        # [x, y] relative to ego frame (m)
    velocity: Tuple[float, float]        # [vx, vy] (m/s)
    heading: float                       # Heading (rad)
    dimensions: Tuple[float, float, float] # [length, width, height] (m)
    confidence: float


@dataclass
class PredictedTrajectory:
    waypoints: List[Tuple[float, float, float]]  # List of (x, y, dt_offset)
    probability: float                           # Trajectory confidence [0.0, 1.0]


@dataclass
class DynamicPrediction:
    track_id: int
    trajectories: List[PredictedTrajectory]
    
    # Set as default internal fields so M4 doesn't have to provide them!
    # M5 will overwrite these once the Costmap & TTC logic runs.
    risk_level: str = "UNKNOWN"                              
    collision_probability: float = 0.0


@dataclass
class DrivableSpace:
    resolution: float = 0.2                      # meters per grid cell
    grid_size: Tuple[int, int] = (200, 100)      # (length_x, width_y) in cells -> 40m x 20m
    origin_offset: Tuple[float, float] = (10.0, 10.0) # Ego position relative to bottom-left (m)
    occupancy_mask: np.ndarray = field(
        default_factory=lambda: np.ones((200, 100), dtype=np.uint8)
    )  # 1 = Drivable, 0 = Non-drivable boundary


@dataclass
class TrajectoryPoint:
    x: float
    y: float
    v_target: float
    yaw: float
    t: float


@dataclass
class PlanningOutput:
    behavior_state: str                          # "CRUISE", "FOLLOW", "AVOID", "YIELD", "EMERGENCY_BRAKE"
    trajectory: List[TrajectoryPoint]            # Planned spatial-temporal waypoints
    target_speed: float                          # Desired speed for controller (m/s)
    replan_triggered: bool                       # True if replanned during this cycle
    emergency_stop: bool                         # Hard brake flag for M6
    computation_time_ms: float                   # Latency metric
    timestamp: float

import time
from dataclasses import dataclass, field
from typing import List

# Change this class at the bottom of data_types.py
@dataclass
class TrajectoryPoint:
    x: float
    y: float
    yaw: float          # Changed from 'heading'
    v_target: float     # Changed from 'velocity'
    t: float = 0.0      # Changed from 'time_offset'

@dataclass
class PlannedTrajectory:
    timestamp: float = field(default_factory=time.time)
    points: List[TrajectoryPoint] = field(default_factory=list)
    target_behavior: str = "TRACK"  # "TRACK", "STOP", "EMERGENCY"
    is_valid: bool = True