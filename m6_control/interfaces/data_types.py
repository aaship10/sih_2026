import time
from dataclasses import dataclass, field
from typing import List

@dataclass
class TrajectoryPoint:
    x: float
    y: float
    heading: float
    velocity: float
    time_offset: float = 0.0

@dataclass
class PlannedTrajectory:
    timestamp: float = field(default_factory=time.time)
    points: List[TrajectoryPoint] = field(default_factory=list)
    target_behavior: str = "TRACK"  # "TRACK", "STOP", "EMERGENCY"
    is_valid: bool = True