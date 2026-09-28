"""M5's own data contract, both directions. Plain dataclasses, not pydantic
-- M5's input is UDP JSON already built+validated on M4's side
(output/m5_udp_schema.py there mirrors this exactly), and re-validating a
trusted, same-machine, same-pipeline payload would just add overhead on the
10Hz hot path for no real safety gain (unlike M2/M3/M4's HTTP boundaries,
which face a network hop and independent processes that could send
anything)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrajectoryPoint:
    x: float
    y: float
    t: float


@dataclass
class PredictedTrajectory:
    mode: str
    probability: float
    points: list[TrajectoryPoint]
    sigma_m: list[float] = field(default_factory=list)   # open item B, resolved: see M4's output/m5_udp_schema.py


@dataclass
class Obstacle:
    track_id: int
    class_name: str
    confidence: float
    pos_x: float
    pos_y: float
    vel_x: float
    vel_y: float
    width: float
    length: float
    trajectories: list[PredictedTrajectory]


@dataclass
class StaticObstacle:
    track_id: int
    class_name: str
    confidence: float  # needed by the sensor-artifact filter (see config.py)
    pos_x: float
    pos_y: float
    width: float
    length: float


@dataclass
class ObstaclePacket:
    """What M5 receives from M4 over UDP (M4's output/m5_udp_schema.py)."""
    timestamp: float
    frame_id: int
    ego_position: list[float]
    ego_velocity: list[float]
    ego_yaw_deg: float
    obstacles: list[Obstacle]
    statics: list[StaticObstacle]


def parse_obstacle_packet(raw: dict) -> ObstaclePacket:
    def _traj(t: dict) -> PredictedTrajectory:
        return PredictedTrajectory(
            mode=t["mode"], probability=t["probability"],
            points=[TrajectoryPoint(**p) for p in t["points"]],
            sigma_m=t.get("sigma_m", []),
        )

    def _obstacle(o: dict) -> Obstacle:
        return Obstacle(
            track_id=o["track_id"], class_name=o["class"], confidence=o["confidence"],
            pos_x=o["pos_x"], pos_y=o["pos_y"], vel_x=o["vel_x"], vel_y=o["vel_y"],
            width=o["width"], length=o["length"],
            trajectories=[_traj(t) for t in o["trajectories"]],
        )

    def _static(s: dict) -> StaticObstacle:
        return StaticObstacle(
            track_id=s["track_id"], class_name=s["class"], confidence=s.get("confidence", 1.0),
            pos_x=s["pos_x"], pos_y=s["pos_y"], width=s["width"], length=s["length"],
        )

    return ObstaclePacket(
        timestamp=raw["timestamp"], frame_id=raw["frame_id"],
        ego_position=raw["ego_position"], ego_velocity=raw["ego_velocity"], ego_yaw_deg=raw["ego_yaw_deg"],
        obstacles=[_obstacle(o) for o in raw.get("obstacles", [])],
        statics=[_static(s) for s in raw.get("statics", [])],
    )


@dataclass
class Waypoint:
    x: float
    y: float
    v_target: float
    yaw: float
    t: float


@dataclass
class DecisionPacket:
    """What M5 broadcasts to M1 over UDP (spec section 1, "Output Broadcast
    to M1"). Frame/units: WORLD frame, meters, radians -- see config.py."""
    timestamp: float
    frame_id: int
    behavior_state: str            # "CRUISE" | "FOLLOW" | "AVOID" | "YIELD" | "EMERGENCY_BRAKE"
    emergency_stop: bool
    target_speed: float
    trajectory: list[Waypoint]     # exactly config.N_WAYPOINTS entries
    replan_triggered: bool
    computation_time_ms: float

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "behavior_state": self.behavior_state,
            "emergency_stop": self.emergency_stop,
            "target_speed": self.target_speed,
            "trajectory": [{"x": w.x, "y": w.y, "v_target": w.v_target, "yaw": w.yaw, "t": w.t} for w in self.trajectory],
            "replan_triggered": self.replan_triggered,
            "computation_time_ms": self.computation_time_ms,
        }
