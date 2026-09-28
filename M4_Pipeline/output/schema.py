"""M4 -> M5 interface (section 40). No M5 code exists yet (checked: no
M5_Pipeline folder in the repo), so this is a proposal, not an adaptation of
an existing contract -- but it deliberately follows the exact transport
pattern M2->M3 and M3->M4 already use (HTTP POST, JSON body, fire-and-
forget from a background task, short timeout) so M5 slots into the same
pipeline shape everything else already has, and M5 never blocks M4 the same
way M4 never blocks M3.

Frame: WORLD frame (same as M3's tracked objects), for the same reason M3
uses world frame internally (see M3_Pipeline/tracker.py's docstring) --
positions/velocities are stable across the ego's own maneuvers, and M5 gets
`ego_position`/`ego_yaw_deg` in every packet to convert to ego-relative
coordinates itself if its planner wants that instead. M4 states this choice
explicitly rather than silently picking one.

Design principle (section 41): M5 sees only WHAT an object will likely do
and HOW uncertain that is -- never the model that produced it. There is no
field here that leaks "this came from a Kalman filter" or "this used
constant acceleration"; `trajectories[i].mode` is a semantic label
("nominal"/"stop"/"lateral"), not an implementation detail. M4 deliberately
does NOT compute TTC/collision-probability/risk-score/risk-level here (cut
from scope -- see README.md's "Status" section); if M5 needs a danger
assessment, it derives one itself from these raw predicted trajectories.
"""
from __future__ import annotations

from pydantic import BaseModel


class TrajectoryOut(BaseModel):
    mode: str                        # "nominal" | "stop" | "lateral"
    points: list[list[float]]        # [[x, y], ...], world frame, one per time_step
    probability: float
    along_sigma_m: list[float]       # per-point along-track std-dev, meters
    lateral_sigma_m: list[float]     # per-point cross-track std-dev, meters
    heading_rad: float               # ellipse major-axis orientation for along/lateral sigma


class ObjectPrediction(BaseModel):
    track_id: int
    class_name: str
    is_static: bool
    trajectories: list[TrajectoryOut]
    horizon_s: float
    time_step_s: float

    input_quality: float             # tracking_input/quality.py score for this object this frame, [0,1]
    confidence: float                # overall prediction confidence, see predictor.py's docstring for how this differs from input_quality


class PredictionPacket(BaseModel):
    frame_id: int
    timestamp: float
    ego_position: list[float]        # [x, y], world frame -- so M5 can convert any prediction to ego-relative itself
    ego_velocity: list[float]        # [vx, vy], world frame
    ego_yaw_deg: float
    predictions: list[ObjectPrediction]
