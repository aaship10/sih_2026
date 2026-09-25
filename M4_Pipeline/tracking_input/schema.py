"""Pydantic models for the EXACT wire format M3 sends, verified field-by-
field against the current M3_Pipeline/m3_server.py `_to_track_dict` and the
`downstream()` handler's `_forward_to_m4` payload (both read directly, not
inferred from the mentor-notes document). If M3's code changes this shape,
update this file to match -- M3 is frozen and is the source of truth (see
M4_Pipeline/README.md's "verified against M3" section).

`last_match_pass`/`last_match_cost` are M3-internal diagnostics ("primary" |
"cascade" | "spawn", plus a raw match cost that may be null). M4 does not
currently use them for anything, but keeps them (Optional, permissive) so a
packet round-trips through the recorder/replay path unchanged.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TrackedObjectPacket(BaseModel):
    # extra="allow": forward-compatible if M3 adds a field M4 doesn't know
    # about yet. populate_by_name: lets code construct this model using the
    # Python-side name `class_name` while `class` (a reserved word, can't be
    # a field name) remains the accepted/emitted JSON key via the alias.
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    track_id: int
    class_name: str = Field(alias="class")
    position: list[float]                 # [x, y, z], CARLA world frame
    position_ego_relative: list[float]     # [x, y, z], ego frame (X-forward, Y-right, Z-up)
    velocity: list[float]                  # [vx, vy, vz], world frame, absolute m/s
    speed_mps: float
    heading_deg: Optional[float] = None
    size: list[float]                      # [l, w, h], axis-aligned LiDAR cluster extent (NOT oriented to object heading)
    confidence: float                      # M2's classification confidence (x0.6 camera-only, 0.4 lidar-only) -- NOT a tracking-quality score
    age: int
    hits: int
    time_since_update: float
    last_seen: float
    sensor_sources: list[str]
    timestamp: float
    track_status: str
    is_static: bool
    last_match_pass: Optional[str] = None
    last_match_cost: Optional[float] = None


class FramePacket(BaseModel):
    """Top-level payload M3 POSTs to /api/v1/tracks. Field-for-field match of
    m3_server.py's `_forward_to_m4` call site."""
    model_config = ConfigDict(extra="allow")

    frame_id: int
    timestamp: float
    ego_position: list[float]              # [x, y, z], world frame
    ego_velocity: list[float] = [0.0, 0.0, 0.0]  # [vx, vy, vz], world frame -- NOTE: m3_tracks.jsonl (M3's own log) omits this; only the live packet has it
    ego_yaw_deg: float = 0.0
    tracked_objects: list[TrackedObjectPacket] = []
