"""Converts a raw M3 FramePacket into M4's internal shapes (EgoState,
TrackedObject) and enforces packet ordering.

M3 forwards each frame as an independent asyncio background task (see
m3_server.py's `_forward_to_m4`), so packets can arrive at M4 out of order
or overlapping under load. `PacketOrderer` is the single gate everything
else in M4 sits behind: a packet older than the newest one already accepted
is dropped here, before it can touch any track history.

Only x,y are used from every 3D field (z/vz are tracked by M3 but noted as
unreliable and unnecessary -- section 7.2). Heading is stored in radians
internally (all of numpy's trig works in radians); the wire format's
heading_deg is converted once, here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from tracking_input.schema import FramePacket, TrackedObjectPacket


@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    position: np.ndarray            # [x, y] world frame, meters
    velocity: np.ndarray            # [vx, vy] world frame, m/s
    heading_rad: float | None       # None until the track has exceeded M3's 0.5 m/s heading-update threshold at least once
    size: tuple[float, float, float]
    confidence: float
    age: int
    hits: int
    time_since_update: float
    last_seen: float
    sensor_sources: frozenset[str]
    is_static: bool
    track_status: str
    timestamp: float
    quality: float = 1.0            # filled in by tracking_input/quality.py

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))


@dataclass
class EgoState:
    position: np.ndarray   # [x, y] world frame
    velocity: np.ndarray   # [vx, vy] world frame
    speed: float
    heading_rad: float     # from ego_yaw_deg
    timestamp: float


@dataclass
class Frame:
    frame_id: int
    timestamp: float
    ego: EgoState
    objects: list[TrackedObject]


def _to_tracked_object(p: TrackedObjectPacket) -> TrackedObject:
    return TrackedObject(
        track_id=p.track_id,
        class_name=p.class_name,
        position=np.array(p.position[:2], dtype=float),
        velocity=np.array(p.velocity[:2], dtype=float),
        heading_rad=math.radians(p.heading_deg) if p.heading_deg is not None else None,
        size=(p.size[0], p.size[1], p.size[2]) if len(p.size) >= 3 else (0.6, 0.6, 1.0),
        confidence=p.confidence,
        age=p.age,
        hits=p.hits,
        time_since_update=p.time_since_update,
        last_seen=p.last_seen,
        sensor_sources=frozenset(p.sensor_sources),
        is_static=p.is_static,
        track_status=p.track_status,
        timestamp=p.timestamp,
    )


def to_frame(packet: FramePacket) -> Frame:
    """Pure conversion, no ordering/dedup -- callers that need ordering
    should go through PacketOrderer.accept() instead of calling this
    directly (the live server and replay reader both do)."""
    ego = EgoState(
        position=np.array(packet.ego_position[:2], dtype=float),
        velocity=np.array(packet.ego_velocity[:2], dtype=float),
        speed=float(np.linalg.norm(packet.ego_velocity[:2])),
        heading_rad=math.radians(packet.ego_yaw_deg),
        timestamp=packet.timestamp,
    )
    objects = [_to_tracked_object(p) for p in packet.tracked_objects]
    return Frame(frame_id=packet.frame_id, timestamp=packet.timestamp, ego=ego, objects=objects)


@dataclass
class PacketOrderer:
    """Drops any packet whose (timestamp, frame_id) is not strictly newer
    than the last accepted packet. Ties on timestamp (shouldn't happen, but
    defensively) are broken on frame_id."""
    _last_timestamp: float = field(default=-1.0)
    _last_frame_id: int = field(default=-1)
    dropped_count: int = field(default=0)

    def accept(self, packet: FramePacket) -> Frame | None:
        key = (packet.timestamp, packet.frame_id)
        last_key = (self._last_timestamp, self._last_frame_id)
        if key <= last_key:
            self.dropped_count += 1
            return None
        self._last_timestamp, self._last_frame_id = key
        return to_frame(packet)
