"""Per-track history buffers, keyed by M3's track_id.

M3's track_ids churn continuously (~1.4 new ids/frame on live data, mostly
short-lived "unknown" clutter -- M3 README 7B.7): a real physical object can
also silently jump to a new track_id. TrackHistory therefore:
  - bounds each track's buffer to config.HISTORY_MAX_LEN samples (a ring,
    not an ever-growing list),
  - prunes any track_id not updated within config.TRACK_STALE_SECONDS,
so memory is bounded by (number of currently-live tracks) x HISTORY_MAX_LEN,
not by run length.

A coasted sample (time_since_update > 0, i.e. M3 dead-reckoned it rather
than measuring it -- M3 README 7B.2) is still appended, but tagged
`measured=False`, so downstream code (motion models, quality-weighted
fitting) can choose to down-weight or skip it instead of treating a
predicted state as if it were a real observation.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

import config
from tracking_input.adapter import TrackedObject


@dataclass
class HistorySample:
    timestamp: float
    position: np.ndarray
    velocity: np.ndarray
    measured: bool   # False if this sample had time_since_update > 0 (coasted, not a real M3 measurement)
    quality: float


class TrackHistory:
    def __init__(self, max_len: int = config.HISTORY_MAX_LEN) -> None:
        self._max_len = max_len
        self._buffers: dict[int, deque[HistorySample]] = {}
        self._last_seen: dict[int, float] = {}

    def update(self, obj: TrackedObject, frame_timestamp: float) -> None:
        buf = self._buffers.setdefault(obj.track_id, deque(maxlen=self._max_len))
        buf.append(HistorySample(
            timestamp=obj.timestamp,
            position=obj.position.copy(),
            velocity=obj.velocity.copy(),
            measured=obj.time_since_update <= 0.0,
            quality=obj.quality,
        ))
        self._last_seen[obj.track_id] = frame_timestamp

    def get(self, track_id: int) -> list[HistorySample]:
        buf = self._buffers.get(track_id)
        return list(buf) if buf else []

    def prune(self, current_timestamp: float, stale_after_s: float = config.TRACK_STALE_SECONDS) -> int:
        """Drop any track not updated within `stale_after_s`. Returns the
        number of tracks dropped. Call once per frame from predictor.py."""
        stale_ids = [
            tid for tid, last in self._last_seen.items()
            if current_timestamp - last > stale_after_s
        ]
        for tid in stale_ids:
            self._buffers.pop(tid, None)
            self._last_seen.pop(tid, None)
        return len(stale_ids)

    def __len__(self) -> int:
        return len(self._buffers)
