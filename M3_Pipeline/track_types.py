"""Data structures for M3 tracks."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import config
from kalman_filter import KalmanFilter6D


@dataclass
class Track:
    track_id: int
    kf: KalmanFilter6D
    class_name: str
    class_confidence: float
    size: np.ndarray
    sensor_sources: set
    created_at: float
    last_update_time: float
    first_position_world: np.ndarray = field(default=None)
    hits: int = 1
    misses: int = 0
    age_frames: int = 1
    status: str = "tentative"  # "tentative" | "confirmed"
    heading_deg: float | None = None
    last_radar_range_rate: float | None = None
    radar_streak: int = 0
    radar_miss_streak: int = 0
    is_static: bool = False
    # Diagnostic only (not part of the frozen M3->M4 contract -- see README):
    # which association pass last updated this track, and how close that
    # match was to its pass's own gate edge. Added to attribute a residual
    # rare velocity-jump tail case (Pass 1 leak vs. Pass 2 cascade) without
    # guessing at another algorithm change first.
    last_match_pass: str = "spawn"  # "spawn" | "primary" | "cascade"
    last_match_cost: float | None = None

    def __post_init__(self) -> None:
        if self.first_position_world is None:
            self.first_position_world = self.kf.position.copy()

    def refresh_static_flag(self) -> None:
        """Re-evaluated on every successful update (see config.py's
        static-cascade notes) so a track can un-flag itself if it turns out
        to actually be moving."""
        displacement = np.linalg.norm(self.kf.position - self.first_position_world)
        speed = np.linalg.norm(self.kf.velocity[:2])
        # bool(...) matters: `displacement < ...` and `speed < ...` are
        # numpy.bool_, and `python_bool and numpy_bool and numpy_bool`
        # short-circuits to a raw numpy.bool_ -- which json.dumps cannot
        # serialize. Live testing hit exactly this: it silently 500'd the
        # tracks-log write for ~150 consecutive frames and, since M4
        # forwarding is fire-and-forget, silently broke 100% of M3->M4
        # delivery with no visible error. See also m3_server.py's defensive
        # `_json_default` as a second line of defense against this class of
        # bug.
        self.is_static = bool(
            self.hits >= config.TRACK_CONFIRM_HITS
            and displacement < config.STATIC_DISPLACEMENT_THRESHOLD_M
            and speed < config.STATIC_SPEED_THRESHOLD_MPS
        )
