"""Multi-object tracking: turns per-frame fused detections into persistent
tracks with stable IDs, via a constant-velocity Kalman filter per track and
Hungarian-algorithm nearest-neighbor data association (M3 mentor notes,
section 15 -- deliberately the simplest reliable design: no learned
association, no appearance embeddings).

Tracking is done in the CARLA *world* frame, not the ego frame: the ego
vehicle accelerates and turns, which would inject fictitious forces into a
constant-velocity model if tracking were done in a frame attached to it.
Fused detections are converted ego -> world (using the ego pose carried on
each FramePacket) before being handed to `update`.

Association runs in two passes (added, then revised, after two rounds of
live CARLA testing surfaced concrete failure modes -- see README "Live CARLA
test findings & fixes"):

  Pass 1 -- Mahalanobis-gated Hungarian assignment across ALL tracks and
  detections, using each track's own Kalman covariance instead of a flat
  Euclidean radius. A fixed-radius gate let a faster, unrelated cluster
  hijack a well-established track on live data; weighting by the track's
  actual uncertainty (plus a hard absolute-distance safety cap) fixes that.

  Pass 2 -- a more permissive cascade for still-unmatched "unknown"-class
  tracks against still-unmatched "unknown"-class detections (see
  config.py's "Unknown-class cascade" notes for why this is gated on class,
  not on `Track.is_static` as it originally was -- that had a
  chicken-and-egg problem where a track needed to survive Pass 1 long enough
  to prove itself static before it could get help surviving Pass 1). LiDAR
  clustering jitter on clutter (curbs, potholes, building edges, parked
  vehicles) produces cluster shapes that shift frame to frame even when the
  object hasn't moved; without this pass, every jitter that fell outside the
  tight Pass-1 gate spawned a brand-new track_id for the same physical
  object. Pass 2's match itself is Euclidean-only (no Mahalanobis check --
  see config.py's "Run #3 finding" for why a hard distance/dt cutoff doesn't
  work here either), so `_apply_measurement` feeds it into the Kalman filter
  with a deliberately loose position measurement noise
  (config.CASCADE_POSITION_MEASUREMENT_STD) instead of the normal one: a
  wrong one-off cascade match then only nudges the state gently instead of
  forcing a full-trust jump, which is what let a fast unrelated cluster drag
  a slow/static track's velocity up to 29.8 m/s on live data before this fix.

Radar association gets its own gate inside `_apply_measurement`
(`KalmanFilter6D.radar_los_innovation`): sensor_fusion.py proposes a radar
candidate by proximity alone (nearest within RADAR_ASSOC_MAX_DIST_M), with
no idea which track it's near, so a fast unrelated target can be the
"nearest" radar return to an unrelated, slow/static track. Before a radar
reading counts toward RADAR_MIN_STREAK at all, it must be statistically
consistent with the track's OWN current velocity estimate and uncertainty --
a reading that's wildly inconsistent with a converged, tight-covariance
track is rejected outright (and doesn't reset progress towards nothing, it
just doesn't start a streak), while a track that's still uncertain (young,
or genuinely accelerating) naturally has a wider gate and isn't blocked.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment

import config
from kalman_filter import KalmanFilter6D
from track_types import Track

# scipy's linear_sum_assignment raises ValueError("cost matrix is infeasible")
# when a row/column is entirely np.inf (e.g. a single track vs. a single
# detection with no valid match) -- a large finite sentinel avoids that while
# still being trivially distinguishable from any real gated cost.
_NO_MATCH_COST = 1e6


class Tracker:
    def __init__(self) -> None:
        self._next_id = itertools.count(1)
        self.tracks: list[Track] = []

    def predict(self, dt: float) -> None:
        for track in self.tracks:
            track.kf.predict(dt)

    def update(self, detections: list[dict], timestamp: float, ego_position_world: np.ndarray) -> None:
        """`detections` are fused objects with a `position_world` key (see
        m3_server.py, where ego-frame fused objects are converted to world
        frame before this call)."""
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        if self.tracks and detections:
            cost = np.full((len(self.tracks), len(detections)), _NO_MATCH_COST)
            for i, track in enumerate(self.tracks):
                H, P, R = track.kf.H, track.kf.P, track.kf.R
                S_inv = np.linalg.inv(H @ P @ H.T + R)
                for j, det in enumerate(detections):
                    y = det["position_world"] - track.kf.position
                    if np.linalg.norm(y) > config.TRACK_GATING_MAX_DIST_M:
                        continue
                    mahalanobis_sq = float(y.T @ S_inv @ y)
                    if mahalanobis_sq <= config.TRACK_GATING_CHI2:
                        cost[i, j] = mahalanobis_sq
            row_idx, col_idx = linear_sum_assignment(cost)
            for r, c in zip(row_idx, col_idx):
                if cost[r, c] < _NO_MATCH_COST:
                    self._apply_measurement(
                        self.tracks[r], detections[c], timestamp, ego_position_world, cost=float(cost[r, c])
                    )
                    matched_tracks.add(r)
                    matched_dets.add(c)

        unmatched_track_idx = [i for i in range(len(self.tracks)) if i not in matched_tracks]
        unmatched_det_idx = [j for j in range(len(detections)) if j not in matched_dets]

        # Gated on class, not `is_static` -- see this module's and config.py's
        # docstrings for why (chicken-and-egg: `is_static` needs hits this
        # cascade exists to help a track accumulate in the first place).
        unknown_track_idx = [i for i in unmatched_track_idx if self.tracks[i].class_name == "unknown"]
        unknown_det_idx = [j for j in unmatched_det_idx if detections[j]["class_name"] == "unknown"]
        if unknown_track_idx and unknown_det_idx:
            cascade_radius = config.STATIC_CASCADE_DIST_FACTOR * config.TRACK_GATING_MAX_DIST_M
            cost2 = np.full((len(unknown_track_idx), len(unknown_det_idx)), _NO_MATCH_COST)
            for a, i in enumerate(unknown_track_idx):
                for b, j in enumerate(unknown_det_idx):
                    d = np.linalg.norm(self.tracks[i].kf.position - detections[j]["position_world"])
                    if d <= cascade_radius:
                        cost2[a, b] = d
            row2, col2 = linear_sum_assignment(cost2)
            cascade_matched_tracks: set[int] = set()
            cascade_matched_dets: set[int] = set()
            for a, b in zip(row2, col2):
                if cost2[a, b] < _NO_MATCH_COST:
                    i, j = unknown_track_idx[a], unknown_det_idx[b]
                    self._apply_measurement(
                        self.tracks[i], detections[j], timestamp, ego_position_world,
                        cascade=True, cost=float(cost2[a, b]),
                    )
                    cascade_matched_tracks.add(i)
                    cascade_matched_dets.add(j)
            unmatched_track_idx = [i for i in unmatched_track_idx if i not in cascade_matched_tracks]
            unmatched_det_idx = [j for j in unmatched_det_idx if j not in cascade_matched_dets]

        for i in unmatched_track_idx:
            track = self.tracks[i]
            track.misses += 1
            track.radar_streak = 0
            track.radar_miss_streak += 1

        for j in unmatched_det_idx:
            self._spawn_track(detections[j], timestamp)

        for track in self.tracks:
            track.age_frames += 1
            if track.status == "tentative" and track.hits >= config.TRACK_CONFIRM_HITS:
                track.status = "confirmed"

        self.tracks = [
            t for t in self.tracks
            if t.misses <= config.TRACK_MAX_MISSES and (timestamp - t.last_update_time) <= config.TRACK_MAX_AGE_S
        ]

    def _apply_measurement(
        self, track: Track, det: dict, timestamp: float, ego_position_world: np.ndarray,
        cascade: bool = False, cost: float | None = None,
    ) -> None:
        track.last_match_pass = "cascade" if cascade else "primary"
        track.last_match_cost = cost

        if cascade:
            cascade_R = np.eye(3) * (config.CASCADE_POSITION_MEASUREMENT_STD ** 2)
            track.kf.update(det["position_world"], R=cascade_R)
        else:
            track.kf.update(det["position_world"])

        if "radar_target" in det:
            radar_sensor_world = det.get("radar_sensor_position_world", ego_position_world)
            range_rate = det["radar_range_rate_mps"]
            innovation = track.kf.radar_los_innovation(radar_sensor_world, range_rate)
            consistent = innovation is not None and (innovation[0] ** 2 / innovation[1]) <= config.RADAR_LOS_GATE_CHI2
            if consistent:
                track.radar_streak += 1
                track.radar_miss_streak = 0
                track.last_radar_range_rate = range_rate
                if track.radar_streak >= config.RADAR_MIN_STREAK:
                    track.kf.update_radar_los(radar_sensor_world, range_rate)
            else:
                # Statistically inconsistent with this track's own current
                # velocity estimate -- almost certainly a nearby-but-unrelated
                # target (see this module's docstring). Treated as no signal
                # for this frame rather than folded in: doesn't reset
                # progress toward anything, just never starts a streak.
                track.radar_streak = 0
                track.radar_miss_streak += 1
        else:
            track.radar_streak = 0

        speed = np.linalg.norm(track.kf.velocity[:2])
        if speed > 0.5:
            track.heading_deg = float(np.degrees(np.arctan2(track.kf.velocity[1], track.kf.velocity[0])))

        if det["class_name"] != "unknown":
            track.class_name = det["class_name"]
            track.class_confidence = det["confidence"]
        elif track.class_name == "unknown":
            track.class_confidence = max(track.class_confidence, det["confidence"])

        track.size = det["size"]
        track.sensor_sources = set(det["sensor_sources"])
        track.hits += 1
        track.misses = 0
        track.last_update_time = timestamp
        track.refresh_static_flag()

    def _spawn_track(self, det: dict, timestamp: float) -> None:
        kf = KalmanFilter6D(det["position_world"])
        track = Track(
            track_id=next(self._next_id),
            kf=kf,
            class_name=det["class_name"],
            class_confidence=det["confidence"],
            size=det["size"],
            sensor_sources=set(det["sensor_sources"]),
            created_at=timestamp,
            last_update_time=timestamp,
        )
        self.tracks.append(track)

    def confirmed_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.status == "confirmed"]
