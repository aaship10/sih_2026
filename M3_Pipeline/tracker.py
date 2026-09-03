"""
tracker.py
==========
Turns single-frame fused detections into persistent objects with IDs
that stay the same across frames ("Object #17" stays "#17" as it
moves), instead of treating every frame as brand-new strangers.

Two jobs happen here:
  1. DATA ASSOCIATION: which existing track does each new detection
     belong to? (simple nearest-position matching)
  2. KALMAN FILTER: smooths noisy position readings and estimates
     velocity, by blending "what we predicted" with "what we measured".

State per track: [x, y, vx, vy]  (position + velocity, 2D is enough
for ground-vehicle path planning; z/height is carried along separately
for reference but not filtered).
"""

import numpy as np


class Track:
    """One tracked object with a persistent ID."""

    _next_id = 1  # class-level counter so every new Track gets a unique ID

    def __init__(self, position_xy, velocity_xy, cls, confidence, size, timestamp):
        self.track_id = Track._next_id
        Track._next_id += 1

        # Kalman filter state: [x, y, vx, vy]
        self.state = np.array([position_xy[0], position_xy[1], velocity_xy[0], velocity_xy[1]], dtype=float)

        # uncertainty (covariance) -- starts fairly uncertain, shrinks as
        # we get more confirming measurements
        self.P = np.eye(4) * 5.0

        self.cls = cls
        self.confidence = confidence
        self.size = size
        self.z = position_xy[2] if len(position_xy) > 2 else 0.0

        self.age = 1                 # how many frames this track has existed
        self.hits = 1                # how many frames it was actually matched to a detection
        self.time_since_update = 0   # frames since last matched detection (for deletion)
        self.last_seen = timestamp

    # ---- Kalman filter math ----

    def predict(self, dt):
        """
        Predict step: "based on last known velocity, where should this
        object be NOW?" Simple constant-velocity motion model:
            x_new = x + vx * dt
            y_new = y + vy * dt
            vx, vy stay the same (we assume no sudden acceleration
            between frames, which is a fine approximation at 5-10 Hz)
        """
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        # process noise: how much we trust the constant-velocity
        # assumption -- small values, since real objects can accelerate
        Q = np.eye(4) * 0.05

        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q
        self.time_since_update += 1
        self.age += 1

    def update(self, position_xy, velocity_xy, confidence, size, timestamp):
        """
        Update step: blend the prediction with the new real measurement
        (weighted by how much we trust each -- this weighting is what
        the Kalman "gain" computes automatically).
        """
        # measurement = [x, y, vx, vy] (we treat radar's velocity as a
        # direct measurement too, when available)
        z_meas = np.array([position_xy[0], position_xy[1], velocity_xy[0], velocity_xy[1]])

        H = np.eye(4)              # we measure the full state directly
        R = np.eye(4) * 0.5        # measurement noise: how noisy we think detections are

        y_residual = z_meas - H @ self.state
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)   # Kalman gain

        self.state = self.state + K @ y_residual
        self.P = (np.eye(4) - K @ H) @ self.P

        if len(position_xy) > 2:
            self.z = position_xy[2]
        self.cls = self.cls if confidence < self.confidence else self.cls
        self.confidence = max(self.confidence, confidence) * 0.9 + confidence * 0.1
        self.size = size if size is not None else self.size
        self.hits += 1
        self.time_since_update = 0
        self.last_seen = timestamp

    def to_output_dict(self):
        """Formats this track exactly in the structure M4 expects."""
        x, y, vx, vy = self.state
        return {
            "track_id": self.track_id,
            "class": self.cls,
            "position": [round(float(x), 2), round(float(y), 2), round(float(self.z), 2)],
            "velocity": [round(float(vx), 2), round(float(vy), 2), 0.0],
            "size": self.size,
            "confidence": round(float(self.confidence), 2),
            "age": self.age,
            "last_seen": round(self.last_seen, 2),
            "sensor_sources": ["camera", "lidar", "radar"],
            "timestamp": round(self.last_seen, 2),
        }


class MultiObjectTracker:
    """
    Manages the full set of currently-tracked objects: matches new
    detections to existing tracks, creates new tracks, deletes stale
    ones, and predicts forward every frame.
    """

    def __init__(self, max_match_distance_m=2.0, min_hits_to_confirm=2,
                 max_age_without_update=5):
        self.tracks = []
        self.max_match_distance_m = max_match_distance_m
        self.min_hits_to_confirm = min_hits_to_confirm
        self.max_age_without_update = max_age_without_update

    def step(self, fused_detections, dt, timestamp):
        """
        Runs one full tracking cycle for the current frame:
            1. Predict all existing tracks forward by dt.
            2. Match detections to tracks (nearest position, within a
               distance threshold).
            3. Update matched tracks with their new measurement.
            4. Create new tracks for detections that matched nothing.
            5. Delete tracks that haven't been seen for too long
               (handles temporary occlusion gracefully -- we don't
               delete instantly, but we do eventually if it's really gone).

        Returns: list of confirmed tracks in M4's expected output format.
        """
        # 1) predict
        for track in self.tracks:
            track.predict(dt)

        # only detections with a known 3D position (i.e. LiDAR-confirmed)
        # can be matched/tracked with real position+velocity
        usable_detections = [d for d in fused_detections if d["position"] is not None]

        # 2) match detections to existing tracks (greedy nearest-neighbor)
        unmatched_detections = list(range(len(usable_detections)))
        unmatched_tracks = list(range(len(self.tracks)))
        matches = []  # (track_index, detection_index)

        for t_idx in list(unmatched_tracks):
            track = self.tracks[t_idx]
            best_d_idx = None
            best_dist = self.max_match_distance_m
            for d_idx in unmatched_detections:
                det = usable_detections[d_idx]
                dx = track.state[0] - det["position"][0]
                dy = track.state[1] - det["position"][1]
                dist = np.hypot(dx, dy)
                if dist < best_dist:
                    best_dist = dist
                    best_d_idx = d_idx
            if best_d_idx is not None:
                matches.append((t_idx, best_d_idx))
                unmatched_tracks.remove(t_idx)
                unmatched_detections.remove(best_d_idx)

        # 3) update matched tracks
        for t_idx, d_idx in matches:
            det = usable_detections[d_idx]
            self.tracks[t_idx].update(
                det["position"], det["velocity"], det["confidence"], det["size"], timestamp
            )

        # 4) create new tracks for leftover detections
        for d_idx in unmatched_detections:
            det = usable_detections[d_idx]
            new_track = Track(
                det["position"], det["velocity"], det["class"],
                det["confidence"], det["size"], timestamp
            )
            self.tracks.append(new_track)

        # 5) delete stale tracks (not seen for too many frames in a row --
        # gives temporarily-occluded objects a grace period instead of
        # deleting them the instant one frame is missed)
        self.tracks = [
            t for t in self.tracks if t.time_since_update <= self.max_age_without_update
        ]

        # only report tracks that have been confirmed by enough hits
        # (avoids reporting a "track" from a single noisy detection)
        confirmed = [t for t in self.tracks if t.hits >= self.min_hits_to_confirm]
        return [t.to_output_dict() for t in confirmed]