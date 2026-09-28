"""M3 configuration: sensor mounting, camera intrinsics, and pipeline thresholds.

Sensor mounting transforms MUST mirror scenario3.py (owned by M1): CAMERA_TRANSFORM,
LIDAR_TRANSFORM and RADAR_TRANSFORM below are copied from that file's constants of the
same name. If M1 changes sensor placement (or ships a new scenario file with different
mounts), update the values here to match, or every 3D position M3 computes will be
silently wrong. Camera resolution/FOV must likewise mirror the `sensor.camera.rgb`
attributes M1 sets on the camera blueprint.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Transform:
    x: float
    y: float
    z: float
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    roll_deg: float = 0.0


# --- Sensor mounts relative to the ego vehicle origin (scenario3.py) ---
CAMERA_TRANSFORM = Transform(x=1.5, y=0.0, z=2.2, pitch_deg=-5.0)
LIDAR_TRANSFORM = Transform(x=0.0, y=0.0, z=2.3)
RADAR_TRANSFORM = Transform(x=2.0, y=0.0, z=1.2)

# --- Camera intrinsics (must mirror scenario3.py camera attributes) ---
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
# Reverted to 90 after a controlled A/B on village_road found no real
# benefit to widening it (M1_Pipeline/common.py's setup_ego_sensors has the
# full rationale and the A/B numbers).
# MUST mirror M1's actual `sensor.camera.rgb` fov attribute exactly -- a
# mismatch here silently corrupts every camera-LiDAR fusion projection
# (camera_lidar_fusion.py / coordinate_transforms.py), not just FOV coverage.
CAMERA_FOV_DEG = 90.0
CAMERA_FX = CAMERA_WIDTH / (2.0 * math.tan(math.radians(CAMERA_FOV_DEG) / 2.0))
CAMERA_FY = CAMERA_FX
CAMERA_CX = CAMERA_WIDTH / 2.0
CAMERA_CY = CAMERA_HEIGHT / 2.0

# --- Assumption shared by ground removal / ROI / back-projection: the ego
# actor's own origin sits approximately at ground level (standard for CARLA
# vehicle blueprints -- the actor transform is at the wheel-contact plane).
# All ego-frame z values are relative to that assumption.
GROUND_PLANE_EGO_Z = 0.0

# --- LiDAR processing ---
LIDAR_GROUND_RANSAC_ITERATIONS = 40
LIDAR_GROUND_DIST_THRESHOLD_M = 0.22
LIDAR_GROUND_MIN_INLIER_RATIO = 0.25
LIDAR_ROI_X_MIN, LIDAR_ROI_X_MAX = -10.0, 70.0
LIDAR_ROI_Y_MIN, LIDAR_ROI_Y_MAX = -35.0, 35.0
LIDAR_ROI_Z_MIN, LIDAR_ROI_Z_MAX = -1.0, 3.0
LIDAR_CLUSTER_EPS_M = 0.6
# Raised from 4 -> 7 after live CARLA testing showed single/few-point noise
# blips contributing to spurious short-lived clusters (see tracker.py Fix 2
# notes and README "Known assumptions/limitations").
LIDAR_CLUSTER_MIN_POINTS = 7

# --- Radar processing ---
RADAR_MAX_RANGE_M = 80.0
# CARLA's docs describe RadarDetection.velocity as "towards the sensor" (i.e.
# positive = approaching). This is genuinely ambiguous without a live check
# against ground truth; if downstream (M4) risk estimates look inverted,
# flip this sign and re-run -- see radar_processing.py.
RADAR_VELOCITY_SIGN = 1.0

# --- Camera-LiDAR association ---
CAM_LIDAR_MIN_IOU = 0.08

# --- Camera-radar / LiDAR-radar association ---
RADAR_ASSOC_MAX_DIST_M = 3.0

# --- Radar-to-track Kalman fusion ---
# Std-dev (m/s) assumed for the radar range-rate measurement noise, used in
# the scalar Kalman update on the line-of-sight velocity component (see
# kalman_filter.KalmanFilter6D.update_radar_los). Replaces the old fixed
# blend-fraction heuristic, which produced a velocity sawtooth on live CARLA
# data whenever radar association flickered on/off (see README).
RADAR_LOS_MEASUREMENT_STD = 1.5
# A radar association must persist for this many consecutive frames on the
# SAME track before it's trusted enough to feed the Kalman update -- a single
# flickering frame, even Kalman-weighted, shouldn't move a stable track.
RADAR_MIN_STREAK = 2
#
# Run #2 finding: this alone wasn't enough. sensor_fusion._nearest_radar()
# proposes radar candidates by proximity only (nearest within
# RADAR_ASSOC_MAX_DIST_M), with no idea which track it's near -- so a fast
# unrelated target (a racer bike, an overtaker) can get glued onto a nearby,
# unrelated, near-static track. The Kalman weighting correctly trusts a
# noisy-but-right measurement more smoothly; it does nothing to protect
# against a confidently *wrong* one, and the streak gate alone just delayed
# the bad correction by RADAR_MIN_STREAK frames before applying it in one
# larger jump (observed live: 0.15 -> 40.27 m/s in a single 0.1s step, right
# as the streak gate first passed). RADAR_LOS_GATE_CHI2 below is the fix:
# gate each candidate radar reading against the TRACK's own velocity
# uncertainty (kalman_filter.KalmanFilter6D.radar_los_innovation) before it
# counts toward the streak at all -- a reading wildly inconsistent with a
# track's current (converged, tight-covariance) velocity estimate is
# rejected outright and never reaches the streak/update step; a track that's
# still uncertain (young, or genuinely accelerating) has a naturally wider
# gate and isn't blocked. 1 DOF, ~95% confidence.
RADAR_LOS_GATE_CHI2 = 3.841

# --- Camera detection pre-filter ---
# M2 runs YOLO at a deliberately low confidence (0.05, see perception_server.py)
# so it stays high-recall; M3 is responsible for gating that down to something
# usable before fusion.
CAMERA_MIN_CONFIDENCE = 0.25

# --- Tracking ---
DEFAULT_DT = 0.05  # matches scenario3.py's fixed_delta_seconds
# Primary association gate: Mahalanobis distance (using each track's own
# Kalman covariance), not a flat Euclidean radius -- a live CARLA run showed
# a fixed 4m Euclidean gate let a faster, unrelated cluster hijack a
# well-established track (see README). TRACK_GATING_CHI2 is the chi-square
# critical value for 3 DOF at ~95% confidence; TRACK_GATING_MAX_DIST_M is a
# hard absolute-distance safety cap so a very uncertain coasting track can't
# match something absurdly far away just because its covariance ballooned.
TRACK_GATING_CHI2 = 7.815
TRACK_GATING_MAX_DIST_M = 8.0
TRACK_CONFIRM_HITS = 3
TRACK_MAX_MISSES = 6
TRACK_MAX_AGE_S = 3.0
PROCESS_NOISE_STD = 1.5
# Run #2 finding: 0.4 (16cm std) models a LiDAR cluster centroid as far more
# precise than it actually is for a large/extended object (a curb, a
# building edge, a parked vehicle silhouette) -- different subsets of points
# return on every sweep, easily shifting the centroid by 1m+ between frames
# even when the object hasn't moved at all. With R this tight, the Kalman
# gain trusts a single measurement so heavily that P collapses to near-R
# within 1-2 hits, which then makes Pass 1's Mahalanobis gate (config.
# TRACK_GATING_CHI2) reject the SECOND observation of the very same static
# clutter it just started tracking -- before the object could ever
# accumulate enough hits to be classified `is_static` and reach the Pass-2
# cascade. This was the actual root cause behind "cascade essentially never
# fires" (0.7% of tracks) on live data: a chicken-and-egg gate collapse, not
# a cascade-radius problem. Raised 0.4 -> 0.8 to better match real
# clustering noise; this trades a little position snappiness for objects
# with genuinely tight LiDAR returns (compact pedestrians, nearby vehicles)
# for a much higher chance that a real re-observation of the same object
# actually re-associates.
POSITION_MEASUREMENT_STD = 0.8

# --- Unknown-class cascade (second association pass) ---
# Pass 2 in tracker.py gives any *still-unmatched* "unknown"-class detection
# a second, more permissive shot at re-associating with any *still-unmatched*
# "unknown"-class track (radius = STATIC_CASCADE_DIST_FACTOR *
# TRACK_GATING_MAX_DIST_M), to absorb ordinary LiDAR clustering jitter on
# clutter (curbs, potholes, building edges, parked vehicles) instead of
# spawning a new track_id every few frames.
#
# Run #2 finding: this pass was originally gated on `track.is_static`
# instead of plain class -- which requires TRACK_CONFIRM_HITS (3) hits
# already accumulated, so a track had to survive Pass 1 long enough to PROVE
# itself static before it could ever benefit from the pass meant to help it
# survive Pass 1. On live data only 0.7% of tracks ever got there (median
# track lifespan 13.5 frames, most dying before hit #3). Gating on class
# instead removes that chicken-and-egg dependency: an "unknown" LiDAR-only
# track is inherently in the class of things clustering jitter affects,
# whether or not it's proven stationary yet. A genuinely-moving unclassified
# object (an unclassified animal, say) isn't hurt by this -- its predicted
# position already tracks its velocity, so it keeps matching in the tighter
# Pass 1 and never needs the cascade. The cascade radius factor was lowered
# 2.0 -> 1.5 alongside this change: the old radius's safety argument ("it's
# proven static, it literally cannot have moved far") no longer fully holds
# once eligibility is broadened to any not-yet-classified track, so the
# radius trades some of that margin back for a lower false-merge risk
# between two distinct nearby unknown obstacles (e.g. dense-market clutter).
#
# `is_static` itself (Track.refresh_static_flag) is kept as-is and still
# reported to M4/M5 (see README) -- it's just no longer the tracker's own
# gate for its cascade.
STATIC_DISPLACEMENT_THRESHOLD_M = 0.75
STATIC_SPEED_THRESHOLD_MPS = 0.5
STATIC_CASCADE_DIST_FACTOR = 1.5

# Run #3 finding: broadening the cascade to any "unknown" track (above) fixed
# fragmentation, but the cascade itself does the match on flat Euclidean
# distance alone, with no Mahalanobis/velocity check at all -- reopening
# exactly the hijack risk Pass 1's Mahalanobis gate exists to close, just
# through the cascade's back door. Observed live: a pure-LiDAR "unknown"
# track jumped 0.29 -> 14.82 -> 29.8 m/s across 3 frames after a fast,
# unrelated cluster (not the original slow/static object) fell within the
# cascade radius and got a full-trust Kalman position update.
#
# A hard dt-normalized "implied speed" cutoff (reject if displacement/dt
# exceeds some plausible-for-clutter threshold) was considered and rejected:
# genuine clustering jitter for a large/extended static object CAN itself
# be several meters within a single ~0.1-0.3s tick (that's the entire reason
# the cascade radius is this generous), so a fixed speed threshold can't
# distinguish "big one-off jitter, still the same static thing" from "a
# different fast object happened to land nearby" using distance/dt alone --
# both look identical to that test in a single frame.
#
# Instead, kalman_filter.KalmanFilter6D.update() takes this as an override R
# for cascade-origin position updates specifically: since a cascade match
# already failed the statistical Pass-1 test, it IS a genuinely less
# certain measurement, so the Kalman gain should be small regardless of
# where it lands within the radius. A wrong one-off match then only nudges
# position slightly and barely touches velocity (which moves through the
# small position-velocity cross-covariance, not directly); a real recurring
# static object's cluster still slowly converges over repeated
# re-associations, the same way a Kalman filter is supposed to average down
# noisy measurements. This degrades gracefully instead of needing a brittle
# distance/dt cutoff.
CASCADE_POSITION_MEASUREMENT_STD = 3.0

# --- Downstream (M4) ---
M4_DOWNSTREAM_URL = "http://127.0.0.1:9500/api/v1/tracks"
M4_TIMEOUT_SECONDS = 5.0

# --- Server ---
HOST = "0.0.0.0"
PORT = 9000

LOG_DIR = "logs"
TRACKS_LOG_FILE = "m3_tracks.jsonl"
