"""M4 configuration: class-specific prediction parameters, uncertainty growth
rates, server ports.

Kept as a single flat module, matching M2_Pipeline/M3_Pipeline's config.py
style, rather than section 47's suggested config/ package -- there's no
per-component config large enough to need splitting yet.

M4 deliberately does NOT compute risk (TTC, time-to-conflict, collision
probability, a risk score/level) -- that was cut from scope; see README.md's
"Status" section. Everything below is prediction/uncertainty config only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --- Server ---
HOST = "0.0.0.0"
PORT = 9500  # M3's config.M4_DOWNSTREAM_URL points here; must not change
              # without also updating M3_Pipeline/config.py (M3 is frozen).

# --- Downstream (M5) ---
# M5's own design doc (M5_Pipeline/README.md) specifies UDP, not HTTP, on
# port 5004 -- this replaces the earlier proposed HTTP/port-10000 contract
# (output/schema.py's PredictionPacket is still used internally for
# recording/replay/evaluation; only the wire format actually sent
# downstream changed). UDP fire-and-forget needs no timeout: a slow or
# absent M5 simply never sees the packet, with no socket to hang M4 on.
M5_UDP_HOST = "127.0.0.1"
M5_UDP_PORT = 5004

# --- Recorder ---
LOG_DIR = "logs"
INPUT_PACKETS_LOG_FILE = "m4_input_packets.jsonl"   # raw M3 packets, verbatim
PREDICTIONS_LOG_FILE = "m4_predictions.jsonl"        # M4's own output, for eval/replay-diff

# --- Track history ---
# M3's own track_ids churn (~1.4 new ids/frame on live data, mostly
# short-lived "unknown" clutter -- see M3 README). History buffers must be
# bounded and pruned, not grow with the run.
HISTORY_MAX_LEN = 30          # samples kept per track (most classes only need ~1-2s of history)
TRACK_STALE_SECONDS = 2.0     # drop a track's history if not updated for this long (independent of M3's own TRACK_MAX_AGE_S=3.0 -- M4 prunes its OWN buffers slightly earlier so dead tracks don't linger in M4's memory after M3 has effectively abandoned them)

# --- Input quality ---
# Used by tracking_input/quality.py to score how much to trust a track's
# CURRENT state (not its class-confidence, which is M2's YOLO score and
# unrelated to tracking quality -- see M3 README 7B.4).
QUALITY_MIN_HITS_FOR_FULL_TRUST = 6     # tracks need meaningfully more than the bare hits>=3 confirmation threshold before being fully trusted
QUALITY_COASTING_PENALTY_PER_SEC = 0.6  # quality multiplier decay per second of time_since_update>0 (coasting / dead-reckoned)
QUALITY_SENSOR_SCORES = {
    frozenset(["camera", "lidar", "radar"]): 1.0,
    frozenset(["camera", "lidar"]): 0.9,
    frozenset(["lidar", "radar"]): 0.7,
    frozenset(["camera", "radar"]): 0.55,
    frozenset(["lidar"]): 0.5,
    frozenset(["camera"]): 0.35,   # camera-only: ground-plane back-projection or radar match, weakest position quality (M3 README 7B.3)
}
QUALITY_DEFAULT_SENSOR_SCORE = 0.3  # any combination not listed above (shouldn't normally occur)

# --- Classes ---
# Full taxonomy from M2_Pipeline/classes.yaml, plus M3's "unknown" fallback
# (LiDAR-only clusters with no camera classification). Static/environmental
# classes get no motion prediction (STATIC_CLASSES below) since they cannot
# move; M4 still reports them as fixed obstacles.
STATIC_CLASSES = frozenset(["road_sign", "traffic_signal", "speed_bumps", "traffic_cones", "pothole"])

# NOTE on "pushcart": the original problem statement lists pushcarts as a
# road user, but classes.yaml has NO "pushcart" class, and M1's scenario
# files spawn pushcarts using RICKSHAW_STANDIN (a small vehicle mesh) --
# M2's YOLO will classify a pushcart as whatever visual class it resembles
# (most likely "rickshaw", "car", or "unknown"), never literally "pushcart".
# M4 has no special handling for a class M3 can never actually send.


@dataclass(frozen=True)
class ClassProfile:
    horizon_s: float          # how far ahead to predict for this class
    time_step_s: float        # sampling interval within that horizon
    along_sigma0_m: float     # initial (t=0) along-track position std-dev, meters
    lateral_sigma0_m: float   # initial cross-track position std-dev, meters
    along_growth_mps: float   # along-track std-dev growth rate (m per second of horizon)
    lateral_growth_mps: float # cross-track std-dev growth rate -- HIGHER for classes that can turn/cross unpredictably
    max_turn_rate_dps: float  # for the multimodal "lateral deviation" mode: max heading change assumed possible, degrees/second
    stop_mode_decel_mps2: float  # deceleration assumed for the multimodal "stop" mode
    mode_probabilities: dict  # base probabilities for {"nominal","stop","lateral"} before any per-track adjustment


# Horizon/timestep reasoning (section 9): pedestrians/animals need a SHORT,
# frequently-resampled horizon because they can change mind/direction in
# under a second and a long constant-velocity projection would be
# meaningless past ~2-3s; cars get a longer horizon because their motion is
# far more constrained (can't instantly reverse direction) so a longer
# look-ahead is still informative; trucks/buses get the longest horizon
# because their own stopping distance is large, so M5 needs the earliest
# possible warning even though their motion is the most predictable of all.
CLASS_PROFILES: dict[str, ClassProfile] = {
    "pedestrian": ClassProfile(3.0, 0.2, 0.3, 0.3, 0.5, 0.9, 90.0, 2.0,
                                {"nominal": 0.55, "stop": 0.25, "lateral": 0.20}),
    "animal": ClassProfile(2.5, 0.2, 0.4, 0.4, 0.6, 1.1, 120.0, 1.5,
                            {"nominal": 0.40, "stop": 0.25, "lateral": 0.35}),
    "bicycle": ClassProfile(3.0, 0.2, 0.3, 0.3, 0.6, 0.8, 60.0, 2.5,
                             {"nominal": 0.60, "stop": 0.15, "lateral": 0.25}),
    "motorcycle": ClassProfile(3.0, 0.2, 0.4, 0.4, 0.9, 0.9, 45.0, 3.5,
                                {"nominal": 0.55, "stop": 0.10, "lateral": 0.35}),
    "rickshaw": ClassProfile(3.0, 0.2, 0.4, 0.3, 0.7, 0.6, 30.0, 3.0,
                              {"nominal": 0.65, "stop": 0.15, "lateral": 0.20}),
    "car": ClassProfile(4.0, 0.3, 0.4, 0.25, 0.7, 0.35, 15.0, 4.0,
                         {"nominal": 0.80, "stop": 0.10, "lateral": 0.10}),
    "tempo": ClassProfile(4.0, 0.3, 0.45, 0.25, 0.7, 0.3, 12.0, 4.0,
                           {"nominal": 0.80, "stop": 0.10, "lateral": 0.10}),
    "truck": ClassProfile(5.0, 0.3, 0.5, 0.2, 0.6, 0.2, 8.0, 3.0,
                           {"nominal": 0.85, "stop": 0.10, "lateral": 0.05}),
    "bus": ClassProfile(5.0, 0.3, 0.5, 0.2, 0.6, 0.2, 8.0, 3.0,
                         {"nominal": 0.85, "stop": 0.10, "lateral": 0.05}),
    # "unknown": no class info at all (LiDAR-only cluster, M3 README 7B.4) --
    # short horizon, wide uncertainty growth on BOTH axes since we cannot
    # even guess whether it behaves like a vehicle or a pedestrian.
    "unknown": ClassProfile(2.0, 0.2, 0.5, 0.5, 0.7, 0.7, 60.0, 2.5,
                             {"nominal": 0.55, "stop": 0.20, "lateral": 0.25}),
}
DEFAULT_CLASS_PROFILE = CLASS_PROFILES["unknown"]  # any class M2 adds that M4 doesn't yet special-case

# --- Uncertainty / confidence ---
# Ellipse semi-major axis (evaluated at the end of a trajectory's own
# horizon, see uncertainty/covariance.py's normalized_spread) beyond which
# the uncertainty-derived confidence penalty saturates at its maximum.
UNCERTAINTY_MAX_M = 4.0

# --- Multimodal ---
N_MODES = 3  # nominal, stop, lateral -- see multimodal/modes.py

# Windowed motion features (multimodal/features.py): how many recent
# MEASURED history samples to average for a smoothed velocity and to derive
# speed/heading volatility from. Too short doesn't suppress noise; too long
# lags real acceleration (defeats the point of the CA model). 5 samples at
# M3's ~6-20Hz effective rate is roughly 0.25-0.8s of real history.
MOTION_FEATURE_WINDOW = 5
# Below this many measured samples, fuzzy mode-weighting (multimodal/fuzzy.py)
# and velocity smoothing fall back to today's static per-class defaults +
# crisp slow-speed threshold -- mirrors constant_acceleration.py's own
# MIN_SAMPLES_FOR_ACCELERATION fallback pattern: don't trust a
# volatility/smoothing computation built from 1-2 points.
MIN_SAMPLES_FOR_FUZZY = 3
