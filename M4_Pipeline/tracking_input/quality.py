"""Input-quality scoring (M3 README section 7B, design consequence: "an
input-quality layer... gate on hits / time_since_update / sensor_sources,
and pass an input-quality score into prediction confidence and risk").

This is NOT M2's classification confidence (`TrackedObject.confidence`,
M3 README 7B.4) -- it answers a different question: "how much should M4
trust this track's CURRENT position/velocity state", independent of what
class M2 thinks it is.

quality is a single float in [0, 1], the product of three independent
factors:
  - hits_factor:     ramps from the bare M3 confirmation threshold up to 1.0
                      as hits grows (a track with hits=3 was JUST confirmed;
                      M3's own Kalman velocity estimate is still converging
                      from its zero-velocity initialization at that point --
                      see M3 README 7B.1).
  - coasting_factor:  decays with time_since_update -- M3 keeps publishing a
                      track for up to TRACK_MAX_MISSES/TRACK_MAX_AGE_S after
                      its last real measurement, dead-reckoned by the
                      constant-velocity filter (M3 README 7B.2). A coasting
                      track's state is a prediction already, not a
                      measurement -- M4 must not treat it as equally solid.
  - sensor_factor:    from config.QUALITY_SENSOR_SCORES, reflecting the
                      three different position-quality tiers M3's fusion
                      produces (M3 README 7B.3).
"""
from __future__ import annotations

import config
from tracking_input.adapter import TrackedObject


def hits_factor(hits: int) -> float:
    min_hits = config.QUALITY_MIN_HITS_FOR_FULL_TRUST
    if hits <= 3:
        return 3.0 / min_hits if min_hits > 0 else 1.0
    if hits >= min_hits:
        return 1.0
    return hits / min_hits


def coasting_factor(time_since_update: float) -> float:
    if time_since_update <= 0.0:
        return 1.0
    decay = config.QUALITY_COASTING_PENALTY_PER_SEC
    return max(0.0, 1.0 - decay * time_since_update)


def sensor_factor(sensor_sources: frozenset[str]) -> float:
    return config.QUALITY_SENSOR_SCORES.get(sensor_sources, config.QUALITY_DEFAULT_SENSOR_SCORE)


def compute_quality(obj: TrackedObject) -> float:
    q = hits_factor(obj.hits) * coasting_factor(obj.time_since_update) * sensor_factor(obj.sensor_sources)
    return max(0.0, min(1.0, q))


def annotate(obj: TrackedObject) -> TrackedObject:
    """Mutates and returns `obj` with `.quality` filled in -- called once per
    object per frame by predictor.py before anything else touches it."""
    obj.quality = compute_quality(obj)
    return obj
