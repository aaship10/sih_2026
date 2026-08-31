"""
derive.py
=========
M3 doesn't send heading or acceleration (see the integration notes) --
this file derives both from M4's own rolling buffer (buffer.py) of raw
M3 snapshots, plus one extra derived signal (lateral velocity trend)
that multimodal.py uses to decide how much to upweight the "crossing"
branch. All three functions take `track_history` in buffer.py's format:
oldest -> newest, most-recent-last.
"""

import math

HEADING_SPEED_THRESHOLD = 0.2   # m/s -- below this, atan2(vy, vx) is
                                 # dominated by sensor/Kalman noise, not
                                 # real direction of travel (a stopped
                                 # pedestrian's [vx, vy] jitters near
                                 # zero and would otherwise spin the
                                 # heading randomly frame to frame --
                                 # exactly the "garbage heading" the
                                 # integration notes warn about)

LATERAL_TREND_LOOKBACK = 5      # frames to look back for the trend


def estimate_heading(track_history):
    """
    heading = atan2(vy, vx) using the most recent frame whose speed is
    above HEADING_SPEED_THRESHOLD, scanning backward through the buffer
    until one is found. This is the "hold last valid heading" fallback
    the integration notes call for -- a pedestrian stopped mid-crossing
    keeps the heading they were walking with, instead of a fresh
    heading recomputed from near-zero velocity.

    Returns 0.0 only if the object has NEVER moved fast enough to have
    a trustworthy heading in the whole buffered window.
    """
    for frame in reversed(track_history):
        vx, vy = frame["velocity"][0], frame["velocity"][1]
        if math.hypot(vx, vy) > HEADING_SPEED_THRESHOLD:
            return math.atan2(vy, vx)
    return 0.0


def estimate_acceleration(track_history):
    """
    a = (v_t - v_{t-1}) / dt, using the two most recent buffered
    frames' OWN timestamps for dt rather than assuming a fixed frame
    rate -- a dropped or delayed frame from M3 would otherwise
    silently corrupt this. Don't ask M3 to add acceleration -- this is
    all the finite difference needs.
    """
    if len(track_history) < 2:
        return [0.0, 0.0]

    prev, latest = track_history[-2], track_history[-1]
    dt = latest["timestamp"] - prev["timestamp"]
    if dt <= 1e-6:
        return [0.0, 0.0]

    ax = (latest["velocity"][0] - prev["velocity"][0]) / dt
    ay = (latest["velocity"][1] - prev["velocity"][1]) / dt
    return [ax, ay]


def estimate_lateral_velocity_trend(track_history, lookback=LATERAL_TREND_LOOKBACK):
    """
    How fast the object's lateral (y, in the ego frame -- M3's
    transforms.py convention: x forward, y left/right) speed has been
    GROWING over the recent window, in m/s per second. Positive means
    "this object's sideways motion is picking up right now" -- exactly
    the per-frame evidence multimodal.py uses to upweight its crossing
    branch (the "if lateral velocity component is growing over the
    last few frames, upweight the crossing branch" design note, and
    Sections 32/33's sudden-pedestrian / cattle-crossing scenarios).

    Uses |vy| (not signed vy) because what matters for "is this turning
    into a crossing" is the MAGNITUDE of sideways motion, not which
    side it's on -- direction is handled separately, in
    multimodal._lateral_sign().

    Returns 0.0 if there isn't enough buffered history yet, or if the
    lookback window collapsed to (near) zero elapsed time.
    """
    hist = track_history[-lookback:]
    if len(hist) < 2:
        return 0.0

    dt = hist[-1]["timestamp"] - hist[0]["timestamp"]
    if dt <= 1e-6:
        return 0.0

    lateral_start = abs(hist[0]["velocity"][1])
    lateral_end = abs(hist[-1]["velocity"][1])
    return (lateral_end - lateral_start) / dt