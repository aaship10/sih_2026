"""M4 prediction interface.

M4 is responsible only for short-horizon, multi-modal motion prediction.
Risk/TTC/collision analysis belongs downstream in M5.
"""

from .multimodal import generate_branches

PREDICTION_HORIZON = 3.0
TIME_STEP = 0.2


def predict(track_id, cls, track_history, confidence,
            horizon=PREDICTION_HORIZON, time_step=TIME_STEP):
    """Predict one tracked object's future motion for M5.

    ``track_history`` contains M3 snapshots, oldest -> newest.
    No ego state is required here because M4 no longer computes TTC,
    collision probability, or risk.
    """
    if not track_history:
        raise ValueError(f"predict() called with empty history for track {track_id}")

    latest = track_history[-1]
    position_xy = latest["position"][:2]
    velocity_xy = latest["velocity"][:2]
    size_xyz = latest.get("size") or [0.0, 0.0, 0.0]

    lateral_trend = 0.0
    if len(track_history) >= 2:
        from .derive import estimate_lateral_velocity_trend
        lateral_trend = estimate_lateral_velocity_trend(track_history)

    branches = generate_branches(
        position_xy,
        velocity_xy,
        cls,
        track_history,
        horizon,
        time_step,
        lateral_trend=lateral_trend,
    )

    return {
        "track_id": int(track_id),
        "class": cls,
        "confidence": round(float(confidence), 3),
        "position": [round(float(position_xy[0]), 3), round(float(position_xy[1]), 3)],
        "velocity": [round(float(velocity_xy[0]), 3), round(float(velocity_xy[1]), 3)],
        "size": [
            round(float(size_xyz[0]), 3),
            round(float(size_xyz[1]), 3),
            round(float(size_xyz[2]), 3) if len(size_xyz) > 2 else 0.0,
        ],
        "trajectories": [
            {
                "mode": b["mode"],
                "probability": b["probability"],
                "points": [
                    {
                        "x": point[0],
                        "y": point[1],
                        "t": round((i + 1) * time_step, 3),
                    }
                    for i, point in enumerate(b["points"])
                ],
            }
            for b in branches
        ],
    }
