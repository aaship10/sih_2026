"""Small history-derived signal used by the multi-modal predictor."""


LATERAL_TREND_LOOKBACK = 5


def estimate_lateral_velocity_trend(track_history, lookback=LATERAL_TREND_LOOKBACK):
    """Return growth of |lateral velocity| over the recent history."""
    hist = track_history[-lookback:]
    if len(hist) < 2:
        return 0.0

    dt = hist[-1]["timestamp"] - hist[0]["timestamp"]
    if dt <= 1e-6:
        return 0.0

    lateral_start = abs(hist[0]["velocity"][1])
    lateral_end = abs(hist[-1]["velocity"][1])
    return (lateral_end - lateral_start) / dt
