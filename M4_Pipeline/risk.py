"""
risk.py
=======
Combines everything into one normalized risk score in [0, 1] and a
discrete risk level M5 can branch on without knowing any of M4's
internals.

    risk = w1*collision_probability + w2*TTC_risk + w3*distance_risk
         + w4*uncertainty_risk      + w5*vulnerability

All terms normalized to [0, 1] before combining. Starting weights below
are a reasonable prior, NOT tuned -- Section 21 explicitly asks for
these to be tuned using CARLA experiments (Phase 12/19-20 in the
roadmap): run the 5 required scenarios, check whether risk_level
matches human judgment of "should the car actually react here", and
adjust weights/thresholds accordingly. Treat the numbers below as v0.

Term definitions:
    collision_probability -- from collision.py, already in [0,1].
    TTC_risk               -- 1.0 when TTC (or time-to-conflict) is
                              ~immediate, decaying to 0 by TTC_SATURATION
                              seconds. Missing/None TTC -> 0 (no
                              closing-velocity or spatial-overlap
                              conflict detected).
    distance_risk           -- 1.0 at combined_radius (already touching),
                              decaying to 0 by DISTANCE_SATURATION meters.
    uncertainty_risk        -- normalized mean predicted sigma (wider
                              spread = harder to plan around safely,
                              independent of whether the mean prediction
                              collides).
    vulnerability           -- object-class risk factor (Section 22):
                              pedestrians/cyclists weighted highest
                              (physical vulnerability + unpredictability),
                              NOT an ethical/prioritization judgment --
                              purely "how much does uncertainty about
                              this class's motion matter physically."
"""

TTC_SATURATION = 4.0          # seconds -- TTC/time-to-conflict beyond this contributes ~0 risk
DISTANCE_SATURATION = 8.0     # meters -- min-distance beyond this contributes ~0 risk
UNCERTAINTY_SATURATION = 3.0  # meters of sigma -- beyond this, uncertainty term saturates at 1.0

VULNERABILITY = {
    "pedestrian": 1.0,
    "bicycle": 0.9,
    "motorcycle": 0.8,
    "animal": 0.85,
    "rickshaw": 0.5,        # was "auto_rickshaw" -- must match classes.yaml exactly
    "car": 0.4,
    "tempo": 0.35,          # was missing entirely
    "bus": 0.3,
    "truck": 0.3,
    "traffic_cones": 0.3,   # was missing -- lightweight, low physical vulnerability if struck
    "pothole": 0.2,
    "speed_bumps": 0.2,     # was "speed_bump" -- must match classes.yaml exactly (plural)
    "road_sign": 0.6,       # was missing -- rigid pole, real collision consequence
    "traffic_signal": 0.6,  # was missing -- same reasoning as road_sign
}
DEFAULT_VULNERABILITY = 0.5

WEIGHTS = {
    "collision_probability": 0.35,
    "ttc": 0.25,
    "distance": 0.15,
    "uncertainty": 0.10,
    "vulnerability": 0.15,
}

# Section 20: tune these against the 5 CARLA scenarios, not blindly.
RISK_THRESHOLDS = {
    "LOW": 0.25,
    "MEDIUM": 0.5,
    "HIGH": 0.75,
    # >= 0.75 -> CRITICAL
}


def _saturating_risk(value, saturation, invert=True):
    """Linear ramp from 1.0 at value=0 down to 0.0 at value=saturation (invert=True),
    clamped to [0, 1]. Used for "smaller time/distance = higher risk" terms."""
    if value is None:
        return 0.0
    frac = 1.0 - (value / saturation)
    return max(0.0, min(1.0, frac)) if invert else max(0.0, min(1.0, value / saturation))


def compute_risk_score(collision_probability, ttc_or_conflict_seconds, min_distance,
                        mean_uncertainty_sigma, obj_class):
    ttc_risk = _saturating_risk(ttc_or_conflict_seconds, TTC_SATURATION)
    distance_risk = _saturating_risk(min_distance, DISTANCE_SATURATION)
    uncertainty_risk = _saturating_risk(mean_uncertainty_sigma, UNCERTAINTY_SATURATION, invert=False)
    vulnerability = VULNERABILITY.get(obj_class, DEFAULT_VULNERABILITY)

    score = (
        WEIGHTS["collision_probability"] * collision_probability
        + WEIGHTS["ttc"] * ttc_risk
        + WEIGHTS["distance"] * distance_risk
        + WEIGHTS["uncertainty"] * uncertainty_risk
        + WEIGHTS["vulnerability"] * vulnerability
    )
    return round(max(0.0, min(1.0, score)), 3)


def risk_level_from_score(score):
    if score < RISK_THRESHOLDS["LOW"]:
        return "LOW"
    if score < RISK_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    if score < RISK_THRESHOLDS["HIGH"]:
        return "HIGH"
    return "CRITICAL"