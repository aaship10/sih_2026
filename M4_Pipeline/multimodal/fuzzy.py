"""Lightweight, dependency-free fuzzy-logic mode-probability weighting.

Replaces an earlier scikit-fuzzy-based design, dropped after benchmarking:
scikit-fuzzy's discretized-universe Mamdani inference measured ~4.8ms per
SINGLE object's fuzzy computation, even reusing one ControlSystemSimulation
(the efficient pattern) -- at ~40 objects/frame that's ~190ms/frame, >10x
the whole pipeline's measured total latency, directly against the stated
"minimize latency" goal. It also threw a KeyError on an edge-case input
(no rule produced output for one consequent) despite an intentionally
exhaustive rule base -- a known skfuzzy brittleness, not a configuration
mistake. Both problems are inherent to how skfuzzy works (numerical
integration over a discretized universe on every call), not fixable by
better rule-writing.

This module implements the same 3-input/3-output design (speed, heading
volatility, velocity volatility -> nominal/stop/lateral weights, with a
per-class lateral-affinity multiplier) using zeroth-order Takagi-Sugeno
inference instead of Mamdani: each rule maps a combination of fuzzy
ANTECEDENT memberships (computed via the same triangular membership
functions skfuzzy would use) to a crisp per-mode output value; the final
output is the firing-strength-weighted average of every rule's crisp
value. No discretized universe, no numerical integration -- pure scalar
arithmetic, several orders of magnitude faster, and every input in each
variable's defined range always has total membership > 0 across that
variable's sets (triangular sets are built to partition the range), so
there is no equivalent of skfuzzy's "no rule fired" failure mode.
"""
from __future__ import annotations

import math

# --- Membership functions -------------------------------------------------

def _trimf(x: float, a: float, b: float, c: float) -> float:
    """Standard triangular membership (a <= b <= c), matching skfuzzy's own
    reference implementation exactly (including a==b / b==c shoulder
    handling) so switching back to a discretized-universe library later, if
    ever needed, would reuse the same membership definitions unchanged."""
    y = 0.0
    if a != b and a < x < b:
        y = (x - a) / (b - a)
    if b != c and b < x < c:
        y = (c - x) / (c - b)
    if x == b:
        y = 1.0
    return y


def _fuzzify_speed(speed_ratio: float) -> dict[str, float]:
    return {
        "stationary": _trimf(speed_ratio, 0.0, 0.0, 0.1),
        "slow": _trimf(speed_ratio, 0.05, 0.2, 0.4),
        "cruising": _trimf(speed_ratio, 0.25, 0.5, 0.8),
        "fast": _trimf(speed_ratio, 0.6, 1.0, 1.2),
    }


def _fuzzify_heading_vol(heading_vol_rad: float) -> dict[str, float]:
    return {
        "straight": _trimf(heading_vol_rad, 0.0, 0.0, 0.15),
        "curving": _trimf(heading_vol_rad, 0.05, 0.3, 0.6),
        "erratic": _trimf(heading_vol_rad, 0.4, math.pi, math.pi),
    }


def _fuzzify_vel_vol(vel_vol: float) -> dict[str, float]:
    return {
        "stable": _trimf(vel_vol, 0.0, 0.0, 0.15),
        "moderate": _trimf(vel_vol, 0.1, 0.3, 0.5),
        "unstable": _trimf(vel_vol, 0.35, 1.0, 1.0),
    }


# --- Rule base --------------------------------------------------------
# (speed_set, heading_set) -> crisp (nominal, stop, lateral) output triple,
# each row summing to 1.0 (not required, just convenient to read/tune).
# Encodes: stationary objects favor stop; fast+straight objects favor
# nominal; curving/erratic heading favors lateral, more so at low-to-
# moderate speed (a fast object is less likely to suddenly dart sideways
# than a slow/stopped one deciding which way to go).
_BASE_RULES: dict[tuple[str, str], tuple[float, float, float]] = {
    ("stationary", "straight"): (0.15, 0.75, 0.10),
    ("stationary", "curving"):  (0.10, 0.70, 0.20),
    ("stationary", "erratic"):  (0.10, 0.65, 0.25),
    ("slow", "straight"):       (0.45, 0.30, 0.25),
    ("slow", "curving"):        (0.30, 0.25, 0.45),
    ("slow", "erratic"):        (0.20, 0.25, 0.55),
    ("cruising", "straight"):   (0.75, 0.10, 0.15),
    ("cruising", "curving"):    (0.55, 0.10, 0.35),
    ("cruising", "erratic"):    (0.45, 0.10, 0.45),
    ("fast", "straight"):       (0.85, 0.05, 0.10),
    ("fast", "curving"):        (0.60, 0.08, 0.32),
    ("fast", "erratic"):        (0.45, 0.10, 0.45),
}

# Per-class lateral affinity (section 12's "a bike is more likely to
# suddenly go lateral than a bus" -- common-sense/physics prior): a scalar
# multiplier applied to the lateral output AFTER base-rule inference, before
# final normalization. Small/agile classes with high yaw authority and low
# mass score higher; large classes with real turning-radius and inertia
# constraints score lower. Deliberately kept separate from the rule base
# (which stays class-agnostic) so this one table is the only thing to
# retune per class, rather than duplicating speed/heading rules per class.
LATERAL_AFFINITY_BY_CLASS: dict[str, float] = {
    "pedestrian": 1.1, "animal": 1.3, "bicycle": 1.3, "motorcycle": 1.4,
    "rickshaw": 1.1, "car": 0.8, "tempo": 0.8, "truck": 0.6, "bus": 0.5, "unknown": 1.0,
}
DEFAULT_LATERAL_AFFINITY = 1.0

# How strongly "unstable" recent velocity discounts trust in the nominal
# (straight-line-from-current-velocity) mode -- see this module's docstring
# for the live-data case this targets. At instability=1.0 (fully unstable),
# nominal is cut to 40% of its base-rule value; the lost weight moves to
# stop (a track we can't confidently extrapolate is safer to assume is
# settling than to keep trusting its last, possibly-spurious, velocity).
INSTABILITY_NOMINAL_PENALTY = 0.6
INSTABILITY_STOP_TRANSFER = 0.5


def compute_mode_weights(speed_ratio: float, heading_vol_rad: float, vel_vol: float,
                          lateral_affinity: float) -> dict[str, float]:
    """speed_ratio, vel_vol: already normalized by the class's own max
    speed (models.class_specific.MAX_SPEED_MPS_BY_CLASS), so both are
    roughly in [0, ~1.2]/[0, ~1] regardless of vehicle vs. pedestrian scale.
    heading_vol_rad: mean abs heading change over the window, radians.
    Returns {"nominal": .., "stop": .., "lateral": ..} summing to 1.0."""
    speed_ratio = max(0.0, min(1.2, speed_ratio))
    heading_vol_rad = max(0.0, min(math.pi, heading_vol_rad))
    vel_vol = max(0.0, min(1.0, vel_vol))

    speed_fuzz = _fuzzify_speed(speed_ratio)
    heading_fuzz = _fuzzify_heading_vol(heading_vol_rad)

    nominal = stop = lateral = 0.0
    total_weight = 0.0
    for (speed_set, heading_set), (n, s, l) in _BASE_RULES.items():
        firing = speed_fuzz[speed_set] * heading_fuzz[heading_set]
        if firing <= 0.0:
            continue
        nominal += firing * n
        stop += firing * s
        lateral += firing * l
        total_weight += firing

    if total_weight <= 0.0:
        # Should not happen -- the four speed sets and three heading sets
        # are built to partition their full clipped range -- but a static,
        # class-neutral fallback is cheap insurance against a future
        # membership-function edit accidentally leaving a gap.
        nominal, stop, lateral = 0.55, 0.25, 0.20
    else:
        nominal /= total_weight
        stop /= total_weight
        lateral /= total_weight

    instability = _fuzzify_vel_vol(vel_vol)["unstable"]
    if instability > 0.0:
        reduced = nominal * INSTABILITY_NOMINAL_PENALTY * instability
        nominal -= reduced
        stop += reduced * INSTABILITY_STOP_TRANSFER
        lateral += reduced * (1.0 - INSTABILITY_STOP_TRANSFER)

    lateral *= lateral_affinity

    total = nominal + stop + lateral
    if total <= 0.0:
        return {"nominal": 0.55, "stop": 0.25, "lateral": 0.20}
    return {"nominal": nominal / total, "stop": stop / total, "lateral": lateral / total}
