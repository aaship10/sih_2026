"""
uncertainty.py
==============
Section 14's "recommend ONE primary representation": a single growing
Gaussian (isotropic sigma per predicted timestep), per the core design
decision --

    sigma(t)^2 = sigma0^2 + k^2 * t^2

-- rather than a full anisotropic covariance matrix, multiple sampled
trajectory sets, or a per-branch uncertainty. One scalar sigma per
timestep is enough to (a) draw a confidence-circle/occupancy region
per predicted point (Section 15) and (b) feed a single uncertainty
term into risk.py, without the bookkeeping of a 2x2 covariance that
this heuristic (non-learned) predictor can't actually justify the
extra precision of.

sigma0 (uncertainty AT t=0) comes from M3's own track confidence --
a track M3 is less sure about starts every rollout already fuzzier
(confidence_to_sigma0). The GROWTH RATE k is class-dependent
(motion_models.CLASS_PARAMS["uncertainty_k"]) -- Section 33/22:
pedestrians and especially animals get uncertain fast (they can
change direction on a dime), trucks/buses grow slowly (large,
momentum-bound, hard to redirect quickly).

occupancy_radius() turns a sigma into the single scalar radius
Section 15's "possible region" needs for visualization/downstream
use, using a 2-sigma (~95%, for a 2D isotropic Gaussian) circle --
conservative enough for a safety-relevant system without ballooning
into an unusably large region by the end of the horizon.
"""

# --- sigma0: baseline uncertainty at t=0, from M3's track confidence ---
SIGMA0_MIN = 0.15   # meters -- floor even for a fully-confident (confidence=1.0) track;
                     # M3's own position/velocity noise never truly hits zero
SIGMA0_MAX = 1.00   # meters -- ceiling at confidence=0.0 (M3 essentially unsure this is real)

# --- occupancy circle: how many sigmas to inflate to for a "region" radius ---
OCCUPANCY_N_SIGMA = 2.0  # ~95% coverage for an isotropic 2D Gaussian


def confidence_to_sigma0(confidence):
    """
    Linear interpolation: confidence=1.0 -> SIGMA0_MIN, confidence=0.0
    -> SIGMA0_MAX. M3's confidence is already smoothed (0.9/0.1 blend
    per the integration notes) so no further smoothing needed here --
    just a direct mapping to a starting spread.
    """
    confidence = max(0.0, min(1.0, confidence))
    return SIGMA0_MAX - confidence * (SIGMA0_MAX - SIGMA0_MIN)


def trajectory_uncertainty(cls, confidence, horizon, time_step):
    """
    Returns a list of sigma (meters), one per predicted timestep
    (t = time_step, 2*time_step, ..., horizon), matching the same
    indexing as motion_models.constant_velocity_rollout()'s points --
    sigmas[i] is the uncertainty of trajectory point i.

    sigma(t) = sqrt(sigma0^2 + k^2 * t^2), k = class's uncertainty_k
    from motion_models.CLASS_PARAMS.
    """
    # local import to avoid a module-load cycle (motion_models doesn't
    # import uncertainty, but keeping this import here mirrors how
    # multimodal.py also pulls class params from motion_models)
    from motion_models import get_class_params

    k = get_class_params(cls)["uncertainty_k"]
    sigma0 = confidence_to_sigma0(confidence)

    n_steps = max(1, round(horizon / time_step))
    sigmas = []
    for i in range(1, n_steps + 1):
        t = i * time_step
        sigma_t = (sigma0 ** 2 + (k * t) ** 2) ** 0.5
        sigmas.append(round(sigma_t, 3))
    return sigmas


def occupancy_radius(sigma, n_sigma=OCCUPANCY_N_SIGMA):
    """Single scalar 'possible region' radius (meters) for one predicted
    point's sigma -- Section 15's P(object at x,y,t) represented as a
    circle of this radius centered on the predicted point."""
    return round(n_sigma * sigma, 3)