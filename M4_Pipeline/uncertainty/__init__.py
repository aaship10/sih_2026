"""Uncertainty representation (section 14). Chosen primary representation:
a growing along-track / cross-track Gaussian standard-deviation pair per
predicted timestep, oriented by the trajectory's heading -- i.e. an
axis-aligned-in-trajectory-frame confidence ellipse at each sample, not a
single fixed-size blob for the whole trajectory. See covariance.py's
docstring for why this one representation was picked over the alternatives
section 14 lists."""
from uncertainty.covariance import apply_uncertainty, normalized_spread

__all__ = ["apply_uncertainty", "normalized_spread"]
