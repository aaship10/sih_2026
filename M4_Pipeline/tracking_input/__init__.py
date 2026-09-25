"""Adapter layer between M3's raw wire format and M4's internal prediction
core (predictor.py). Nothing outside this package should import M3's field
names directly -- the rest of M4 only ever sees the internal shapes defined
in adapter.py (TrackedObject / EgoState)."""
