"""Hand-computed cases for footprint.py's 3-circle ego footprint."""
import math

import config
from costmap import NONE, RED, Costmap
from footprint import circle_centers, footprint_risk


def test_circle_centers_face_along_heading_zero():
    centers = circle_centers(0.0, 0.0, 0.0)
    assert len(centers) == 3
    xs = sorted(c[0] for c in centers)
    assert xs[0] < 0.0 < xs[-1]  # rear circle behind, front circle ahead
    assert all(abs(c[1]) < 1e-9 for c in centers)  # heading 0 -> all on the x-axis


def test_circle_centers_rotate_with_heading():
    centers = circle_centers(0.0, 0.0, math.pi / 2)  # facing +y
    xs = [c[0] for c in centers]
    assert all(abs(x) < 1e-9 for x in xs)  # now all on the y-axis instead


def test_footprint_flags_obstacle_directly_ahead():
    cm = Costmap(center_x=0.0, center_y=0.0)
    cm.mark_obstacle(config.EGO_FOOTPRINT_OFFSETS_M[-1], 0.0)  # exactly at the front circle's own center
    assert footprint_risk(0.0, 0.0, 0.0, cm) == RED


def test_footprint_ignores_obstacle_far_to_the_side():
    cm = Costmap(center_x=0.0, center_y=0.0)
    cm.mark_obstacle(0.0, 30.0)
    assert footprint_risk(0.0, 0.0, 0.0, cm) == NONE
