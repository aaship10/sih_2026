"""Hand-computed cases for costmap.py's EDT-based risk-zone classification."""
import math

import config
from costmap import GREEN, NONE, RED, YELLOW, Costmap, anisotropic_offset, classify_distance, worst_risk


def test_classify_distance_zone_boundaries():
    assert classify_distance(0.0) == RED
    assert classify_distance(config.COSTMAP_RED_RADIUS_M) == RED
    assert classify_distance(config.COSTMAP_RED_RADIUS_M + 0.01) == YELLOW
    assert classify_distance(config.COSTMAP_YELLOW_RADIUS_M) == YELLOW
    assert classify_distance(config.COSTMAP_GREEN_RADIUS_M) == GREEN
    assert classify_distance(config.COSTMAP_GREEN_RADIUS_M + 0.01) == NONE


def test_point_at_obstacle_is_red():
    cm = Costmap(center_x=0.0, center_y=0.0)
    cm.mark_obstacle(10.0, 0.0)
    assert cm.risk_at(10.0, 0.0) == RED


def test_point_far_from_obstacle_is_none():
    cm = Costmap(center_x=0.0, center_y=0.0)
    cm.mark_obstacle(10.0, 0.0)
    assert cm.risk_at(-30.0, -30.0) == NONE


def test_empty_costmap_is_all_none():
    cm = Costmap(center_x=0.0, center_y=0.0)
    assert cm.risk_at(0.0, 0.0) == NONE
    assert cm.distance_to_nearest_obstacle_m(5.0, 5.0) == float("inf")


def test_distance_roughly_matches_geometry():
    cm = Costmap(center_x=0.0, center_y=0.0, resolution_m=0.25)
    cm.mark_obstacle(10.0, 0.0)
    d = cm.distance_to_nearest_obstacle_m(6.0, 0.0)
    assert abs(d - 4.0) < 0.5  # within one grid cell's worth of quantization error


def test_footprint_radius_reduces_effective_clearance():
    cm = Costmap(center_x=0.0, center_y=0.0)
    cm.mark_obstacle(10.0, 0.0)
    # 8m away from a wide (2m radius) footprint has 6m clearance -> still GREEN,
    # but a much wider (7m radius) footprint has -1m clearance -> RED.
    assert cm.risk_at(2.0, 0.0, footprint_radius_m=2.0) == GREEN
    assert cm.risk_at(2.0, 0.0, footprint_radius_m=7.0) == RED


def test_worst_risk_picks_most_severe():
    assert worst_risk([NONE, GREEN, YELLOW]) == YELLOW
    assert worst_risk([RED, NONE]) == RED
    assert worst_risk([]) == NONE


def test_anisotropic_offset_identity_when_facing_plus_x():
    # heading=0 (facing +x): along=dx, lateral=dy, no stretch distortion when stretch=1.
    along, lateral = anisotropic_offset(3.0, 4.0, heading_rad=0.0, lateral_stretch=1.0)
    assert math.isclose(along, 3.0)
    assert math.isclose(lateral, 4.0)


def test_anisotropic_offset_stretches_only_lateral():
    along, lateral = anisotropic_offset(3.0, 4.0, heading_rad=0.0, lateral_stretch=2.5)
    assert math.isclose(along, 3.0)          # along-heading component untouched
    assert math.isclose(lateral, 10.0)       # lateral component scaled by 2.5


def test_anisotropic_offset_respects_heading_rotation():
    # facing +y (90 deg): what used to be "ahead" (+y) is now along; what
    # used to be lateral (+x) is now the stretched axis.
    along, lateral = anisotropic_offset(4.0, 3.0, heading_rad=math.pi / 2, lateral_stretch=2.0)
    assert math.isclose(along, 3.0, abs_tol=1e-9)
    assert math.isclose(lateral, -8.0, abs_tol=1e-9)  # magnitude 4.0 * stretch 2.0, sign is arbitrary/unused elsewhere


def test_lateral_obstacle_reads_farther_than_same_distance_ahead():
    # Live-testing finding this fixes: a roadside cone/sign reads as lower
    # risk than the same physical distance directly ahead.
    stretch = config.FOOTPRINT_LATERAL_STRETCH
    cm_lateral = Costmap(center_x=0.0, center_y=0.0, heading_rad=0.0, lateral_stretch=stretch)
    cm_lateral.mark_obstacle(0.0, 1.8)  # purely to the side (heading is +x)
    cm_ahead = Costmap(center_x=0.0, center_y=0.0, heading_rad=0.0, lateral_stretch=stretch)
    cm_ahead.mark_obstacle(1.8, 0.0)    # same raw distance, directly ahead

    assert cm_lateral.risk_at(0.0, 0.0, config.EGO_FOOTPRINT_RADIUS_M) != RED
    assert cm_ahead.risk_at(0.0, 0.0, config.EGO_FOOTPRINT_RADIUS_M) == RED


def test_isotropic_default_unaffected_by_lateral_stretch_change():
    # Default construction (no heading/stretch args) must stay exactly the
    # old isotropic behavior -- every pre-existing Costmap test above
    # relies on this.
    cm = Costmap(center_x=0.0, center_y=0.0)
    cm.mark_obstacle(0.0, 1.8)
    assert cm.risk_at(0.0, 0.0, config.EGO_FOOTPRINT_RADIUS_M) == RED
