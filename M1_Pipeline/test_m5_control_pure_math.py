"""Plain-Python unit tests for m5_control.py's CARLA-free helpers -- no
`carla` import needed, catching a formula/off-by-one bug cheaply before
spending a live CARLA run on it (m5_control.py itself has no CARLA-based
test in this pass, see M5_Pipeline/README.md's "Known limitations")."""
import math

import m5_control
from m5_control import (
    HALF_WIDTH_FALLBACK_M,
    KNOWN_HAZARD_DEDUP_DIST_M,
    KNOWN_HAZARD_MAX_REMEMBERED,
    KNOWN_HAZARD_RADIUS_M,
    LOOKAHEAD_MAX_M,
    LOOKAHEAD_MIN_M,
    MAX_RECOVERY_ATTEMPTS,
    MAX_STEER_RAD_FALLBACK,
    RECOVERY_DURATION_TICKS,
    RECOVERY_RESET_SPEED_MPS,
    RECOVERY_CLEAR_TICKS_REQUIRED,
    RECOVERY_THROTTLE,
    STUCK_BRAKE_THRESHOLD,
    STUCK_BRAKE_TICKS_THRESHOLD,
    STUCK_PARTIAL_PROGRESS_SPEED_MPS,
    STUCK_SPEED_THRESHOLD_MPS,
    STUCK_TICKS_THRESHOLD,
    STUCK_THROTTLE_THRESHOLD,
    _clamp_lateral_offset_to_lane,
    _lane_ahead_point,
    _nudge_away_from_known_hazards,
    _query_half_width_m,
    _query_max_steer_rad,
    _speed_scaled_lookahead,
)


def test_lookahead_floors_at_min_when_stationary():
    assert _speed_scaled_lookahead(0.0) == LOOKAHEAD_MIN_M


def test_lookahead_scales_with_speed():
    la_slow = _speed_scaled_lookahead(2.0)
    la_fast = _speed_scaled_lookahead(10.0)
    assert LOOKAHEAD_MIN_M < la_slow < la_fast


def test_lookahead_caps_at_max_for_very_high_speed():
    assert _speed_scaled_lookahead(100.0) == LOOKAHEAD_MAX_M


def test_lookahead_at_cruise_speed_is_longer_than_old_fixed_4m():
    # The bug this replaces: a fixed 4m lookahead was only a ~0.4s preview
    # at CRUISE's 10 m/s target speed -- a known pure-pursuit oscillation
    # source. The new value must clearly exceed that at typical speeds.
    assert _speed_scaled_lookahead(10.0) > 4.0


class _StubWheel:
    def __init__(self, max_steer_angle):
        self.max_steer_angle = max_steer_angle


class _StubPhysics:
    def __init__(self, wheel_angles):
        self.wheels = [_StubWheel(a) for a in wheel_angles]


class _StubEgo:
    def __init__(self, wheel_angles):
        self._physics = _StubPhysics(wheel_angles)

    def get_physics_control(self):
        return self._physics


def test_query_max_steer_rad_uses_front_wheels():
    # wheels[0]=FL, wheels[1]=FR, wheels[2:]=rear (typically 0 or unused for steering)
    ego = _StubEgo([69.99, 70.0, 0.0, 0.0])
    result = _query_max_steer_rad(ego)
    assert math.isclose(result, math.radians(70.0))


def test_query_max_steer_rad_falls_back_when_front_wheels_report_zero():
    ego = _StubEgo([0.0, 0.0, 45.0, 45.0])  # e.g. a rear-steer-only oddity, or bad data
    result = _query_max_steer_rad(ego)
    assert result == MAX_STEER_RAD_FALLBACK


class _StubExtent:
    def __init__(self, y):
        self.y = y


class _StubBoundingBox:
    def __init__(self, half_width):
        self.extent = _StubExtent(half_width)


class _StubEgoWithBoundingBox:
    def __init__(self, half_width):
        self.bounding_box = _StubBoundingBox(half_width)


def test_query_half_width_uses_bounding_box():
    assert _query_half_width_m(_StubEgoWithBoundingBox(0.9)) == 0.9


def test_query_half_width_falls_back_on_zero():
    assert _query_half_width_m(_StubEgoWithBoundingBox(0.0)) == HALF_WIDTH_FALLBACK_M


class _StubLocation:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _StubRotation:
    def __init__(self, yaw):
        self.yaw = yaw


class _StubTransform:
    def __init__(self, x, y, yaw):
        self.location = _StubLocation(x, y)
        self.rotation = _StubRotation(yaw)


class _StubWaypoint:
    def __init__(self, x, y, yaw, lane_width, next_result=None):
        self.transform = _StubTransform(x, y, yaw)
        self.lane_width = lane_width
        self._next_result = next_result

    def next(self, distance):
        if self._next_result is not None:
            return self._next_result
        yaw_rad = math.radians(self.transform.rotation.yaw)
        nx = self.transform.location.x + math.cos(yaw_rad) * distance
        ny = self.transform.location.y + math.sin(yaw_rad) * distance
        return [_StubWaypoint(nx, ny, self.transform.rotation.yaw, self.lane_width)]


class _StubWaypointNextRaises(_StubWaypoint):
    def next(self, distance):
        raise RuntimeError("map query fault")


def test_clamp_leaves_offset_within_lane_untouched():
    # Lane centered at origin, heading along +x, 3.5m wide -> max lateral
    # for a 0.9m half-width vehicle is 3.5/2 - 0.9 - 0.3(margin) = 0.55m.
    wp = _StubWaypoint(x=0.0, y=0.0, yaw=0.0, lane_width=3.5)
    target, clamped = _clamp_lateral_offset_to_lane(raw_x=0.0, raw_y=0.4, wp=wp, half_vehicle_width_m=0.9)
    assert not clamped
    assert math.isclose(target["y"], 0.4, abs_tol=1e-9)


def test_clamp_reduces_offset_that_would_leave_the_lane():
    # AVOID's raw ~2.5m offset on a 3.5m-wide lane would clearly leave it.
    wp = _StubWaypoint(x=0.0, y=0.0, yaw=0.0, lane_width=3.5)
    target, clamped = _clamp_lateral_offset_to_lane(raw_x=0.0, raw_y=2.5, wp=wp, half_vehicle_width_m=0.9)
    assert clamped
    assert 0.0 < target["y"] < 2.5
    assert math.isclose(target["y"], 0.55, abs_tol=1e-9)


def test_clamp_preserves_the_direction_of_the_offset():
    wp = _StubWaypoint(x=0.0, y=0.0, yaw=0.0, lane_width=3.5)
    target, clamped = _clamp_lateral_offset_to_lane(raw_x=0.0, raw_y=-2.5, wp=wp, half_vehicle_width_m=0.9)
    assert clamped
    assert target["y"] < 0.0


def test_clamp_works_with_a_rotated_lane_heading():
    # Lane heading along +y this time -- the clamp must respect the LANE's
    # own orientation, not assume the world x-axis.
    wp = _StubWaypoint(x=10.0, y=10.0, yaw=90.0, lane_width=3.5)
    target, clamped = _clamp_lateral_offset_to_lane(raw_x=12.5, raw_y=10.0, wp=wp, half_vehicle_width_m=0.9)
    assert clamped
    # clamped point must still be very close to the lane's own longitudinal line (x=10 here)
    assert abs(target["x"] - 10.0) < 0.6
    assert math.isclose(target["y"], 10.0, abs_tol=1e-9)


def test_lane_ahead_point_is_none_without_a_current_waypoint():
    # No ego-lane query available (map fault) -- must not fabricate a point.
    assert _lane_ahead_point(wp_here=None, lookahead_m=3.0) is None


def test_lane_ahead_point_walks_forward_along_the_lane_heading():
    # Straight lane along +x -- the point must land lookahead_m further
    # along +x from the ego's own current-lane waypoint, independent of
    # whatever M5's own (possibly heading-drifted) pick was.
    wp_here = _StubWaypoint(x=5.0, y=-95.0, yaw=0.0, lane_width=3.5)
    point = _lane_ahead_point(wp_here, lookahead_m=3.0)
    assert point is not None
    assert math.isclose(point["x"], 8.0, abs_tol=1e-6)
    assert math.isclose(point["y"], -95.0, abs_tol=1e-6)


def test_lane_ahead_point_respects_a_rotated_lane_heading():
    wp_here = _StubWaypoint(x=0.0, y=0.0, yaw=90.0, lane_width=3.5)
    point = _lane_ahead_point(wp_here, lookahead_m=4.0)
    assert math.isclose(point["x"], 0.0, abs_tol=1e-6)
    assert math.isclose(point["y"], 4.0, abs_tol=1e-6)


def test_lane_ahead_point_is_none_when_the_lane_graph_has_no_next_waypoint():
    wp_here = _StubWaypoint(x=0.0, y=0.0, yaw=0.0, lane_width=3.5, next_result=[])
    assert _lane_ahead_point(wp_here, lookahead_m=3.0) is None


def test_lane_ahead_point_is_none_on_a_map_query_fault():
    wp_here = _StubWaypointNextRaises(x=0.0, y=0.0, yaw=0.0, lane_width=3.5)
    assert _lane_ahead_point(wp_here, lookahead_m=3.0) is None


# ---- Stuck recovery (live dense_market finding: one AVOID contact against
# a lane-width-infeasible obstacle pinned the vehicle at near-zero speed
# with throttle saturated for 171 further collision-sensor ticks) ----

class _StubVehicleControl:
    def __init__(self, throttle=0.0, steer=0.0, brake=0.0, reverse=False):
        self.throttle, self.steer, self.brake, self.reverse = throttle, steer, brake, reverse


class _StubCarlaModule:
    VehicleControl = _StubVehicleControl


def _make_executor():
    # __new__ bypasses __init__ (which binds a real UDP socket) -- these
    # tests only exercise _apply_stuck_recovery's own state machine, not
    # socket/CARLA plumbing, matching this file's existing "pure math,
    # no CARLA" scope.
    executor = m5_control.M5ControlExecutor.__new__(m5_control.M5ControlExecutor)
    executor._stuck_tick_count = 0
    executor._brake_stuck_tick_count = 0
    executor._recovery_ticks_remaining = 0
    executor._recovery_attempts = 0
    executor._recovery_clear_tick_count = 0
    executor._known_hazard_locations = []
    return executor


def test_not_stuck_when_moving_normally():
    executor = _make_executor()
    control = _StubVehicleControl(throttle=1.0)
    result, state = executor._apply_stuck_recovery(control, "AVOID", ego_speed=2.0, carla=_StubCarlaModule)
    assert result is control
    assert state == "AVOID"
    assert executor._stuck_tick_count == 0


def test_not_stuck_when_stationary_but_not_pushing_throttle():
    # EMERGENCY_BRAKE/STALE_FALLBACK: throttle=0 even at zero speed must
    # never trigger the THROTTLE-side trigger, by construction. It can
    # still eventually be caught by the separate, longer brake-side
    # trigger (see the tests below) -- this one only checks that a SHORT
    # brief brake, well under STUCK_BRAKE_TICKS_THRESHOLD, is left alone
    # (an ordinary, resolving safety stop, not a wedge).
    executor = _make_executor()
    control = _StubVehicleControl(throttle=0.0, brake=1.0)
    for _ in range(STUCK_TICKS_THRESHOLD + 5):
        result, state = executor._apply_stuck_recovery(control, "EMERGENCY_BRAKE", ego_speed=0.0, carla=_StubCarlaModule)
    assert result is control
    assert state == "EMERGENCY_BRAKE"
    assert executor._stuck_tick_count == 0


def test_does_not_trigger_before_the_tick_threshold():
    executor = _make_executor()
    control = _StubVehicleControl(throttle=1.0)
    for _ in range(STUCK_TICKS_THRESHOLD - 1):
        result, state = executor._apply_stuck_recovery(control, "AVOID", ego_speed=0.05, carla=_StubCarlaModule)
    assert result is control
    assert state == "AVOID"
    assert executor._stuck_tick_count == STUCK_TICKS_THRESHOLD - 1


def test_triggers_reverse_recovery_once_threshold_is_reached():
    executor = _make_executor()
    control = _StubVehicleControl(throttle=1.0)
    for _ in range(STUCK_TICKS_THRESHOLD - 1):
        executor._apply_stuck_recovery(control, "AVOID", ego_speed=0.05, carla=_StubCarlaModule)
    result, state = executor._apply_stuck_recovery(control, "AVOID", ego_speed=0.05, carla=_StubCarlaModule)
    assert state == "STUCK_RECOVERY"
    assert result.reverse is True
    assert result.throttle == RECOVERY_THROTTLE
    assert executor._recovery_attempts == 1


def test_recovery_holds_reverse_for_its_full_duration_then_returns_control():
    executor = _make_executor()
    control = _StubVehicleControl(throttle=1.0)
    for _ in range(STUCK_TICKS_THRESHOLD):
        executor._apply_stuck_recovery(control, "AVOID", ego_speed=0.05, carla=_StubCarlaModule)
    # Already 1 recovery tick consumed by the triggering call above.
    for _ in range(RECOVERY_DURATION_TICKS - 1):
        result, state = executor._apply_stuck_recovery(control, "AVOID", ego_speed=0.0, carla=_StubCarlaModule)
        assert state == "STUCK_RECOVERY"
        assert result.reverse is True
    # Recovery window has fully elapsed -- next tick falls back to normal
    # control flow (the caller's own control/state, not another reverse).
    result, state = executor._apply_stuck_recovery(control, "AVOID", ego_speed=0.05, carla=_StubCarlaModule)
    assert state == "AVOID"
    assert result is control


def test_gives_up_after_max_recovery_attempts_in_the_same_stuck_spell():
    executor = _make_executor()
    control = _StubVehicleControl(throttle=1.0)

    def run_one_full_stuck_and_recovery_cycle():
        for _ in range(STUCK_TICKS_THRESHOLD):
            executor._apply_stuck_recovery(control, "AVOID", ego_speed=0.05, carla=_StubCarlaModule)
        for _ in range(RECOVERY_DURATION_TICKS - 1):
            executor._apply_stuck_recovery(control, "AVOID", ego_speed=0.05, carla=_StubCarlaModule)

    for _ in range(MAX_RECOVERY_ATTEMPTS):
        run_one_full_stuck_and_recovery_cycle()
    assert executor._recovery_attempts == MAX_RECOVERY_ATTEMPTS

    # A 4th stuck spell, still never having driven normally in between,
    # must NOT trigger yet another reverse -- give up and hand back
    # whatever control the caller (AVOID/EMERGENCY_BRAKE/etc) computed,
    # rather than reverse/forward oscillating forever if genuinely boxed in.
    for _ in range(STUCK_TICKS_THRESHOLD + 5):
        result, state = executor._apply_stuck_recovery(control, "AVOID", ego_speed=0.05, carla=_StubCarlaModule)
    assert state == "AVOID"
    assert result is control


def test_recovery_attempts_reset_after_driving_normally_again():
    executor = _make_executor()
    control = _StubVehicleControl(throttle=1.0)
    for _ in range(STUCK_TICKS_THRESHOLD):
        executor._apply_stuck_recovery(control, "AVOID", ego_speed=0.05, carla=_StubCarlaModule)
    for _ in range(RECOVERY_DURATION_TICKS - 1):
        executor._apply_stuck_recovery(control, "AVOID", ego_speed=0.05, carla=_StubCarlaModule)
    assert executor._recovery_attempts == 1

    # Genuinely driving again well above the reset threshold, SUSTAINED for
    # RECOVERY_CLEAR_TICKS_REQUIRED consecutive ticks -- a single tick is no
    # longer enough (see the sustained-window tests below).
    for _ in range(RECOVERY_CLEAR_TICKS_REQUIRED):
        executor._apply_stuck_recovery(control, "CRUISE", ego_speed=RECOVERY_RESET_SPEED_MPS + 1.0, carla=_StubCarlaModule)
    assert executor._recovery_attempts == 0


# ---- Brake-side stuck recovery (second live dense_market finding: the
# FSM correctly emitted EMERGENCY_BRAKE against a stationary bicycle the
# ego was already touching, but braking alone never creates clearance --
# the throttle-only trigger above never fires when throttle=0, leaving the
# vehicle wedged with brake=1.0 for the rest of the run) ----

def test_brief_emergency_brake_does_not_trigger_brake_side_recovery():
    # An ordinary, resolving safety stop -- well under
    # STUCK_BRAKE_TICKS_THRESHOLD -- must be left alone.
    executor = _make_executor()
    control = _StubVehicleControl(throttle=0.0, brake=STUCK_BRAKE_THRESHOLD)
    for _ in range(STUCK_BRAKE_TICKS_THRESHOLD - 1):
        result, state = executor._apply_stuck_recovery(control, "EMERGENCY_BRAKE", ego_speed=0.0, carla=_StubCarlaModule)
    assert result is control
    assert state == "EMERGENCY_BRAKE"
    assert executor._brake_stuck_tick_count == STUCK_BRAKE_TICKS_THRESHOLD - 1


def test_triggers_reverse_recovery_once_brake_tick_threshold_is_reached():
    executor = _make_executor()
    control = _StubVehicleControl(throttle=0.0, brake=1.0)
    for _ in range(STUCK_BRAKE_TICKS_THRESHOLD - 1):
        executor._apply_stuck_recovery(control, "EMERGENCY_BRAKE", ego_speed=0.0, carla=_StubCarlaModule)
    result, state = executor._apply_stuck_recovery(control, "EMERGENCY_BRAKE", ego_speed=0.0, carla=_StubCarlaModule)
    assert state == "STUCK_RECOVERY"
    assert result.reverse is True
    assert result.throttle == RECOVERY_THROTTLE
    assert executor._recovery_attempts == 1


def test_light_partial_braking_below_threshold_does_not_count_as_stuck():
    # A moderate PID brake (e.g. gently shedding speed toward a lower
    # target) must not be mistaken for the near-full EMERGENCY_BRAKE/
    # STALE_FALLBACK brake this trigger targets.
    executor = _make_executor()
    control = _StubVehicleControl(throttle=0.0, brake=STUCK_BRAKE_THRESHOLD - 0.1)
    for _ in range(STUCK_BRAKE_TICKS_THRESHOLD + 10):
        result, state = executor._apply_stuck_recovery(control, "FOLLOW", ego_speed=0.0, carla=_StubCarlaModule)
    assert result is control
    assert executor._brake_stuck_tick_count == 0


def test_moving_while_braked_does_not_count_as_stuck():
    # brake=1.0 but ego_speed above the stuck threshold -- e.g. still
    # decelerating, not yet actually stopped -- must not count.
    executor = _make_executor()
    control = _StubVehicleControl(throttle=0.0, brake=1.0)
    for _ in range(STUCK_BRAKE_TICKS_THRESHOLD + 10):
        result, state = executor._apply_stuck_recovery(control, "EMERGENCY_BRAKE", ego_speed=2.0, carla=_StubCarlaModule)
    assert result is control
    assert executor._brake_stuck_tick_count == 0


# ---- Sustained-window reset (third live dense_market finding: the reset
# check was satisfied by a single tick of the reverse maneuver's own
# residual momentum, one tick after handing control back, clearing
# recovery_attempts before the vehicle had actually driven away -- traced
# live to a repeating stuck/recover/return-to-the-same-wall cycle) ----

def _drive_stuck_and_trigger_one_recovery(executor, control):
    for _ in range(STUCK_TICKS_THRESHOLD):
        executor._apply_stuck_recovery(control, "AVOID", ego_speed=0.05, carla=_StubCarlaModule)
    for _ in range(RECOVERY_DURATION_TICKS - 1):
        executor._apply_stuck_recovery(control, "AVOID", ego_speed=0.05, carla=_StubCarlaModule)
    assert executor._recovery_attempts == 1


def test_a_single_tick_above_reset_speed_does_not_clear_attempts():
    # Reproduces the exact live failure: one tick of speed above
    # RECOVERY_RESET_SPEED_MPS (the reverse maneuver's own residual
    # momentum) must NOT be enough on its own.
    executor = _make_executor()
    control = _StubVehicleControl(throttle=1.0)
    _drive_stuck_and_trigger_one_recovery(executor, control)

    executor._apply_stuck_recovery(control, "YIELD", ego_speed=RECOVERY_RESET_SPEED_MPS + 0.5, carla=_StubCarlaModule)
    assert executor._recovery_attempts == 1
    assert executor._recovery_clear_tick_count == 1


def test_speed_collapsing_before_the_window_completes_resets_the_clear_counter():
    # Reproduces the live tick-by-tick trace exactly: speed crosses the
    # reset threshold, then collapses back below it 2 ticks later (the
    # vehicle's own momentum reversing) well before
    # RECOVERY_CLEAR_TICKS_REQUIRED is reached -- the streak must restart
    # from zero, not just pause.
    executor = _make_executor()
    control = _StubVehicleControl(throttle=1.0)
    _drive_stuck_and_trigger_one_recovery(executor, control)

    executor._apply_stuck_recovery(control, "YIELD", ego_speed=RECOVERY_RESET_SPEED_MPS + 0.8, carla=_StubCarlaModule)
    executor._apply_stuck_recovery(control, "YIELD", ego_speed=0.08, carla=_StubCarlaModule)  # momentum reversal point
    assert executor._recovery_clear_tick_count == 0
    assert executor._recovery_attempts == 1

    # Ramping back up afterward must need the FULL window again, not credit
    # for the earlier partial streak.
    for _ in range(RECOVERY_CLEAR_TICKS_REQUIRED - 1):
        executor._apply_stuck_recovery(control, "YIELD", ego_speed=RECOVERY_RESET_SPEED_MPS + 1.0, carla=_StubCarlaModule)
    assert executor._recovery_attempts == 1
    executor._apply_stuck_recovery(control, "YIELD", ego_speed=RECOVERY_RESET_SPEED_MPS + 1.0, carla=_StubCarlaModule)
    assert executor._recovery_attempts == 0


def test_sustained_speed_for_the_full_window_clears_attempts():
    executor = _make_executor()
    control = _StubVehicleControl(throttle=1.0)
    _drive_stuck_and_trigger_one_recovery(executor, control)

    for _ in range(RECOVERY_CLEAR_TICKS_REQUIRED - 1):
        executor._apply_stuck_recovery(control, "CRUISE", ego_speed=RECOVERY_RESET_SPEED_MPS + 1.0, carla=_StubCarlaModule)
        assert executor._recovery_attempts == 1  # not yet -- window still filling
    executor._apply_stuck_recovery(control, "CRUISE", ego_speed=RECOVERY_RESET_SPEED_MPS + 1.0, carla=_StubCarlaModule)
    assert executor._recovery_attempts == 0


# ---- Known-hazard repulsion (live dense_market finding: the ego collided
# with static.static -- raw level geometry never reported in any M4/M5
# obstacle packet -- and kept driving back into the same spot after every
# successful stuck-recovery, since M5 has no way to know it's there) ----

_HAZARD_LANE = _StubWaypoint(x=0.0, y=0.0, yaw=0.0, lane_width=3.5)  # heading +x; max_lateral = 3.5/2-0.9-0.3 = 0.55
_HAZARD_WIDE_LANE = _StubWaypoint(x=0.0, y=0.0, yaw=0.0, lane_width=10.0)  # max_lateral = 10/2-0.9-0.3 = 3.8, well above KNOWN_HAZARD_RADIUS_M


def test_nudge_unchanged_with_no_known_hazards():
    waypoint = {"x": 5.0, "y": 0.0}
    result = _nudge_away_from_known_hazards(waypoint, _HAZARD_LANE, 0.9, [])
    assert result is waypoint


def test_nudge_unchanged_when_hazard_is_far_along_the_lane():
    waypoint = {"x": 5.0, "y": 0.0}
    hazard = (105.0, 0.0)  # 100m further along -- irrelevant regardless of lateral proximity
    result = _nudge_away_from_known_hazards(waypoint, _HAZARD_LANE, 0.9, [hazard])
    assert result == waypoint


def test_nudge_unchanged_when_hazard_is_laterally_far():
    waypoint = {"x": 5.0, "y": 0.0}  # lateral = 0.0
    hazard = (5.0, 3.0)  # lateral = 3.0, well outside KNOWN_HAZARD_RADIUS_M
    result = _nudge_away_from_known_hazards(waypoint, _HAZARD_LANE, 0.9, [hazard])
    assert result == waypoint


def test_nudge_pushes_toward_the_side_with_more_room_when_clamped():
    # Hazard sits at lateral=-1.0; the two +-RADIUS candidates (1.0, -3.0)
    # both get clamped to the lane's own +-0.55m -- the negative-side
    # candidate loses almost all its separation, so the positive side must
    # be chosen even though pre-clamp both were equidistant from the hazard.
    waypoint = {"x": 5.0, "y": 0.0}
    hazard = (5.0, -1.0)
    result = _nudge_away_from_known_hazards(waypoint, _HAZARD_LANE, 0.9, [hazard])
    assert result["y"] > 0.0
    assert math.isclose(result["y"], 0.55, abs_tol=1e-9)


def test_nudge_achieves_full_radius_separation_when_lane_is_wide_enough():
    waypoint = {"x": 5.0, "y": 0.0}
    hazard = (5.0, 0.4)
    result = _nudge_away_from_known_hazards(waypoint, _HAZARD_WIDE_LANE, 0.9, [hazard])
    achieved = abs(result["y"] - hazard[1])
    assert math.isclose(achieved, KNOWN_HAZARD_RADIUS_M, abs_tol=1e-9)


def test_nudge_preserves_the_along_lane_lookahead_distance():
    # The fix must only ever adjust the LATERAL component -- the forward
    # lookahead distance driving the pure-pursuit steering must be untouched.
    waypoint = {"x": 5.0, "y": 0.0}
    hazard = (5.0, -1.0)
    result = _nudge_away_from_known_hazards(waypoint, _HAZARD_LANE, 0.9, [hazard])
    assert math.isclose(result["x"], 5.0, abs_tol=1e-9)


class _StubCollisionEvent:
    def __init__(self, x, y):
        self.transform = _StubTransform(x, y, 0.0)


def test_on_collision_records_a_new_location():
    executor = _make_executor()
    executor._on_collision(_StubCollisionEvent(10.0, 20.0))
    assert executor._known_hazard_locations == [(10.0, 20.0)]


def test_on_collision_dedups_sustained_contact_at_the_same_spot():
    # A wedged vehicle's collision sensor re-fires every tick at
    # essentially the same point -- must not grow the list from that alone.
    executor = _make_executor()
    for _ in range(50):
        executor._on_collision(_StubCollisionEvent(10.0, 20.0))
    assert len(executor._known_hazard_locations) == 1


def test_on_collision_records_distinct_far_apart_points():
    executor = _make_executor()
    executor._on_collision(_StubCollisionEvent(0.0, 0.0))
    executor._on_collision(_StubCollisionEvent(50.0, 50.0))
    assert len(executor._known_hazard_locations) == 2


def test_on_collision_bounds_the_remembered_list_fifo():
    executor = _make_executor()
    for i in range(KNOWN_HAZARD_MAX_REMEMBERED + 3):
        # Spaced well beyond KNOWN_HAZARD_DEDUP_DIST_M so each one is distinct.
        executor._on_collision(_StubCollisionEvent(i * (KNOWN_HAZARD_DEDUP_DIST_M * 10), 0.0))
    assert len(executor._known_hazard_locations) == KNOWN_HAZARD_MAX_REMEMBERED
    # Oldest three evicted, most recent kept.
    assert (0.0, 0.0) not in executor._known_hazard_locations
    last_x = (KNOWN_HAZARD_MAX_REMEMBERED + 2) * (KNOWN_HAZARD_DEDUP_DIST_M * 10)
    assert (last_x, 0.0) in executor._known_hazard_locations


# ---- Partial-progress stuck detection (fourth live incident, highway_merge:
# sustained light contact with an itself near-stationary lane vehicle held
# the ego at 0.4-0.5 m/s despite throttle 0.76-0.91 and a 2.0 m/s YIELD
# target -- well above the old "not moving at all" threshold, so the
# original throttle-side trigger never saw it as stuck) ----

def test_regression_partial_progress_below_new_threshold_triggers_recovery():
    # Reproduces the toyota.prius incident's numbers directly: speed pinned
    # well above the old 0.15 m/s threshold but far below both the new
    # 0.6 m/s threshold and the FSM's own 2.0 m/s YIELD target.
    executor = _make_executor()
    control = _StubVehicleControl(throttle=0.85)
    for _ in range(STUCK_TICKS_THRESHOLD - 1):
        result, state = executor._apply_stuck_recovery(control, "YIELD", ego_speed=0.45, carla=_StubCarlaModule)
    assert state == "YIELD"
    result, state = executor._apply_stuck_recovery(control, "YIELD", ego_speed=0.45, carla=_StubCarlaModule)
    assert state == "STUCK_RECOVERY"
    assert result.reverse is True


def test_speed_just_above_the_new_threshold_does_not_count_as_stuck():
    executor = _make_executor()
    control = _StubVehicleControl(throttle=0.85)
    for _ in range(STUCK_TICKS_THRESHOLD + 5):
        result, state = executor._apply_stuck_recovery(
            control, "YIELD", ego_speed=STUCK_PARTIAL_PROGRESS_SPEED_MPS + 0.05, carla=_StubCarlaModule,
        )
    assert result is control
    assert state == "YIELD"


def test_normal_launch_from_a_stop_does_not_trigger_recovery():
    # Real recorded launch profile (CRUISE, throttle=1.0 throughout): 9
    # ticks of static-friction breakaway at exactly 0.0 m/s, then a fast
    # ramp clearing STUCK_PARTIAL_PROGRESS_SPEED_MPS well inside
    # STUCK_TICKS_THRESHOLD. Must never be mistaken for a wedge.
    executor = _make_executor()
    control = _StubVehicleControl(throttle=1.0)
    launch_speeds = [0.0] * 9 + [0.388, 1.261, 1.930, 2.506]
    for speed in launch_speeds:
        result, state = executor._apply_stuck_recovery(control, "CRUISE", ego_speed=speed, carla=_StubCarlaModule)
    assert result is control
    assert state == "CRUISE"
    assert executor._stuck_tick_count == 0  # reset the moment speed cleared the threshold
