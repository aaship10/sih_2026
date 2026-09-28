"""Integration-level tests: a synthetic ObstaclePacket through the full
M5Engine.step(), checking the pieces (ttc/costmap/fsm/quintic) actually
compose into a sane DecisionPacket."""
import pytest

import config
from planner import M5Engine, _pick_safe_avoid_offset, _required_avoid_offset_m, _swept_path_conflicts
from schema import parse_obstacle_packet


def _packet(obstacles=(), statics=(), ego_pos=(0.0, 0.0), ego_vel=(5.0, 0.0), ego_yaw_deg=0.0, frame_id=1):
    return parse_obstacle_packet({
        "timestamp": 0.0, "frame_id": frame_id,
        "ego_position": list(ego_pos), "ego_velocity": list(ego_vel), "ego_yaw_deg": ego_yaw_deg,
        "obstacles": list(obstacles), "statics": list(statics),
    })


def _obstacle(track_id, pos, vel, points, mode="nominal", time_step_s=0.3, class_name="car", width=2.0, length=4.5):
    return {
        "track_id": track_id, "class": class_name, "confidence": 0.9,
        "pos_x": pos[0], "pos_y": pos[1], "vel_x": vel[0], "vel_y": vel[1],
        "width": width, "length": length,
        "trajectories": [{
            "mode": mode, "probability": 1.0,
            "points": [{"x": p[0], "y": p[1], "t": i * time_step_s} for i, p in enumerate(points)],
            "sigma_m": [0.5] * len(points),
        }],
    }


def _phantom_static_artifact(pos):
    # Reproduces the live-incident signature EXACTLY (confirmed by
    # replaying the actual recorded incident data): both offending tracks
    # were M3-flagged is_static=True -- unknown class, LiDAR-only
    # confidence, close to the ego. StaticObstacle carries no velocity.
    return {
        "track_id": 99, "class": "unknown", "confidence": 0.4,
        "pos_x": pos[0], "pos_y": pos[1], "width": 0.5, "length": 0.5,
    }


def _roadside_static(pos, track_id=50):
    # A real, non-artifact static (high confidence, so the sensor-artifact
    # filter never touches it) -- e.g. a traffic cone or road sign at the
    # curb, the exact live-testing failure case the anisotropic fix targets.
    return {"track_id": track_id, "class": "unknown", "confidence": 0.9, "pos_x": pos[0], "pos_y": pos[1], "width": 0.3, "length": 0.3}


def test_regression_roadside_static_does_not_force_emergency_brake():
    # Live 5-scenario validation finding: with ego facing +x, a static
    # object 1.8m purely to the SIDE (matching these scenarios' actual
    # traffic-cone/road-sign placement, 1.6-2.8m off the lane) used to read
    # as full RED risk -- exactly like something 1.8m directly ahead -- and
    # forced EMERGENCY_BRAKE despite nothing blocking the lane. The same
    # object directly AHEAD at the same distance must still correctly brake.
    engine_side = M5Engine()
    decision_side = engine_side.step(_packet(statics=[_roadside_static((0.0, 1.8))], ego_vel=(5.0, 0.0)))
    assert decision_side.behavior_state != "EMERGENCY_BRAKE"

    engine_ahead = M5Engine()
    decision_ahead = engine_ahead.step(_packet(statics=[_roadside_static((1.8, 0.0))], ego_vel=(5.0, 0.0)))
    assert decision_ahead.behavior_state == "EMERGENCY_BRAKE"


def test_regression_static_ahead_at_yellow_risk_triggers_avoid_not_yield():
    # Live 5-scenario validation finding (dense_market): a static object
    # directly ahead at YELLOW now-risk used to trigger YIELD instead of
    # AVOID, because _future_footprint_risk fed statics in too and their
    # future risk (worse than "now", since the ego's own straight-line
    # projection drives right through a stationary object) preempted the
    # AVOID branch in fsm.py's _raw_level. YIELD's path is a straight-line
    # continuation (README) -- the ego crept forward at full throttle
    # directly into the static and got physically wedged (393
    # collision-sensor contacts in the live run) instead of steering
    # around it.
    engine = M5Engine()
    decision = engine.step(_packet(statics=[_roadside_static((5.5, 0.0))], ego_vel=(5.0, 0.0)))
    assert decision.behavior_state == "AVOID"
    assert abs(decision.trajectory[-1].y) > 0.5  # real lateral swerve, not a straight line


def test_regression_near_stationary_dynamic_obstacle_ahead_triggers_avoid_not_yield():
    # Second live 5-scenario validation finding (dense_market, SAME run):
    # excluding statics from _future_footprint_risk (the fix above) wasn't
    # enough -- the actual colliding object (vehicle.bh.crossbike, a parked
    # bicycle) is tracked as a DYNAMIC obstacle (M3 sees velocity jitter,
    # never flags is_static), not a static, so that fix never applied to
    # it. Root cause was more general: fsm.py's _raw_level checked
    # footprint_future_risk (-> YIELD) before AVOID-eligibility at all,
    # and for ANY roughly-stationary obstacle -- static or a near-zero-
    # velocity dynamic track alike -- its CV-predicted future position
    # tracks its current position, so future_risk ~= now_risk and YIELD
    # kept winning regardless of object classification. Same live signature
    # (YIELD, SNAPPED, throttle=1.0, ego crawling at <0.1 m/s wedged
    # against the bicycle) recurred even after the statics-only fix (447
    # collision-sensor contacts, worse than the original 393). Fixed by
    # reordering _raw_level so AVOID-eligibility (TTC or now-risk) is
    # checked before the future-risk YIELD branch.
    obstacle = _obstacle(
        track_id=77, pos=(5.5, 0.0), vel=(0.02, 0.01),
        points=[(5.5 + 0.02 * i * 0.3, 0.0 + 0.01 * i * 0.3) for i in range(11)],
    )
    engine = M5Engine()
    decision = engine.step(_packet(obstacles=[obstacle], ego_vel=(0.6, 0.0)))
    assert decision.behavior_state == "AVOID"
    assert abs(decision.trajectory[-1].y) > 0.5  # real lateral swerve, not a straight line


def test_pick_safe_avoid_offset_flips_to_the_clear_mirrored_side():
    # Third live incident (cattle_crossing: static.prop.streetsign01, 206
    # collision-sensor contacts; same signature as scenario3's curb-strike):
    # AVOID's candidate offset used to be applied with no idea whether that
    # side was itself clear -- ego yaw drifted steadily TOWARD the sign
    # over consecutive AVOID ticks instead of away from the original
    # obstacle. A static sits exactly at the naive +2.5m offset target;
    # _pick_safe_avoid_offset must flip to the clear -2.5m mirror instead.
    packet = _packet(statics=[_roadside_static((0.0, 2.5), track_id=60)])
    offset = _pick_safe_avoid_offset(2.5, 0.0, 0.0, 0.0, packet)
    assert offset < 0.0
    assert abs(offset - (-2.5)) < 0.01


def test_pick_safe_avoid_offset_falls_back_to_zero_when_both_sides_blocked():
    # Both the candidate offset AND its mirror are blocked BY RED (genuine
    # overlap/near-contact, distance 0 from each check point) -- must not
    # pick either, relying on EMERGENCY_BRAKE/YIELD to actually prevent the
    # collision instead of forcing a swerve into a second obstacle either way.
    packet = _packet(statics=[_roadside_static((0.0, 2.5), track_id=60), _roadside_static((0.0, -2.5), track_id=61)])
    offset = _pick_safe_avoid_offset(2.5, 0.0, 0.0, 0.0, packet)
    assert offset == 0.0


def test_pick_safe_avoid_offset_escalates_past_a_wedged_base_offset():
    # Live cattle_crossing finding: once the ego is already touching an
    # obstacle, the base candidate AND its mirror can BOTH read RED (the
    # obstacle's own RED contact radius reaches both points), and the old
    # code gave up to 0.0 right there -- confirmed live as 146-346 repeated
    # collisions with the same actor in a single run. A static this close
    # (0.5m either side of center) keeps both the 1.0m base AND its mirror
    # RED, but clears to YELLOW by 2.0m, well under AVOID_MAX_OFFSET_M=2.5 --
    # escalation must find and return it instead of surrendering.
    packet = _packet(statics=[_roadside_static((0.0, 0.5), track_id=60), _roadside_static((0.0, -0.5), track_id=61)])
    offset = _pick_safe_avoid_offset(1.0, 0.0, 0.0, 0.0, packet)
    assert abs(offset - 2.0) < 0.01


def test_pick_safe_avoid_offset_escalation_still_respects_the_lane_ceiling():
    # Same wedged-base-offset statics, which also keep 1.5m RED (only 2.0m
    # clears, per the previous test) -- with M1 reporting a lane whose
    # ceiling is 1.5m, escalation must not reach the 2.0m that would
    # otherwise work, and must not overshoot the ceiling just because
    # AVOID_MAX_OFFSET_M would otherwise allow more room.
    packet = _packet(statics=[_roadside_static((0.0, 0.5), track_id=60), _roadside_static((0.0, -0.5), track_id=61)])
    offset = _pick_safe_avoid_offset(1.0, 0.0, 0.0, 0.0, packet, max_achievable_offset_m=1.5)
    assert offset == 0.0


def test_pick_safe_avoid_offset_accepts_yellow_only_rejects_red():
    # Direct unit check of the RED-vs-YELLOW threshold itself: a candidate
    # offset whose check point reads YELLOW (not RED) must be accepted, not
    # treated the same as a genuinely blocked (RED) point.
    packet_yellow = _packet(statics=[_roadside_static((0.0, 4.0), track_id=62)])  # ~4m from check point (0,2.5) -> YELLOW, not RED
    assert _pick_safe_avoid_offset(2.5, 0.0, 0.0, 0.0, packet_yellow) == pytest.approx(2.5)

    packet_red = _packet(statics=[_roadside_static((0.0, 2.5), track_id=63)])  # distance 0 -> RED
    assert _pick_safe_avoid_offset(2.5, 0.0, 0.0, 0.0, packet_red) != pytest.approx(2.5)


def test_no_swerve_for_a_moving_pedestrian_triggering_avoid():
    # Explicit user direction: for a genuinely MOVING pedestrian (or any
    # other config.NO_SWERVE_CLASSES member actually underway), stopping
    # early and letting it clear is safer than relying on a swerve.
    #
    # Two earlier attempts suppressed the LATERAL OFFSET instead (full
    # suppression, then a capped modest-nudge ceiling) -- both confirmed
    # live as WORSE than an unrestricted swerve (1023 and 566 collisions
    # respectively, vs. 49 for plain swerve): AVOID's forward target speed
    # was left unchanged in both, so the ego kept approaching at full speed
    # with reduced/zero ability to dodge, only actually stopping once TTC
    # collapsed to EMERGENCY_BRAKE's 1.5s cutoff -- too close, and a
    # stationary vehicle can't reposition laterally by steering at all. The
    # real fix (see _plan_path's own comment) forces target_speed to 0.0
    # instead, so the planner decelerates from AVOID's very first tick --
    # the lateral offset itself stays completely unrestricted, same as any
    # other obstacle, as a fallback for whatever TTC doesn't leave room to
    # stop in.
    required = _required_avoid_offset_m(obstacle_half_width=0.25)
    points = [(10.0 + 0.0 * i * 0.3, 0.0 + 1.4 * i * 0.3) for i in range(11)]
    obstacles = [_obstacle(
        1, pos=(10.0, 0.0), vel=(0.0, 1.4), points=points, class_name="pedestrian", width=0.5, length=0.5,
    )]
    engine = M5Engine()
    decision = engine.step(_packet(obstacles=obstacles))
    assert decision.behavior_state == "AVOID"
    assert decision.target_speed == 0.0  # early-stop override, not AVOID_SPEED_MPS
    offset = decision.trajectory[-1].y
    assert offset != 0.0  # swerve itself is NOT suppressed -- still available as a fallback
    assert abs(offset) == pytest.approx(required, abs=0.01)  # nothing else in the scene forces escalation past it
    assert abs(offset) < config.AVOID_MAX_OFFSET_M


def test_no_swerve_moving_obstacle_gets_the_same_offset_a_regular_obstacle_would():
    # Revised design (see test_no_swerve_for_a_moving_pedestrian_triggering_
    # avoid's comment): a moving NO_SWERVE_CLASSES obstacle no longer gets a
    # capped, non-escalating offset ceiling -- that approach measured worse
    # live. The offset computation (_pick_safe_avoid_offset, including its
    # escalation past a blocked initial offset) is now completely untouched
    # by obstacle class; the only thing the class/speed gate changes is
    # target_speed. Proven directly here: identical geometry/velocity,
    # differing only in class_name (one inside config.NO_SWERVE_CLASSES, one
    # not) must produce the IDENTICAL offset, with target_speed differing.
    points = [(10.0, 0.0 + 1.4 * i * 0.3) for i in range(11)]

    engine_pedestrian = M5Engine()
    ped = _obstacle(1, pos=(10.0, 0.0), vel=(0.0, 1.4), points=points, class_name="pedestrian", width=0.5, length=0.5)
    decision_pedestrian = engine_pedestrian.step(_packet(obstacles=[ped]))

    # "traffic_cones" is a real classes.yaml class outside config.NO_SWERVE_
    # CLASSES -- an unrealistic thing to see moving, but this is a
    # controlled unit check of the offset-vs-class independence, not a
    # claim about real cone behavior.
    engine_regular = M5Engine()
    cone = _obstacle(2, pos=(10.0, 0.0), vel=(0.0, 1.4), points=points, class_name="traffic_cones", width=0.5, length=0.5)
    decision_regular = engine_regular.step(_packet(obstacles=[cone]))

    assert decision_pedestrian.behavior_state == decision_regular.behavior_state == "AVOID"
    assert decision_pedestrian.trajectory[-1].y == pytest.approx(decision_regular.trajectory[-1].y)
    assert decision_pedestrian.target_speed == 0.0
    assert decision_regular.target_speed == config.AVOID_SPEED_MPS


def test_swerve_still_happens_for_a_parked_no_swerve_class_obstacle():
    # The same class, but essentially stationary (a parked bicycle, a
    # stopped car) -- it will never clear the way an actually-moving one
    # will, so it still needs the same swerve-around behavior as any other
    # stationary hazard, not a wait-it-out stop. Below config.
    # NO_SWERVE_MIN_SPEED_MPS is what makes the difference here.
    obstacle = _obstacle(
        track_id=77, pos=(5.5, 0.0), vel=(0.02, 0.01),
        points=[(5.5 + 0.02 * i * 0.3, 0.0 + 0.01 * i * 0.3) for i in range(11)],
        class_name="bicycle",
    )
    engine = M5Engine()
    decision = engine.step(_packet(obstacles=[obstacle], ego_vel=(0.6, 0.0)))
    assert decision.behavior_state == "AVOID"
    assert decision.target_speed == config.AVOID_SPEED_MPS  # below NO_SWERVE_MIN_SPEED_MPS -- no override
    assert abs(decision.trajectory[-1].y) > 0.5


def test_no_swerve_class_still_gets_an_emergency_brake_escape_offset():
    # Revised design: EMERGENCY_BRAKE's own target speed is already 0.0
    # (config.EMERGENCY_SPEED_MPS) for every obstacle class, so the
    # early-stop override is a no-op here -- and the lateral offset itself
    # was never suppressed by this fix (see _plan_path's own comment), only
    # gated by NO_SWERVE_MIN_SPEED_MPS's speed check on entry. A moving
    # NO_SWERVE_CLASSES obstacle at EMERGENCY_BRAKE range must still get a
    # real escape offset, same as test_emergency_brake_still_computes_an_
    # escape_offset_when_room_allows, not a suppressed one.
    # Far enough away (25m) that footprint_now_risk isn't RED at the ego's
    # actual current position -- EMERGENCY_BRAKE fires via fast-closing TTC
    # alone (20 m/s closing speed -> ~1.25s TTC). Deliberately NOT placed
    # close enough to be RED-now: a dynamic obstacle already overlapping the
    # ego's current footprint makes _swept_path_conflicts' own t=0 sample
    # (which always checks the obstacle's t=0 position against the ego's
    # OWN CURRENT position, before any candidate offset takes effect) read
    # RED for every candidate offset alike -- an unrelated, pre-existing
    # limitation of the swept-path check for already-overlapping dynamic
    # obstacles, not something this test is about.
    obstacle = _obstacle(
        track_id=1, pos=(25.0, 0.3), vel=(0.0, 1.4),
        points=[(25.0, 0.3 + 1.4 * i * 0.3) for i in range(11)],
        class_name="pedestrian", width=0.5, length=0.5,
    )
    engine = M5Engine()
    decision = engine.step(_packet(obstacles=[obstacle], ego_vel=(20.0, 0.0)))
    assert decision.behavior_state == "EMERGENCY_BRAKE"
    assert decision.target_speed == 0.0
    assert decision.trajectory[-1].y != 0.0


def test_emergency_brake_still_computes_an_escape_offset_when_room_allows():
    # EMERGENCY_BRAKE fix (live-run finding): a pedestrian occluded until
    # very close (cattle_crossing's own "sudden appearance" design) can push
    # TTC straight into EMERGENCY_BRAKE with AVOID never getting a turn at
    # all on the very first tick it's seen -- M1's own emergency-stop
    # handling used to hard-zero steer in that case (brake dead straight,
    # regardless of what M5 sent), confirmed live as repeated contact while
    # cycling EMERGENCY_BRAKE -> a brake-stuck-triggered reverse -> creep
    # forward -> back into EMERGENCY_BRAKE range -> repeat, for most of a
    # run, never actually clearing the pedestrian. M5 must offer a real
    # lateral escape offset for EMERGENCY_BRAKE too, not just AVOID, so
    # M1 has something to steer toward while still braking at full force.
    engine = M5Engine()
    decision = engine.step(_packet(statics=[_roadside_static((1.8, 0.3))], ego_vel=(5.0, 0.0)))
    assert decision.behavior_state == "EMERGENCY_BRAKE"
    assert decision.trajectory[-1].y != 0.0


def test_avoid_direction_locks_across_ticks_despite_a_moving_obstacles_bearing_flipping():
    # Live cattle_crossing finding: a pedestrian actively crossing the ego's
    # path sweeps _nearest_risky_obstacle_direction's bearing from one side
    # to the other as it crosses -- without a lock, this flips the AVOID
    # offset's sign every tick, producing a violent steering oscillation
    # (+0.89 <-> -0.78 every few ticks, a 4m+ lateral zigzag) that never
    # converges on actually avoiding it. Same obstacle position, mirrored
    # across two consecutive ticks on the SAME engine -- the chosen swerve
    # side must stay locked to whatever the first tick picked.
    engine = M5Engine()
    right_side = _roadside_static((3.38, -0.91), track_id=61)
    left_side = _roadside_static((3.38, 0.91), track_id=61)

    first = engine.step(_packet(statics=[right_side], ego_vel=(5.0, 0.0)))
    assert first.behavior_state == "AVOID"
    first_sign = first.trajectory[-1].y > 0

    second = engine.step(_packet(statics=[left_side], ego_vel=(5.0, 0.0)))
    assert second.behavior_state == "AVOID"
    second_sign = second.trajectory[-1].y > 0

    assert first_sign == second_sign


def test_avoid_direction_lock_resets_after_leaving_avoid():
    # The lock must not persist forever -- once the engine leaves AVOID
    # (obstacle cleared) and later enters AVOID again for a genuinely
    # different obstacle on the other side, it must be free to pick that
    # new side, not stay stuck on whatever the previous episode chose.
    engine = M5Engine()
    right_side = _roadside_static((3.38, -0.91), track_id=61)
    first = engine.step(_packet(statics=[right_side], ego_vel=(5.0, 0.0)))
    assert first.behavior_state == "AVOID"

    clear = engine.step(_packet(statics=[], ego_vel=(5.0, 0.0)))
    assert clear.behavior_state != "AVOID"

    left_side = _roadside_static((3.38, 0.91), track_id=62)
    second = engine.step(_packet(statics=[left_side], ego_vel=(5.0, 0.0)))
    assert second.behavior_state == "AVOID"
    assert (second.trajectory[-1].y > 0) != (first.trajectory[-1].y > 0)


def test_regression_dense_scene_avoid_offset_not_rejected_by_unrelated_clutter():
    # The live dense_market finding (108 collisions after the offset-
    # magnitude fix, all against one traffic cone): once a scene has ~30+
    # simultaneous tracked objects, _pick_safe_avoid_offset's old
    # "not in (RED, YELLOW)" criterion made both the candidate AND its
    # mirror fail almost every tick -- COSTMAP_YELLOW_RADIUS_M=5.0m is wide
    # enough that some UNRELATED clutter object is nearly always within
    # range of any offset point in a crowded scene, even though that
    # clutter has nothing to do with the obstacle actually being avoided.
    # Confirmed via direct replay of the real live incident: candidate read
    # YELLOW, mirror read RED, every tick, permanently falling back to 0.0
    # (a dead-straight path into the very obstacle AVOID was computed to
    # clear). Reproduces the minimal shape: a primary obstacle at a shallow
    # bearing (AVOID-eligible, YELLOW at the ego's own position) plus one
    # UNRELATED clutter object positioned near the candidate offset's
    # target point (not the primary, and not close enough to the ego
    # itself to change now_risk there) -- the offset must still be
    # delivered, not collapse to 0.0.
    primary = _roadside_static((3.38, -0.91), track_id=61)  # ~15deg bearing, YELLOW at ego's position (AVOID-eligible)
    clutter = _roadside_static((0.0, 3.8), track_id=70)  # unrelated to primary; sits near the candidate check point (0, 1.8)
    engine = M5Engine()
    decision = engine.step(_packet(statics=[primary, clutter], ego_vel=(5.0, 0.0)))
    assert decision.behavior_state == "AVOID"
    assert decision.trajectory[-1].y != 0.0  # must NOT collapse to a dead-straight path
    assert abs(decision.trajectory[-1].y) > 1.0  # a real, decisive swerve, not a token nudge


def _predicted_path_to(ego_x, ego_y, target_x, target_y):
    # A dynamic obstacle whose OWN predicted trajectory exactly tracks the
    # straight-line approach _swept_path_conflicts assumes for a candidate
    # AVOID offset -- guarantees an exact (time, position) match at every
    # sample, i.e. a genuine, unambiguous swept-path collision.
    time_step_s = config.PLAN_HORIZON_S / (config.N_WAYPOINTS - 1)
    points = []
    for i in range(config.N_WAYPOINTS):
        frac = i / (config.N_WAYPOINTS - 1)
        points.append((ego_x + (target_x - ego_x) * frac, ego_y + (target_y - ego_y) * frac))
    return points, time_step_s


def test_swept_path_conflicts_true_when_a_dynamic_obstacles_predicted_path_matches_the_candidate():
    # Live scenario3 incident this closes: a current-tick point-check at the
    # candidate offset's END point reads clear, but the obstacle's predicted
    # path crosses somewhere ALONG the approach, at a future time -- only a
    # genuine swept check catches it. `pos` (the obstacle's CURRENT position,
    # separate from its predicted trajectory) is placed well away from
    # either check point so this test isolates the swept check from
    # _now_footprint_risk's own, already-covered point check.
    points, time_step_s = _predicted_path_to(0.0, 0.0, 0.0, 2.0)
    obstacle = _obstacle(1, pos=(50.0, 50.0), vel=(0.0, 0.0), points=points, time_step_s=time_step_s)
    packet = _packet(obstacles=[obstacle])
    assert _swept_path_conflicts(0.0, 0.0, 0.0, 2.0, 0.0, packet) is True


def test_swept_path_conflicts_false_when_predicted_path_stays_well_clear():
    points = [(100.0, 100.0)] * config.N_WAYPOINTS
    time_step_s = config.PLAN_HORIZON_S / (config.N_WAYPOINTS - 1)
    obstacle = _obstacle(1, pos=(100.0, 100.0), vel=(0.0, 0.0), points=points, time_step_s=time_step_s)
    packet = _packet(obstacles=[obstacle])
    assert _swept_path_conflicts(0.0, 0.0, 0.0, 2.0, 0.0, packet) is False


def test_swept_path_conflicts_ignores_statics_by_construction():
    # StaticObstacle has no `trajectories` field at all (schema.py) -- a
    # static's relative position under any projection is already fully
    # covered by _now_footprint_risk, re-evaluated fresh every tick (same
    # reasoning _future_footprint_risk's own docstring gives).
    packet = _packet(statics=[_roadside_static((0.0, 1.0), track_id=80)])
    assert _swept_path_conflicts(0.0, 0.0, 0.0, 2.0, 0.0, packet) is False


def _predicted_path_sliding_into(start_x, start_y, end_x, end_y):
    # Like _predicted_path_to, but the obstacle approaches its own,
    # independent starting point rather than the ego's current position --
    # every candidate/mirror path also starts AT the ego (frac=0 is always
    # (ego_x, ego_y) by construction), so anchoring the obstacle there too
    # would spuriously "conflict" with every offset, not just the one it's
    # actually converging on by the end of the horizon.
    time_step_s = config.PLAN_HORIZON_S / (config.N_WAYPOINTS - 1)
    points = [
        (start_x + (end_x - start_x) * i / (config.N_WAYPOINTS - 1),
         start_y + (end_y - start_y) * i / (config.N_WAYPOINTS - 1))
        for i in range(config.N_WAYPOINTS)
    ]
    return points, time_step_s


def test_pick_safe_avoid_offset_rejects_a_side_that_sweeps_into_a_second_moving_obstacle():
    # The candidate side's CURRENT-tick point-check reads clear (nothing is
    # AT (0, 2.5) right now) but a second, independently-moving obstacle's
    # own predicted path converges on that same point by the end of the
    # horizon -- must flip to the mirror, exactly as it already does for a
    # static blocking the point itself.
    points, time_step_s = _predicted_path_sliding_into(3.0, 2.5, 0.0, 2.5)
    crossing_vehicle = _obstacle(1, pos=(50.0, 50.0), vel=(0.0, 0.0), points=points, time_step_s=time_step_s)
    packet = _packet(obstacles=[crossing_vehicle])
    offset = _pick_safe_avoid_offset(2.5, 0.0, 0.0, 0.0, packet)
    assert offset < 0.0
    assert abs(offset - (-2.5)) < 0.01


def test_pick_safe_avoid_offset_falls_back_to_zero_when_both_sides_sweep_into_moving_obstacles():
    points_pos, time_step_s = _predicted_path_sliding_into(3.0, 2.5, 0.0, 2.5)
    points_neg, _ = _predicted_path_sliding_into(3.0, -2.5, 0.0, -2.5)
    crossing_vehicle_a = _obstacle(1, pos=(50.0, 50.0), vel=(0.0, 0.0), points=points_pos, time_step_s=time_step_s)
    crossing_vehicle_b = _obstacle(2, pos=(-50.0, -50.0), vel=(0.0, 0.0), points=points_neg, time_step_s=time_step_s)
    packet = _packet(obstacles=[crossing_vehicle_a, crossing_vehicle_b])
    offset = _pick_safe_avoid_offset(2.5, 0.0, 0.0, 0.0, packet)
    assert offset == 0.0


def test_regression_phantom_near_ego_artifact_does_not_force_emergency_brake():
    # The actual live-CARLA incident this filter fixes: two persistent
    # sensor-artifact statics right next to the ego, with NO real hazard
    # anywhere else in the scene, used to force a permanent false RED
    # costmap reading -> EMERGENCY_BRAKE the ego could never recover from
    # (until scenario3.py's own stuck-recovery teleported it into a real
    # vehicle -- the actual sequence, confirmed against recorded logs).
    statics = [_phantom_static_artifact(pos=(1.95, 0.0)), _phantom_static_artifact(pos=(3.55, 0.0))]
    engine = M5Engine()
    decision = engine.step(_packet(statics=statics, ego_vel=(0.0, 0.0)))
    assert decision.behavior_state == "CRUISE"
    assert not decision.emergency_stop
    assert engine.last_diagnostics["n_filtered_artifacts"] == 2


def test_regression_ghost_tracking_a_moving_ego_does_not_force_emergency_brake():
    # SECOND live incident (same track_id, confirmed via M3's own
    # is_static flag flipping False on it mid-run): the same ghost's
    # reported WORLD velocity climbed smoothly toward the ego's own speed
    # as the ego accelerated away, while staying ~1.5-2.5m from the ego the
    # whole time -- a "noise misread as motion" pattern that used to slip
    # past the (then absolute-speed) filter the instant it exceeded
    # SENSOR_ARTIFACT_MAX_SPEED_MPS, even though it never actually moved
    # relative to the car. This is a dynamic obstacle (is_static=False by
    # the time its velocity climbed), unlike the first incident's statics.
    ego_vel = (0.0, 6.0)
    obstacles = [{
        "track_id": 23, "class": "unknown", "confidence": 0.4,
        "pos_x": 1.5, "pos_y": 2.0, "vel_x": 0.02, "vel_y": 5.9,  # ~matches ego_vel -> ~0 relative speed
        "width": 0.5, "length": 0.5, "trajectories": [],
    }]
    engine = M5Engine()
    decision = engine.step(_packet(obstacles=obstacles, ego_vel=ego_vel))
    assert decision.behavior_state == "CRUISE"
    assert not decision.emergency_stop
    assert engine.last_diagnostics["n_filtered_artifacts"] == 1


def test_regression_stability_window_catches_ghost_that_velocity_alone_missed():
    # Reproduces the ACTUAL recorded second-incident dynamics that defeated
    # the relative-velocity fix above: the ghost's reported velocity gives
    # a relative speed of ~1.2 m/s every tick (never near-zero, so the
    # velocity path never catches it), but its distance to the ego stays
    # essentially constant (~2.02m) tick after tick. Before
    # SENSOR_ARTIFACT_STABILITY_MIN_SAMPLES ticks of history accumulate,
    # nothing can catch it and the FSM is (falsely) stuck in
    # EMERGENCY_BRAKE; once enough history exists, the stability check
    # kicks in on the very next tick and the ego recovers immediately.
    def packet(i):
        # Ghost placed DIRECTLY AHEAD (same x, +1.5m in the direction of
        # travel) rather than to the side -- this test targets the
        # stability-window mechanism specifically, not the anisotropic
        # lateral-risk fix (costmap.py), so it must stay RED regardless of
        # FOOTPRINT_LATERAL_STRETCH.
        ego_y = float(i)
        ghost_y = ego_y + 1.5
        return _packet(
            obstacles=[{
                "track_id": 23, "class": "unknown", "confidence": 0.4,
                "pos_x": 0.0, "pos_y": ghost_y, "vel_x": 0.01, "vel_y": 0.8,
                "width": 0.5, "length": 0.5, "trajectories": [],
            }],
            ego_pos=(0.0, ego_y), ego_vel=(0.0, 2.0), ego_yaw_deg=90.0, frame_id=i,
        )

    engine = M5Engine()
    min_samples = config.SENSOR_ARTIFACT_STABILITY_MIN_SAMPLES
    for i in range(min_samples - 1):
        decision = engine.step(packet(i))
        assert engine.last_diagnostics["n_filtered_artifacts"] == 0
        assert decision.behavior_state == "EMERGENCY_BRAKE"

    decision = engine.step(packet(min_samples - 1))
    assert engine.last_diagnostics["n_filtered_artifacts"] == 1
    assert decision.behavior_state == "CRUISE"
    assert not decision.emergency_stop


def test_regression_implausibly_small_vehicle_class_track_filtered_as_artifact():
    # THIRD live incident (dense_market): track_id=58, class="car",
    # multi-sensor confidence 0.55 -- ABOVE SENSOR_ARTIFACT_MAX_CONFIDENCE,
    # so the class=='unknown' pathway never touches it -- born at the exact
    # spot/moment the ego first contacted a real parked car, LiDAR-cluster
    # extent never exceeding 0.54m for its ~95-tick life. It became M5's
    # "nearest obstacle" and anchored every AVOID offset at a fraction of
    # the clearance the real car actually needed, driving repeated wedging
    # for the rest of the scenario. Reproduced here with a single such
    # track and nothing else in the scene: it must be filtered on sight,
    # with no history/stability window needed.
    obstacles = [{
        "track_id": 58, "class": "car", "confidence": 0.55,
        "pos_x": 2.0, "pos_y": 0.0, "vel_x": 0.0, "vel_y": 0.0,
        "width": 0.39, "length": 0.43, "trajectories": [],
    }]
    engine = M5Engine()
    decision = engine.step(_packet(obstacles=obstacles, ego_vel=(0.0, 0.0)))
    assert engine.last_diagnostics["n_filtered_artifacts"] == 1
    assert decision.behavior_state == "CRUISE"
    assert not decision.emergency_stop


def test_plausibly_sized_vehicle_class_track_is_not_filtered():
    # A real car-sized obstacle (matching _obstacle()'s own 2.0x4.5m
    # default) at the same confidence/distance as the artifact above must
    # NOT be caught by the new size check -- this is what actually
    # distinguishes "implausible" from "genuinely small but real".
    obstacles = [_obstacle(58, pos=(2.0, 0.0), vel=(0.0, 0.0), points=[(2.0, 0.0)] * 11)]
    engine = M5Engine()
    decision = engine.step(_packet(obstacles=obstacles, ego_vel=(0.0, 0.0)))
    assert engine.last_diagnostics["n_filtered_artifacts"] == 0


def test_implausibly_small_non_vehicle_class_is_not_filtered_by_size_check():
    # traffic_cones (and the other small-by-nature classes) are legitimately
    # small in real life -- the new check is scoped to VEHICLE_LIKE_CLASSES
    # only and must leave a small cone alone.
    obstacles = [{
        "track_id": 58, "class": "traffic_cones", "confidence": 0.55,
        "pos_x": 2.0, "pos_y": 0.0, "vel_x": 0.0, "vel_y": 0.0,
        "width": 0.3, "length": 0.3, "trajectories": [],
    }]
    engine = M5Engine()
    decision = engine.step(_packet(obstacles=obstacles, ego_vel=(0.0, 0.0)))
    assert engine.last_diagnostics["n_filtered_artifacts"] == 0


def test_regression_implausibly_small_vehicle_class_static_filtered_as_artifact():
    # Static-side counterpart: track_id=58 was briefly is_static=True on
    # its very first tick before M3 flipped it to a dynamic track.
    statics = [{
        "track_id": 58, "class": "car", "confidence": 0.55,
        "pos_x": 2.0, "pos_y": 0.0, "width": 0.39, "length": 0.43,
    }]
    engine = M5Engine()
    decision = engine.step(_packet(statics=statics, ego_vel=(0.0, 0.0)))
    assert engine.last_diagnostics["n_filtered_artifacts"] == 1
    assert decision.behavior_state == "CRUISE"


def test_empty_scene_cruises_with_full_waypoint_count():
    engine = M5Engine()
    decision = engine.step(_packet())
    assert decision.behavior_state == "CRUISE"
    assert not decision.emergency_stop
    assert decision.target_speed == config.CRUISE_SPEED_MPS
    assert len(decision.trajectory) == config.N_WAYPOINTS
    assert decision.trajectory[0].t == 0.0


def test_close_fast_closing_obstacle_triggers_emergency_stop():
    # Obstacle 3m ahead, ego closing at 5 m/s -> TTC = 3/5 = 0.6s (< 1.5s emergency threshold).
    obstacles = [_obstacle(1, pos=(3.0, 0.0), vel=(0.0, 0.0), points=[(3.0, 0.0)] * 11)]
    engine = M5Engine()
    decision = engine.step(_packet(obstacles=obstacles))
    assert decision.behavior_state == "EMERGENCY_BRAKE"
    assert decision.emergency_stop
    assert decision.target_speed == 0.0


def test_distant_slow_closing_obstacle_does_not_trigger_emergency():
    # Obstacle far ahead, low closing speed -> long TTC -> CRUISE or FOLLOW, never emergency.
    obstacles = [_obstacle(1, pos=(200.0, 0.0), vel=(4.0, 0.0), points=[(200.0 + 4.0 * i * 0.3, 0.0) for i in range(11)])]
    engine = M5Engine()
    decision = engine.step(_packet(obstacles=obstacles))
    assert decision.behavior_state in ("CRUISE", "FOLLOW")
    assert not decision.emergency_stop


def test_avoid_state_produces_a_laterally_offset_path():
    # Obstacle ahead drifting away laterally fast enough that its PREDICTED
    # path clears the ego's own nominal path (no YIELD-triggering future
    # conflict), but its current closing TTC alone is still in AVOID's
    # range -> AVOID triggered by TTC, not by current/future costmap risk.
    # Caught a real bug: the lateral-offset direction picker used to only
    # look at RED/YELLOW *current*-position risk, so a TTC-only AVOID
    # produced a perfectly straight "avoidance" path with no offset at all.
    # class_name="unknown": this test is about the offset-computation
    # machinery itself (does a TTC-only AVOID produce a real offset at
    # all?), not about any specific obstacle class's behavior -- "unknown"
    # keeps it clear of config.NO_SWERVE_CLASSES's stop-don't-swerve rule
    # for a fast-moving vehicle-classed obstacle like this one.
    points = [(10.0 + 0.0 * i * 0.3, 0.0 + 8.0 * i * 0.3) for i in range(11)]
    obstacles = [_obstacle(1, pos=(10.0, 0.0), vel=(0.0, 8.0), points=points, class_name="unknown")]
    engine = M5Engine()
    decision = engine.step(_packet(obstacles=obstacles))
    assert decision.behavior_state == "AVOID"
    assert abs(decision.trajectory[-1].y) > 0.5


def test_required_avoid_offset_m_scales_with_obstacle_size():
    # Direct unit check of the new clearance formula: obstacle half-width
    # + ego footprint radius + safety margin, not a sine-of-bearing-angle
    # fraction of an arbitrary constant (the old formula this replaces).
    small = _required_avoid_offset_m(0.15)  # e.g. a traffic cone, width~0.3
    large = _required_avoid_offset_m(0.9)   # e.g. a pushcart, width~1.8
    assert small == pytest.approx(0.15 + config.EGO_FOOTPRINT_RADIUS_M + config.AVOID_SAFETY_MARGIN_M)
    assert large > small  # bigger obstacle demands more clearance


def test_regression_shallow_bearing_obstacle_right_gets_full_clearance_not_sine_scaled():
    # The actual live dense_market incident this fixes (3 real collisions):
    # a static object at roughly a 15 degree bearing off the ego's heading
    # -- NOT dead-ahead (the old formula's only full-2.5m case) and NOT
    # purely lateral -- used to produce candidate_offset = 2.5*sin(15deg)
    # =~ 0.65m, nowhere near enough to actually clear it. Obstacle is to
    # the RIGHT (negative y) of the ego's forward (+x) heading, so AVOID
    # must swerve LEFT (positive y).
    obstacle = _roadside_static((5.3126, -1.4235), track_id=61)  # ~15 deg bearing, width/length default 0.3
    engine = M5Engine()
    decision = engine.step(_packet(statics=[obstacle], ego_vel=(5.0, 0.0)))
    assert decision.behavior_state == "AVOID"
    offset_y = decision.trajectory[-1].y
    required = _required_avoid_offset_m(0.15)  # _roadside_static's width=length=0.3
    assert offset_y > 0  # correct side: LEFT, away from the obstacle on the right
    assert offset_y == pytest.approx(min(required, config.AVOID_MAX_OFFSET_M), abs=0.05)


def test_regression_shallow_bearing_obstacle_left_gets_full_clearance_not_sine_scaled():
    # Mirror of the above: obstacle to the LEFT (positive y) -> AVOID must
    # swerve RIGHT (negative y). Confirms direction is preserved correctly
    # in both cases, not just coincidentally correct in one.
    obstacle = _roadside_static((5.3126, 1.4235), track_id=62)
    engine = M5Engine()
    decision = engine.step(_packet(statics=[obstacle], ego_vel=(5.0, 0.0)))
    assert decision.behavior_state == "AVOID"
    offset_y = decision.trajectory[-1].y
    required = _required_avoid_offset_m(0.15)
    assert offset_y < 0  # correct side: RIGHT, away from the obstacle on the left
    assert abs(offset_y) == pytest.approx(min(required, config.AVOID_MAX_OFFSET_M), abs=0.05)


def test_avoid_offset_capped_at_max_for_a_wide_obstacle():
    # A wide-enough obstacle demands more clearance than
    # config.AVOID_MAX_OFFSET_M -- must cap there (M5 has no lane/map
    # source of its own, see config.py's AVOID_MAX_OFFSET_M docstring),
    # not exceed it just because a bigger obstacle would technically want more.
    wide_obstacle = {"track_id": 63, "class": "unknown", "confidence": 0.9,
                      "pos_x": 5.3126, "pos_y": -1.4235, "width": 3.0, "length": 3.0}
    engine = M5Engine()
    decision = engine.step(_packet(statics=[wide_obstacle], ego_vel=(5.0, 0.0)))
    assert decision.behavior_state == "AVOID"
    offset_y = decision.trajectory[-1].y
    assert offset_y > 0
    assert offset_y == pytest.approx(config.AVOID_MAX_OFFSET_M, abs=0.05)


# ---- Lane-width feasibility (SPEC.md section 4's M1->M5 map interface,
# finally built: M1 reports the live lane width so AVOID can recognize a
# required clearance the lane physically cannot provide BEFORE committing
# to it, rather than relying on M1's clamp to silently shrink an infeasible
# offset down to whatever fits -- live cattle_crossing/dense_market
# incidents showed that shrunk-but-nonzero offset still doesn't clear the
# obstacle, just collides a little less directly before stuck-recovery
# fires and the identical doomed offset gets recomputed and repeated) ----

def test_pick_safe_avoid_offset_rejects_when_lane_cannot_provide_clearance():
    # No obstacle anywhere near the check points -- if this returns 0.0,
    # it can only be the upfront lane-width rejection, not a point-check.
    packet = _packet()
    offset = _pick_safe_avoid_offset(2.0, 0.0, 0.0, 0.0, packet, max_achievable_offset_m=0.5)
    assert offset == 0.0


def test_pick_safe_avoid_offset_accepts_when_clearance_fits_within_lane():
    packet = _packet()
    offset = _pick_safe_avoid_offset(2.0, 0.0, 0.0, 0.0, packet, max_achievable_offset_m=3.0)
    assert offset == pytest.approx(2.0)


def test_pick_safe_avoid_offset_unconstrained_when_lane_width_not_provided():
    # The default -- every existing caller/test relies on this staying
    # exactly as it was before this parameter existed.
    packet = _packet()
    offset = _pick_safe_avoid_offset(2.0, 0.0, 0.0, 0.0, packet)
    assert offset == pytest.approx(2.0)


def test_regression_narrow_lane_collapses_avoid_to_straight_instead_of_a_doomed_partial_swerve():
    # Same wide obstacle as the AVOID_MAX_OFFSET_M-clamp test above (needs
    # the full 2.5m clamp to clear) -- but now a lane too narrow to ever
    # provide that (3.0m lane -> 0.35m max achievable). Without lane data,
    # this obstacle produces a full 2.5m swerve (confirmed below); with a
    # narrow lane reported, AVOID must recognize that swerve can't actually
    # clear the obstacle and fall back to straight (0.0), the same
    # EMERGENCY_BRAKE/YIELD-reliant fallback already used when both sides
    # read RED -- not commit to a partial swerve M1 would just clamp down
    # to something equally useless anyway.
    wide_obstacle = {"track_id": 63, "class": "unknown", "confidence": 0.9,
                      "pos_x": 5.3126, "pos_y": -1.4235, "width": 3.0, "length": 3.0}

    engine_unconstrained = M5Engine()
    decision_unconstrained = engine_unconstrained.step(_packet(statics=[wide_obstacle], ego_vel=(5.0, 0.0)))
    assert decision_unconstrained.behavior_state == "AVOID"
    assert decision_unconstrained.trajectory[-1].y == pytest.approx(config.AVOID_MAX_OFFSET_M, abs=0.05)

    engine_narrow_lane = M5Engine()
    decision_narrow_lane = engine_narrow_lane.step(
        _packet(statics=[wide_obstacle], ego_vel=(5.0, 0.0)), lane_width_m=3.0,
    )
    assert decision_narrow_lane.behavior_state == "AVOID"
    assert decision_narrow_lane.trajectory[-1].y == 0.0


def test_wide_lane_does_not_interfere_with_a_clearance_that_already_fits():
    wide_obstacle = {"track_id": 63, "class": "unknown", "confidence": 0.9,
                      "pos_x": 5.3126, "pos_y": -1.4235, "width": 3.0, "length": 3.0}
    engine = M5Engine()
    decision = engine.step(_packet(statics=[wide_obstacle], ego_vel=(5.0, 0.0)), lane_width_m=10.0)
    assert decision.behavior_state == "AVOID"
    assert decision.trajectory[-1].y == pytest.approx(config.AVOID_MAX_OFFSET_M, abs=0.05)


def test_decision_dict_round_trips_expected_keys():
    engine = M5Engine()
    decision = engine.step(_packet())
    d = decision.to_dict()
    for key in ("behavior_state", "emergency_stop", "target_speed", "trajectory", "replan_triggered", "computation_time_ms"):
        assert key in d
    assert len(d["trajectory"]) == config.N_WAYPOINTS
    for wp in d["trajectory"]:
        assert set(wp.keys()) == {"x", "y", "v_target", "yaw", "t"}
