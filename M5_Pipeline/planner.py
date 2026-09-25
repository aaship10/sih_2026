"""M5's per-tick decision engine: joins ttc.py, costmap.py/footprint.py,
fsm.py and quintic.py into the one `M5Engine.step()` call m5_server.py runs
at config.PLANNING_HZ. This is the Python-layer + decision-layer split the
spec describes (section 2 and 3), collapsed into one process since both
layers are already Python here for the math (see fsm.py's docstring). The
decision layer itself (fsm.py's BehaviorFSM) now has a real MATLAB/Simulink
Stateflow counterpart -- see config.USE_STATEFLOW_FSM and
M5Engine._create_fsm below -- BehaviorFSM stays as the default, fast,
MATLAB-free implementation and the validated reference the chart is tested
against (tests/test_fsm_stateflow_parity.py).
"""
from __future__ import annotations

import itertools
import logging
import math
import time
from collections import deque

import config
from costmap import RED, Costmap, anisotropic_offset, classify_distance
from footprint import footprint_risk
from fsm import TARGET_SPEED_BY_STATE, BehaviorFSM
from quintic import plan_quintic_path
from schema import DecisionPacket, Obstacle, ObstaclePacket, StaticObstacle, Waypoint
from ttc import compute_ttc, min_ttc as compute_min_ttc

LOGGER = logging.getLogger("planner")

# Classes for which a sub-config.SENSOR_ARTIFACT_MIN_VEHICLE_SIZE_M bounding
# box is physically implausible -- see config.py's matching comment for the
# live dense_market incident (track_id=58) this targets. Deliberately
# excludes pedestrian/animal (body-size LiDAR clusters are legitimately
# variable) and the small-by-nature classes (road_sign, traffic_signal,
# traffic_cones, speed_bumps, pothole) -- this is a size check, not a
# generic ghost filter, so it only applies where "smaller than ~1m" can
# only mean a fragment, never a genuine complete object of that class.
VEHICLE_LIKE_CLASSES = frozenset({"car", "motorcycle", "bus", "tempo", "truck", "rickshaw", "bicycle"})


def _is_implausibly_small_for_class(class_name: str, width: float, length: float) -> bool:
    """True if `class_name` is one of VEHICLE_LIKE_CLASSES but the reported
    LiDAR-cluster extent is smaller than any real vehicle of that class
    could physically present -- see config.SENSOR_ARTIFACT_MIN_VEHICLE_SIZE_M's
    docstring for the live incident this catches. Independent of confidence
    and needs no history: the object is implausible from the moment it's
    this small, not just after a stability window."""
    return class_name in VEHICLE_LIKE_CLASSES and max(width, length) < config.SENSOR_ARTIFACT_MIN_VEHICLE_SIZE_M


def _distance_is_stable(history: "deque[float] | list[float]", band_m: float, min_samples: int) -> bool:
    """True if `history` has enough samples and every one of them falls
    within a `band_m`-wide range -- the live-incident signal that survives
    when velocity readings are unreliable garbage (see config.py's
    SENSOR_ARTIFACT_STABILITY_* comment): a real external object's distance
    to the ego changes as either one moves, but a self-detection ghost's
    distance stays roughly constant regardless of what its own reported
    velocity claims."""
    if len(history) < min_samples:
        return False
    return (max(history) - min(history)) <= band_m


def _is_likely_sensor_artifact(obs: Obstacle, ego_x: float, ego_y: float, ego_vx: float, ego_vy: float,
                                distance_history: "deque[float] | list[float] | None" = None) -> bool:
    """Filters a specific, confirmed real-world failure mode (see
    config.py's matching comment for the live-CARLA-testing incidents this
    targets): a persistent class='unknown', LiDAR-only-confidence track
    sitting a few meters from the ego -- gated on class + confidence +
    distance, plus EITHER of two independent signals that the object isn't
    a real closing/moving hazard:

      1. Low velocity RELATIVE TO THE EGO (catches the simple case
         immediately, no history needed -- e.g. the first incident's
         statics, already stationary from birth).
      2. Distance-to-ego stable over a rolling window (`distance_history`,
         built by M5Engine._update_artifact_history) -- catches the second
         incident's case, where the reported velocity is itself unreliable
         (a Kalman-filter noise-climb/decay artifact with NO relationship
         to the ego's actual motion, confirmed by replaying the recorded
         incident: neither absolute nor relative speed ever drops below
         threshold during the window this ghost is active) but its
         distance to the ego stays essentially fixed the whole time
         regardless of what its velocity claims.

    Either signal alone is sufficient to exclude -- a genuine close call
    (known class, higher confidence, OR failing BOTH checks) is never
    filtered no matter how close.

    A THIRD, independent signal (see config.SENSOR_ARTIFACT_MIN_VEHICLE_SIZE_M):
    a vehicle-like class whose reported size is physically implausible for
    that class, regardless of confidence -- this is checked first and does
    not require class_name == "unknown"."""
    if _is_implausibly_small_for_class(obs.class_name, obs.width, obs.length):
        return True
    if obs.class_name != "unknown" or obs.confidence > config.SENSOR_ARTIFACT_MAX_CONFIDENCE:
        return False
    if math.hypot(obs.pos_x - ego_x, obs.pos_y - ego_y) > config.SENSOR_ARTIFACT_EXCLUSION_RADIUS_M:
        return False
    relative_speed = math.hypot(obs.vel_x - ego_vx, obs.vel_y - ego_vy)
    if relative_speed <= config.SENSOR_ARTIFACT_MAX_SPEED_MPS:
        return True
    if distance_history is not None and _distance_is_stable(
        distance_history, config.SENSOR_ARTIFACT_STABILITY_BAND_M, config.SENSOR_ARTIFACT_STABILITY_MIN_SAMPLES,
    ):
        return True
    return False


def _is_likely_sensor_artifact_static(obs: StaticObstacle, ego_x: float, ego_y: float) -> bool:
    """Same filter as _is_likely_sensor_artifact, minus the velocity check
    -- StaticObstacle carries no velocity at all (M4's output/m5_udp_schema.py:
    "there is nothing to predict for something that isn't moving"), and the
    actual live incident this fixes was a track M3 itself had already
    flagged is_static=True (so a zero-velocity check would be redundant by
    construction here anyway).

    Also applies the implausible-size check from _is_likely_sensor_artifact
    (see its docstring / config.SENSOR_ARTIFACT_MIN_VEHICLE_SIZE_M) --
    the dense_market track_id=58 incident was briefly is_static=True on its
    very first tick, before M3 flipped it to a dynamic track."""
    if _is_implausibly_small_for_class(obs.class_name, obs.width, obs.length):
        return True
    if obs.class_name != "unknown" or obs.confidence > config.SENSOR_ARTIFACT_MAX_CONFIDENCE:
        return False
    return math.hypot(obs.pos_x - ego_x, obs.pos_y - ego_y) <= config.SENSOR_ARTIFACT_EXCLUSION_RADIUS_M


def _now_footprint_risk(ego_x: float, ego_y: float, ego_yaw: float, packet: ObstaclePacket) -> str:
    cm = Costmap(center_x=ego_x, center_y=ego_y, heading_rad=ego_yaw, lateral_stretch=config.FOOTPRINT_LATERAL_STRETCH)
    for obs in packet.obstacles:
        cm.mark_obstacle(obs.pos_x, obs.pos_y)
    for s in packet.statics:
        cm.mark_obstacle(s.pos_x, s.pos_y)
    return footprint_risk(ego_x, ego_y, ego_yaw, cm)


def _future_footprint_risk(ego_x: float, ego_y: float, ego_vx: float, ego_vy: float, ego_yaw: float,
                            packet: ObstaclePacket) -> str:
    """Checks the ego's own nominal (constant-velocity) future path against
    every predicted obstacle trajectory point at MATCHING time -- the spec
    open item E mitigation: catches a crossing/offset conflict a bare TTC
    formula misses, without needing a fresh grid per timestep (see
    costmap.classify_distance's docstring). Anisotropic the same way
    _now_footprint_risk is (see costmap.py's module docstring) -- the
    ego's heading is held constant at `ego_yaw` throughout the horizon,
    matching the straight-line assumption already used for the ego's own
    projected position.

    Deliberately excludes `packet.statics`: a static's relative position
    under this same straight-line projection is fully determined already,
    so unlike a genuinely time-varying dynamic obstacle, its future risk
    carries no information _now_footprint_risk won't already show on the
    very next tick as the ego closes in -- but it DOES change which FSM
    branch fires first (fsm.py's _raw_level checks footprint_future_risk,
    hence YIELD, before footprint_now_risk's AVOID branch). Live 5-scenario
    validation (dense_market) found this starved AVOID almost entirely for
    static roadside obstacles directly ahead: a static's future risk is
    essentially always >= its now risk along the ego's own path, so YIELD
    (a straight-line path, per README) kept winning over AVOID (which
    actually steers around it) -- 393 collision-sensor contacts from the
    ego creeping straight into a cone/bicycle at full throttle rather than
    swerving. Statics remain fully covered for EMERGENCY_BRAKE/AVOID via
    _now_footprint_risk, re-evaluated fresh every tick."""
    times = [config.PLAN_HORIZON_S * i / (config.N_WAYPOINTS - 1) for i in range(config.N_WAYPOINTS)]
    time_tol = config.PLAN_HORIZON_S / config.N_WAYPOINTS  # half a sample step, roughly

    worst = "NONE"
    severity = {"NONE": 0, "GREEN": 1, "YELLOW": 2, "RED": 3}

    for t in times:
        ex, ey = ego_x + ego_vx * t, ego_y + ego_vy * t

        for obs in packet.obstacles:
            for traj in obs.trajectories:
                nearest = min(traj.points, key=lambda p: abs(p.t - t))
                if abs(nearest.t - t) > time_tol:
                    continue
                along, lateral = anisotropic_offset(nearest.x - ex, nearest.y - ey, ego_yaw, config.FOOTPRINT_LATERAL_STRETCH)
                d = math.hypot(along, lateral) - config.EGO_FOOTPRINT_RADIUS_M
                r = classify_distance(d)
                if severity[r] > severity[worst]:
                    worst = r
    return worst


def _nearest_risky_obstacle_direction(
    ego_x: float, ego_y: float, packet: ObstaclePacket,
) -> tuple[tuple[float, float], float, str, float] | None:
    """(unit direction, half_width, class_name, speed_mps) for the single
    nearest obstacle, where the unit vector points FROM that obstacle
    TOWARD the ego -- an AVOID maneuver steers away along this direction,
    sized by the half_width (see _required_avoid_offset_m). Deliberately
    NOT filtered to only RED/YELLOW-now obstacles: AVOID can be triggered
    by TTC alone (an obstacle that's still far enough to read GREEN/NONE on
    the current-position costmap but is closing fast) -- filtering here the
    same way silently produced a dead-straight "avoidance" path with no
    actual avoidance for exactly that case (caught by test_planner.py's
    AVOID case). Includes `packet.statics` alongside `packet.obstacles`:
    a static-only scene (no `packet.obstacles` at all) used to return None
    here unconditionally, so an AVOID state triggered purely by a static
    roadside object (costmap risk, not TTC) got lateral_offset_m=0.0 in
    _plan_path -- an "AVOID" that was actually a dead-straight path, same
    live dense_market finding as _future_footprint_risk's docstring above.
    half_width is max(width, length)/2 -- M5 doesn't treat width/length as
    an oriented box (README/spec note: they're M3's axis-aligned LiDAR-
    cluster extent), so the larger dimension avoids under-estimating the
    clearance actually needed if the real orientation faces its long axis
    at the ego. class_name is M2/M3's own detection class (classes.yaml's
    14-class taxonomy, e.g. "pedestrian", "car", "traffic_cones") -- see
    _plan_path's own use of it to decide swerve-vs-stop, which also needs
    speed_mps (0.0 for anything from `packet.statics` -- StaticObstacle
    carries no velocity at all, consistent with M3 having already judged it
    non-moving) to tell a genuinely moving instance of a class apart from a
    parked/stationary one (a parked bicycle needs to be swerved around like
    any other stationary hazard -- it will never "clear" on its own the way
    an actually-moving one will). None only when there are no obstacles OR
    statics at all."""
    best_d, best_vec, best_half_width, best_class, best_speed = math.inf, None, 0.0, "unknown", 0.0
    for pos_x, pos_y, width, length, class_name, speed in itertools.chain(
        ((o.pos_x, o.pos_y, o.width, o.length, o.class_name, math.hypot(o.vel_x, o.vel_y)) for o in packet.obstacles),
        ((s.pos_x, s.pos_y, s.width, s.length, s.class_name, 0.0) for s in packet.statics),
    ):
        d = math.hypot(pos_x - ego_x, pos_y - ego_y)
        if d < best_d:
            best_d = d
            best_vec = (ego_x - pos_x, ego_y - pos_y)
            best_half_width = max(width, length) / 2.0
            best_class = class_name
            best_speed = speed
    if best_vec is None:
        return None
    norm = math.hypot(*best_vec)
    if norm <= 1e-6:
        return None
    return (best_vec[0] / norm, best_vec[1] / norm), best_half_width, best_class, best_speed


def _has_nearby_moving_no_swerve_obstacle(ego_x: float, ego_y: float, ego_vx: float, ego_vy: float,
                                           packet: ObstaclePacket) -> bool:
    """True if ANY dynamic obstacle that is itself, ON ITS OWN, close/closing
    enough to justify AVOID (the same TTC_AVOID_S criterion _raw_level uses,
    plus a tight-distance fallback for the already-touching/wedged case where
    relative closing speed reads ~0 so TTC is undefined -- see below) is also
    a moving (>= config.NO_SWERVE_MIN_SPEED_MPS) config.NO_SWERVE_CLASSES
    member -- not just whichever single obstacle _nearest_risky_obstacle_
    direction happens to pick as closest. A fixed proximity radius (tried
    first, config.COSTMAP_YELLOW_RADIUS_M) was too tight: it missed a fast-
    closing obstacle still 10m out reading AVOID via TTC alone (TTC_AVOID_S's
    whole point, shared with _raw_level's own TTC branch) -- this obstacle-
    own-TTC check matches that criterion exactly instead of re-deriving a
    separate, inconsistent distance cutoff.

    Live cattle_crossing finding: once the ego is
    wedged against BOTH a moving pedestrian AND an unrelated stationary object
    (a traffic cone) at nearly the same distance, gating the early-stop
    override on only the single nearest obstacle's class let it flicker off
    on whichever tick the cone (not in NO_SWERVE_CLASSES) happened to read
    marginally closer -- snapping target_speed back to the state's full speed
    and commanding full throttle straight into the jam, then STUCK_RECOVERY
    backing up a few cm, repeating in a tight oscillation for ~10s straight
    while the pedestrian kept walking into the jerking vehicle the whole
    time (each AVOID<->EMERGENCY_BRAKE transition during this also re-rolls
    the steer sign-lock, the likely direct cause of several live "collided
    with static.road" contacts during the same window). Checking ANY nearby
    qualifying obstacle instead of only the nearest one keeps the override
    latched for as long as a real moving hazard is in the vicinity,
    regardless of what else happens to be touching the ego at the same
    moment. Statics are never checked here -- speed_mps is always 0.0 for
    them by construction (schema.py's StaticObstacle has no velocity field),
    so they can never qualify anyway."""
    for o in packet.obstacles:
        if o.class_name not in config.NO_SWERVE_CLASSES:
            continue
        if math.hypot(o.vel_x, o.vel_y) < config.NO_SWERVE_MIN_SPEED_MPS:
            continue
        if math.hypot(o.pos_x - ego_x, o.pos_y - ego_y) <= config.COSTMAP_RED_RADIUS_M:
            return True  # already touching/wedged -- TTC below would read None (closing_speed ~0)
        obstacle_ttc = compute_ttc((ego_x, ego_y), (ego_vx, ego_vy), (o.pos_x, o.pos_y), (o.vel_x, o.vel_y))
        if obstacle_ttc is not None and obstacle_ttc <= config.TTC_AVOID_S:
            return True
    return False


def _required_avoid_offset_m(obstacle_half_width: float) -> float:
    """Minimum lateral offset magnitude that actually clears an obstacle
    of the given half-width: the obstacle's own half-width, plus the
    ego's own footprint radius, plus a real safety margin -- NOT a
    sine-of-bearing-angle fraction of an arbitrary constant (see
    _plan_path's old formula / config.AVOID_SAFETY_MARGIN_M's docstring
    for the live-incident evidence this replaces: as little as ~0.25m of
    actual clearance for an obstacle at a shallow bearing, nowhere near
    enough to clear a pushcart/cone plus the ego's own width)."""
    return obstacle_half_width + config.EGO_FOOTPRINT_RADIUS_M + config.AVOID_SAFETY_MARGIN_M


def _swept_path_conflicts(ego_x: float, ego_y: float, target_x: float, target_y: float,
                           ego_yaw: float, packet: ObstaclePacket) -> bool:
    """True if a straight-line approach from the ego's current position to
    (target_x, target_y), reached over config.PLAN_HORIZON_S -- the same
    cheap approximation _future_footprint_risk already uses for the ego's
    own projected motion elsewhere in this module, not the real quintic
    curve _plan_path eventually flies -- crosses a DYNAMIC obstacle's OWN
    predicted trajectory at a matching future time.

    Live scenario3 incident this closes: _pick_safe_avoid_offset's existing
    checks are both single-current-tick point-checks (this function's
    caller's own docstring: "against the CURRENT costmap... a point-check")
    with no notion of time at all -- an offset picked to dodge one obstacle
    can be, and live-CARLA-confirmed was, driven straight into a SECOND,
    independently-moving hazard whose predicted path only crosses that
    offset a few ticks later. Traced live: the ego steered to clear a
    static obstacle while a crossing vehicle (CARLA CROSS-traffic, itself
    also braking to a stop) converged on the same point from a different
    direction -- both point-checks read clear at tick zero, since neither
    obstacle occupied that exact spot yet.

    RED only (see _pick_safe_avoid_offset's own docstring for why YELLOW
    over-rejects once a scene has 30+ tracked objects -- the identical risk
    applies here and this reuses the same threshold on purpose). Statics
    excluded: no `trajectories` field (schema.py) since a fixed object's
    relative position under any projection is already fully covered by
    _now_footprint_risk, re-evaluated fresh every tick -- exactly
    _future_footprint_risk's own reasoning, reused here rather than
    re-derived."""
    times = [config.PLAN_HORIZON_S * i / (config.N_WAYPOINTS - 1) for i in range(config.N_WAYPOINTS)]
    time_tol = config.PLAN_HORIZON_S / config.N_WAYPOINTS

    for t in times:
        frac = t / config.PLAN_HORIZON_S
        ex = ego_x + (target_x - ego_x) * frac
        ey = ego_y + (target_y - ego_y) * frac

        for obs in packet.obstacles:
            for traj in obs.trajectories:
                nearest = min(traj.points, key=lambda p: abs(p.t - t))
                if abs(nearest.t - t) > time_tol:
                    continue
                along, lateral = anisotropic_offset(nearest.x - ex, nearest.y - ey, ego_yaw, config.FOOTPRINT_LATERAL_STRETCH)
                d = math.hypot(along, lateral) - config.EGO_FOOTPRINT_RADIUS_M
                if classify_distance(d) == RED:
                    return True
    return False


def _pick_safe_avoid_offset(candidate_offset: float, ego_x: float, ego_y: float, ego_yaw: float,
                             packet: ObstaclePacket, max_achievable_offset_m: float | None = None) -> float:
    """AVOID's candidate lateral offset (from _plan_path, derived purely
    from the nearest obstacle's direction) has no idea whether the side it
    picked is itself clear -- it can steer straight into a SECOND obstacle,
    or off the road entirely, instead of away from the first one. Live
    5-scenario validation found exactly this: a road sign in cattle_crossing
    (and, matching signature, scenario3's curb-strike) -- the ego's yaw
    drifted steadily TOWARD the obstacle over consecutive AVOID ticks, not
    away, then EMERGENCY_BRAKE<->AVOID cycling repeated the identical bad
    swerve forever once stuck, since nothing about this computation changes
    between ticks with the same obstacle set. Checked here against the
    CURRENT costmap (not the far quintic-path endpoint -- a point-check
    consistent with every other risk check in this module, see
    _future_footprint_risk's docstring) at the pure lateral offset from the
    ego's current position: try the candidate, then its mirror, then no
    offset at all.

    Accepts YELLOW, rejects only RED (originally rejected both). Live
    dense_market re-verification of the offset-magnitude fix (config.
    AVOID_SAFETY_MARGIN_M) found a second, distinct bug this addresses: once
    a scene has ~30+ simultaneous tracked objects, COSTMAP_YELLOW_RADIUS_M's
    generous 5m radius means almost ANY point within a normal lane-width
    offset has SOME object within 5m -- unrelated to the specific obstacle
    actually being avoided. Requiring "not in (RED, YELLOW)" made both the
    candidate AND its mirror fail almost every tick once density crossed
    that threshold (confirmed via direct replay: candidate read YELLOW,
    mirror read RED, every single tick of a 113-frame collision episode),
    permanently falling back to 0.0 -- a dead-straight path directly at the
    very obstacle the offset was computed to clear, reproducing exactly the
    collision this function exists to prevent. RED alone is the right
    threshold: it's COSTMAP_RED_RADIUS_M's tight 2m radius, i.e. genuine
    overlap/near-contact -- both original live incidents (the cattle_crossing
    sign, scenario3's curb-strike) had the ego already touching/wedged
    against the second obstacle, which reads RED, not merely YELLOW, so this
    relaxation does not reopen either of them (test_pick_safe_avoid_offset_
    flips_to_the_clear_mirrored_side / ..._falls_back_to_zero_when_both_
    sides_blocked both place statics exactly AT the check point, distance
    0 -- always RED regardless of this threshold change).
    Falling all the way through to 0.0 (now only when BOTH sides read RED,
    NOW or along their swept path -- see _swept_path_conflicts) relies on
    EMERGENCY_BRAKE/YIELD to actually prevent the collision instead of
    driving further into it.

    Also rejects a side whose approach SWEEPS through a second, independently
    -moving obstacle's own predicted path even though the current-tick point
    -check above reads clear -- see _swept_path_conflicts's own docstring for
    the live scenario3 incident (a crossing vehicle, itself also braking to
    a stop, converging on the chosen offset from a different direction) this
    closes.

    `max_achievable_offset_m` (from M1's own lane-width report -- SPEC.md
    section 4's M1->M5 map interface, see config.M1_LANE_INFO_UDP_PORT) adds
    a THIRD, upfront rejection: if the candidate can't fit within the lane
    at all, neither side is worth even point-checking -- M1's clamp would
    silently shrink whatever we send down to what fits anyway, and live
    cattle_crossing/dense_market incidents showed that shrunk-but-nonzero
    offset still doesn't clear the obstacle, just collides a little less
    directly before the same stuck/recover/re-collide cycle repeats. None
    (the default, and what every existing caller/test still passes) means
    "no lane data received yet" -- behavior is then identical to before
    this parameter existed.

    Escalation fix (live cattle_crossing finding): candidate_offset only
    covers the MINIMUM clearance needed for the obstacle _required_avoid_
    offset_m identified -- once the ego is already touching/wedged against
    it, both that offset and its mirror can independently read RED (the
    obstacle's own COSTMAP_RED_RADIUS_M contact radius can reach both
    points from a near-zero ego-to-obstacle distance), and the old
    candidate/mirror/0.0 sequence gave up right there. Confirmed live:
    146-346 repeated collisions with the same actor in a single run once
    wedged. Before giving up, this now also tries larger magnitudes (in
    config.AVOID_OFFSET_ESCALATION_STEP_M steps, both sides, preserving the
    candidate side first at each step) up to whatever's actually available
    -- max_achievable_offset_m when M1's lane-width data has arrived,
    else AVOID_MAX_OFFSET_M, same ceiling either candidate_offset itself
    was already clamped against. When candidate_offset is already AT that
    ceiling (the common case once an obstacle's required clearance exceeds
    it) there is no room to escalate into and behavior is unchanged --
    see test_..._falls_back_to_zero_when_both_sides_blocked, which
    exercises exactly that case."""
    if max_achievable_offset_m is not None and abs(candidate_offset) > max_achievable_offset_m:
        return 0.0
    if candidate_offset == 0.0:
        return 0.0
    heading = (math.cos(ego_yaw), math.sin(ego_yaw))
    sign = math.copysign(1.0, candidate_offset)
    max_mag = max_achievable_offset_m if max_achievable_offset_m is not None else config.AVOID_MAX_OFFSET_M

    magnitudes = [abs(candidate_offset)]
    next_mag = magnitudes[0] + config.AVOID_OFFSET_ESCALATION_STEP_M
    while next_mag < max_mag:
        magnitudes.append(next_mag)
        next_mag += config.AVOID_OFFSET_ESCALATION_STEP_M
    if max_mag > magnitudes[-1]:
        magnitudes.append(max_mag)

    for magnitude in magnitudes:
        for offset in (sign * magnitude, -sign * magnitude):
            cx = ego_x - heading[1] * offset
            cy = ego_y + heading[0] * offset
            if _now_footprint_risk(cx, cy, ego_yaw, packet) != RED and not _swept_path_conflicts(ego_x, ego_y, cx, cy, ego_yaw, packet):
                return offset
    return 0.0


class M5Engine:
    def __init__(self) -> None:
        self.fsm = self._create_fsm()
        self._last_state: str | None = None
        # Rolling distance-to-ego history per track_id, ONLY for the
        # class='unknown'/low-confidence candidates the artifact filter
        # cares about (see _update_artifact_history) -- bounded by however
        # many such candidates are currently active, self-pruning as track
        # ids churn (M3_Pipeline README 7B.7).
        self._artifact_distance_history: dict[int, deque] = {}
        # Raw TTC/risk inputs behind the last decision -- NOT part of the
        # M1-facing DecisionPacket/UDP contract (spec's wire format is
        # fixed), only exposed here for m5_server.py to fold into the
        # RECORDED decision log, so "why did the FSM pick this state" is
        # visible after the fact without re-deriving it from raw obstacle
        # data every time (needed to debug a live collision: was TTC too
        # short, was the obstacle not seen in time, etc.).
        self.last_diagnostics: dict = {}
        # Direction-lock for AVOID (see _plan_path's own comment): None
        # whenever the engine isn't currently mid-AVOID, so the NEXT AVOID
        # episode (a different obstacle, or the same one re-approached
        # later) always picks a fresh side rather than reusing a stale one.
        self._avoid_sign_lock: float | None = None

    @staticmethod
    def _create_fsm():
        """config.USE_STATEFLOW_FSM=True: the real MATLAB/Simulink
        Stateflow chart (fsm_stateflow.StateflowFSM) -- spec section 3's
        literal requirement, decision-table-identical to BehaviorFSM
        (verified: tests/test_fsm_stateflow_parity.py). Falls back to the
        pure-Python BehaviorFSM reference, loudly logged, on ANY
        initialization failure (no MATLAB install, no license, a
        corrupted .slx, matlabengine not installed in this venv, ...) --
        a decision-engine startup failure must never take down the whole
        planning loop. False (the default): BehaviorFSM directly, no
        MATLAB involved at all."""
        if not config.USE_STATEFLOW_FSM:
            return BehaviorFSM()
        try:
            from fsm_stateflow import StateflowFSM

            return StateflowFSM()
        except Exception as exc:
            LOGGER.warning(
                "config.USE_STATEFLOW_FSM=True but the Stateflow chart failed to "
                "initialize (%s: %s) -- falling back to the pure-Python BehaviorFSM "
                "reference implementation.", type(exc).__name__, exc,
            )
            return BehaviorFSM()

    def _update_artifact_history(self, packet: ObstaclePacket, ego_x: float, ego_y: float) -> None:
        """Appends this tick's distance-to-ego for every class='unknown',
        LiDAR-only-confidence obstacle (the artifact filter's own class/
        confidence gate -- anything else is irrelevant to it and not worth
        tracking), and drops any track_id no longer present so the dict
        stays bounded by currently-active candidates, not the whole run's
        history of track ids."""
        current_ids = set()
        for obs in packet.obstacles:
            if obs.class_name != "unknown" or obs.confidence > config.SENSOR_ARTIFACT_MAX_CONFIDENCE:
                continue
            current_ids.add(obs.track_id)
            history = self._artifact_distance_history.setdefault(
                obs.track_id, deque(maxlen=config.SENSOR_ARTIFACT_STABILITY_WINDOW_TICKS),
            )
            history.append(math.hypot(obs.pos_x - ego_x, obs.pos_y - ego_y))
        for stale_id in set(self._artifact_distance_history) - current_ids:
            del self._artifact_distance_history[stale_id]

    def step(self, packet: ObstaclePacket, lane_width_m: float | None = None) -> DecisionPacket:
        """`lane_width_m`: the current lane's width, reported live by M1
        (SPEC.md section 4's M1->M5 map interface -- see config.
        M1_LANE_INFO_UDP_PORT). None (the default) means no lane data has
        arrived yet, preserving this engine's original, unconstrained
        AVOID behavior exactly -- this parameter is purely additive."""
        t0 = time.perf_counter()

        ego_x, ego_y = packet.ego_position[0], packet.ego_position[1]
        ego_vx, ego_vy = packet.ego_velocity[0], packet.ego_velocity[1]
        ego_yaw = math.radians(packet.ego_yaw_deg)
        ego_speed = math.hypot(ego_vx, ego_vy)

        # History updated BEFORE filtering: a track needs to accumulate
        # samples across ticks before it can ever be judged "stable", so
        # this must run even on ticks where the object doesn't (yet) get
        # filtered.
        self._update_artifact_history(packet, ego_x, ego_y)

        # Filtered once here so it flows through to every use below (TTC,
        # now/future costmap risk, and AVOID's nearest-obstacle direction
        # in _plan_path) -- see _is_likely_sensor_artifact's docstring.
        # Statics need the same filter (the actual live incident's phantom
        # tracks were is_static=True, routed here by M4, not into
        # obstacles -- caught by testing, see planner.py's own test suite).
        n_before = len(packet.obstacles) + len(packet.statics)
        packet.obstacles = [
            o for o in packet.obstacles
            if not _is_likely_sensor_artifact(o, ego_x, ego_y, ego_vx, ego_vy, self._artifact_distance_history.get(o.track_id))
        ]
        packet.statics = [s for s in packet.statics if not _is_likely_sensor_artifact_static(s, ego_x, ego_y)]
        n_filtered_artifacts = n_before - len(packet.obstacles) - len(packet.statics)

        min_ttc = compute_min_ttc(
            (ego_x, ego_y), (ego_vx, ego_vy),
            [((o.pos_x, o.pos_y), (o.vel_x, o.vel_y)) for o in packet.obstacles],
        )
        now_risk = _now_footprint_risk(ego_x, ego_y, ego_yaw, packet)
        future_risk = _future_footprint_risk(ego_x, ego_y, ego_vx, ego_vy, ego_yaw, packet)
        self.last_diagnostics = {
            "min_ttc": min_ttc, "now_risk": now_risk, "future_risk": future_risk,
            "n_obstacles": len(packet.obstacles), "n_statics": len(packet.statics),
            "n_filtered_artifacts": n_filtered_artifacts,
        }

        new_state = self.fsm.step(min_ttc, now_risk, future_risk)
        replan_triggered = new_state != self._last_state
        self._last_state = new_state

        target_speed = TARGET_SPEED_BY_STATE[new_state]
        # Position-agnostic upper bound: even a perfectly-centered ego could
        # achieve at most this much offset to either side before running off
        # the OTHER edge. Approximate on purpose -- M5 still has no actual
        # centerline/current-offset data (spec section 4's boundary), just
        # enough to catch a REQUIRED clearance the lane clearly cannot
        # provide regardless of where the ego actually sits in it.
        max_achievable_offset_m = (
            None if lane_width_m is None else max(0.0, lane_width_m / 2.0 - config.EGO_FOOTPRINT_RADIUS_M)
        )
        trajectory, target_speed = self._plan_path(
            new_state, ego_x, ego_y, ego_vx, ego_vy, ego_yaw, ego_speed, target_speed, packet,
            max_achievable_offset_m,
        )

        computation_time_ms = (time.perf_counter() - t0) * 1000.0
        return DecisionPacket(
            timestamp=packet.timestamp, frame_id=packet.frame_id,
            behavior_state=new_state, emergency_stop=(new_state == "EMERGENCY_BRAKE"),
            target_speed=target_speed, trajectory=trajectory,
            replan_triggered=replan_triggered, computation_time_ms=computation_time_ms,
        )

    def _plan_path(self, state: str, ego_x: float, ego_y: float, ego_vx: float, ego_vy: float,
                    ego_yaw: float, ego_speed: float, target_speed: float, packet: ObstaclePacket,
                    max_achievable_offset_m: float | None = None) -> tuple[list[Waypoint], float]:
        """Returns (trajectory, target_speed) rather than just the trajectory
        -- the early-stop fix below can override target_speed (moving
        NO_SWERVE_CLASSES obstacle -> 0.0), and the caller (step(), for the
        DecisionPacket.target_speed field M1 actually reads) needs to see
        that override, not the pre-override value it passed in."""
        heading = (math.cos(ego_yaw), math.sin(ego_yaw))

        lateral_offset_m = 0.0
        # EMERGENCY_BRAKE fix (live-run finding): a pedestrian occluded until
        # very close (this scenario's own "sudden appearance" design) can
        # push TTC straight past AVOID's window and into EMERGENCY_BRAKE on
        # the very first tick it's seen at all -- AVOID never gets a turn,
        # and M1's own emergency-stop handling hard-zeroes steer (brake
        # only, by design), so the ego just stops dead, however close that
        # ends up being. Confirmed live: repeated brief contact while
        # cycling EMERGENCY_BRAKE (steer=0, holds until the brake-stuck
        # detector fires) -> STUCK_RECOVERY (a short reverse) -> creeps
        # forward -> back into EMERGENCY_BRAKE range -> repeat, for most of
        # a run, never clearing the pedestrian. Computing the same escape
        # offset for EMERGENCY_BRAKE as AVOID (M1's own apply() now steers
        # toward it while still braking at full force -- see its own
        # comment) means the ego brakes AND steers away whenever there's
        # room to, instead of only ever stopping straight.
        if state in ("AVOID", "EMERGENCY_BRAKE"):
            # Checked BEFORE picking `nearest` and independently of it (see
            # _has_nearby_moving_no_swerve_obstacle's own docstring for the
            # live wedged-against-two-obstacles finding this fixes) -- a
            # single flickering "nearest" pick must never be able to turn
            # this override on and off tick to tick.
            if _has_nearby_moving_no_swerve_obstacle(ego_x, ego_y, ego_vx, ego_vy, packet):
                target_speed = 0.0

            nearest = _nearest_risky_obstacle_direction(ego_x, ego_y, packet)
            if nearest is not None:
                # class_name/speed_mps (the last two fields) drive the
                # early-stop override too, but via _has_nearby_moving_no_
                # swerve_obstacle above (which checks ALL obstacles, not just
                # this single nearest pick) -- not used again here.
                direction, obstacle_half_width, _, _ = nearest
                # Direction/sign only, from the same bearing-angle geometry
                # as before (which side is the obstacle on?) -- the OLD bug
                # was using this same lateral_component as the OFFSET
                # MAGNITUDE too (2.5 * sin(bearing)), which is why a
                # shallow-angle obstacle produced almost no swerve. The
                # 0.04 threshold is unchanged from before (the old
                # abs(2.5*x) > 0.1 tie-break, divided through by 2.5): below
                # it the bearing is too close to dead-ahead to trust its
                # sign, so default to a fixed side, same as always.
                lateral_component = direction[0] * -heading[1] + direction[1] * heading[0]
                raw_sign = 1.0 if abs(lateral_component) <= 0.04 else math.copysign(1.0, lateral_component)
                # Direction-lock fix (live-run finding): a MOVING obstacle
                # crossing the ego's path (a pedestrian mid-crossing, unlike
                # every static obstacle this bearing check was designed
                # around) sweeps this bearing from one side to the other as
                # it crosses, flipping raw_sign tick over tick -- confirmed
                # live as a violent steering oscillation (+0.89 <-> -0.78,
                # every few ticks, a 4m+ lateral zigzag) that never actually
                # converges on clearing the obstacle. Once a swerve direction
                # is chosen for a continuous AVOID/EMERGENCY_BRAKE episode,
                # keep it -- only a fresh entry into either (self.
                # _avoid_sign_lock reset to None in the `else` branch below
                # whenever the engine is in neither) picks a new one. The
                # lock deliberately survives an AVOID<->EMERGENCY_BRAKE
                # transition either direction: TTC dropping further mid-
                # swerve, or a sudden EMERGENCY_BRAKE-first entry (see this
                # function's own EMERGENCY_BRAKE comment) both continue
                # whatever side is already committed to, or pick a fresh one
                # if this is the first tick either state has been active.
                if self._avoid_sign_lock is None:
                    self._avoid_sign_lock = raw_sign
                sign = self._avoid_sign_lock
                required = _required_avoid_offset_m(obstacle_half_width)
                candidate_offset = max(-config.AVOID_MAX_OFFSET_M, min(config.AVOID_MAX_OFFSET_M, sign * required))

                # Early-stop fix (explicit user direction; revised twice after
                # live-run findings -- see _has_nearby_moving_no_swerve_
                # obstacle's own docstring for the second one): for a MOVING/
                # living obstacle -- a pedestrian, cattle, another vehicle
                # actually underway -- braking early and letting it clear is
                # safer than relying on a lane-change-risking swerve. The
                # actual lever that matters is WHEN braking starts, not
                # whether swerving is allowed: forcing target_speed to 0.0
                # (done above, before `nearest` is even picked) the moment
                # AVOID/EMERGENCY_BRAKE is entered for a moving NO_SWERVE_
                # CLASSES obstacle makes the quintic planner decelerate from
                # AVOID's very first tick (TTC<=4s) instead of coasting at
                # AVOID_SPEED_MPS until EMERGENCY_BRAKE's 1.5s cutoff -- real,
                # early stopping distance instead of a last-instant slam. The
                # lateral offset computation below is left completely
                # unrestricted (same ceiling, same escalation as any other
                # obstacle) so a genuine swerve is still available as a
                # fallback for whatever TTC doesn't leave enough room to stop
                # in. The speed gate (config.NO_SWERVE_MIN_SPEED_MPS) still
                # matters: a PARKED bicycle or a stopped car is the same
                # class but will never clear on its own, so it's exempt from
                # this override and keeps approaching at the state's normal
                # target speed while swerving around it, same as any other
                # stationary hazard.
                lateral_offset_m = _pick_safe_avoid_offset(
                    candidate_offset, ego_x, ego_y, ego_yaw, packet, max_achievable_offset_m,
                )
        else:
            self._avoid_sign_lock = None

        target_dist = target_speed * config.PLAN_HORIZON_S
        target_x = ego_x + heading[0] * target_dist - heading[1] * lateral_offset_m
        target_y = ego_y + heading[1] * target_dist + heading[0] * lateral_offset_m
        target_vx, target_vy = heading[0] * target_speed, heading[1] * target_speed

        planned = plan_quintic_path(
            start_xy=(ego_x, ego_y), start_vel_xy=(ego_vx, ego_vy), start_acc_xy=(0.0, 0.0),
            target_xy=(target_x, target_y), target_vel_xy=(target_vx, target_vy), target_acc_xy=(0.0, 0.0),
            horizon_s=config.PLAN_HORIZON_S, n_points=config.N_WAYPOINTS,
        )
        waypoints = [Waypoint(x=w.x, y=w.y, v_target=min(w.v_target, target_speed) if target_speed > 0 else 0.0, yaw=w.yaw, t=w.t)
                     for w in planned]
        return waypoints, target_speed
