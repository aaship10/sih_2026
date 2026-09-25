"""M1's control executor (M5_Pipeline's spec section 1b -- replaces the old
M6 role, new work in M1). The ONLY place that calls `ego.apply_control()`
while M5 is driving: scenario3.py must disable the ego's own
`set_autopilot(True, ...)` in this mode (see main()'s `--drive-mode`
handling) -- two writers issuing control the same tick makes the vehicle
flip between "go" and "stop" every frame, per the spec's own warning.

Non-blocking UDP listener on M5_Pipeline.config.M1_UDP_PORT (5006). Each
CARLA tick (called once per `scenario.step()`, i.e. once per DT=0.05s,
faster than M5's own 10Hz), takes the newest queued M5 message if any and
holds it otherwise (M5 runs slower than the CARLA tick -- spec section 1b).
If nothing fresh has arrived within STALENESS_TIMEOUT_S, falls back to a
controlled brake, never to "keep going" with a stale command.

Frames/units: M5's DecisionPacket is WORLD frame, meters, radians (see
M5_Pipeline/config.py's own resolution of the spec's open frame/unit
items) -- converted to CARLA's own units (yaw in DEGREES, steer normalized
to [-1, 1]) only here, at the final control application.

Curb-strike fix (live-smoke-test finding: repeated [COLLISION] ...
static.road while M5 drove): M5's planner.py has no lane/map data (spec
section 4 -- deliberately deferred), so its CRUISE/FOLLOW/YIELD path target
is a pure straight-line continuation of the ego's current heading, which
walks the vehicle off any road curve. A steering SIGN inversion was
suspected and independently ruled out (verified against CARLA's own
shipped agents/navigation/controller.py reference implementation -- this
module's pure-pursuit sign already matches CARLA's convention). The actual
fix, entirely local to M1 (M1 has direct CARLA map access M4/M5
structurally don't): snap the pure-pursuit lookahead target onto the real
lane centerline via world.get_map().get_waypoint() for every state EXCEPT
AVOID (whose ~2.5m deliberate lateral offset a lane-snap would cancel),
plus a speed-scaled lookahead (the old fixed 4m was only a ~0.4s preview at
cruise speed -- a known pure-pursuit oscillation source) and the vehicle's
own real max-steer-angle instead of a guessed constant.

AVOID lane-departure fix (second live-run finding, after the above: with
the sensor-artifact freeze fixed -- see M5_Pipeline/config.py -- AVOID
actually started engaging, and promptly produced [COLLISION] ...
static.sidewalk / traffic.traffic_light hits): AVOID's raw target is
skipped from the full lane-snap above for a reason -- snapping it back to
centerline would cancel the deliberate avoidance offset entirely -- but
with NO correction at all, planner.py's fixed ~2.5m lateral push has no
way to know the current lane's actual width and can push the target clean
off the road. Fix: CLAMP (not snap) AVOID's lateral offset to stay within
the current lane's real width (queried the same way as the snap, via
get_waypoint()'s own `lane_width`), preserving the avoidance direction and
as much magnitude as the lane can actually accommodate.
"""
from __future__ import annotations

import json
import math
import socket
import time
from dataclasses import dataclass, field

M1_UDP_HOST = "0.0.0.0"
M1_UDP_PORT = 5006
STALENESS_TIMEOUT_S = 0.5

# SPEC.md section 4's M1 -> M5 map interface (M5_Pipeline/config.py's
# matching M1_LANE_INFO_UDP_* constants) -- M1 is the only pipeline with
# real CARLA map access (see the AVOID lane-clamp above), so it reports the
# ego's current lane width to M5 every tick. Lets AVOID recognize a
# required clearance the lane can't provide before committing to it,
# instead of M1's own clamp silently shrinking an infeasible offset down to
# whatever fits.
M5_LANE_INFO_UDP_HOST = "127.0.0.1"
M5_LANE_INFO_UDP_PORT = 5007

# Speed-scaled pure-pursuit lookahead: Ld = min + time_s * speed, clamped.
# ~0.7s preview at any speed (standard 0.5-1.0s pure-pursuit rule of thumb)
# replaces the old fixed 4m, which was only a ~0.4s preview at CRUISE's
# 10 m/s target -- a known oscillation source at that short a horizon.
LOOKAHEAD_MIN_M = 3.0
LOOKAHEAD_TIME_S = 0.7
LOOKAHEAD_MAX_M = 12.0

# Fallback only -- _query_max_steer_rad() uses the spawned vehicle's own
# real physics_control value; this is never expected to be hit for a
# normal 4-wheeled car blueprint.
MAX_STEER_RAD_FALLBACK = math.radians(70.0)

# Guards the lane-snap correction near junctions: CARLA's get_waypoint()
# is nearest-POINT projection, not path-aware, so close to an intersection
# the nearest on-road waypoint can belong to a crossing lane whose heading
# disagrees sharply with the ego's own -- skip the snap that tick rather
# than risk steering toward traffic.
MAX_SNAP_HEADING_DELTA_DEG = 60.0

# AVOID lane-clamp: how far inside the lane's own edge the vehicle's own
# body must stay -- not driving with a wheel exactly on the lane paint.
LANE_EDGE_MARGIN_M = 0.3

# Known-hazard repulsion (live dense_market finding: the ego collided with
# static.static -- raw CARLA level geometry, e.g. a wall/curb -- that never
# appears in ANY M4/M5 obstacle packet at all; the nearest actually-tracked
# object was 6.3m away at the moment of contact. M5 has no lane/map data
# (spec section 4) and genuinely cannot see this class of hazard, so no
# amount of AVOID/YIELD tuning on M5's side fixes it -- confirmed live: the
# existing stuck-recovery backed the vehicle a full ~1m away from the wall
# every time, but YIELD's straight-line-continuation plan then drove it
# right back, since M5 never learns the wall is there, repeating the cycle
# 8 times over one run. M1 already has privileged CARLA map access the
# other pipelines structurally don't (see the lane-clamp above) -- this
# extends that same privilege to the ego's OWN collision sensor (set up
# here independently of whatever collision sensor a scenario script owns
# for its own logging; CARLA allows multiple listeners on one actor),
# remembering where contact actually happened and nudging the pure-pursuit
# target away from those exact world locations for the rest of the run,
# regardless of what M5's blind trajectory says.
KNOWN_HAZARD_RADIUS_M = 2.0  # lateral separation to maintain from a remembered contact point
KNOWN_HAZARD_LONGITUDINAL_WINDOW_M = 6.0  # only relevant while the target is actually near the hazard along the lane, not miles down the road
KNOWN_HAZARD_DEDUP_DIST_M = 1.0  # collisions within this distance of an already-remembered point don't add a second entry (sustained contact re-fires the sensor every tick)
KNOWN_HAZARD_MAX_REMEMBERED = 8  # bounded so a long run's memory can't grow without limit

# Longitudinal PID gains, tuned for a throttle/brake in [0, 1] against a
# speed error in m/s (typical urban target speeds, a few m/s to ~10 m/s).
# No evidence of longitudinal oscillation in the live smoke test (clean
# 300/300-tick run, correct terminal EMERGENCY_BRAKE) -- left unchanged.
PID_KP, PID_KI, PID_KD = 0.4, 0.05, 0.05
INTEGRAL_CLAMP = 5.0

STALE_BRAKE = 0.4

# Stuck recovery (live dense_market finding: a single AVOID contact against
# an obstacle the lane genuinely can't clear -- e.g. one that sits partly
# outside the marked lane's own width, confirmed via a live lane-width
# query -- pins the vehicle at near-zero speed with throttle saturated for
# the rest of the run: 172 collision-sensor events traced back to exactly
# ONE initial contact tick, the other 171 being the sensor re-firing every
# tick of sustained wedged contact, not 172 separate planning failures.
# Neither AVOID's offset math nor M1's lane-clamp/pure-pursuit can recover
# from this -- once actually touching, a near-zero-speed vehicle can't be
# steered clear; it needs to physically back away first. This is
# deliberately scoped to recovery only: it does not change WHY the initial
# contact happens (the geometric-feasibility question is a separate,
# not-yet-addressed issue), only what happens once the vehicle is
# genuinely stuck.
STUCK_SPEED_THRESHOLD_MPS = 0.15  # below this, the vehicle isn't meaningfully moving -- used by the brake-side check (is_braked_but_stuck); the throttle-side check uses the wider STUCK_PARTIAL_PROGRESS_SPEED_MPS below (see its own comment for why)
STUCK_THROTTLE_THRESHOLD = 0.5  # only counts as "stuck" if the controller is actively trying to move forward -- excludes EMERGENCY_BRAKE/STALE_FALLBACK (throttle=0) and legitimate stops, by construction
STUCK_TICKS_THRESHOLD = 20  # ~1.0s at the 20Hz CARLA tick rate -- real incidents stayed wedged for 170+ ticks, so this adds negligible delay while avoiding false triggers on a momentary dip
RECOVERY_THROTTLE = 0.5  # moderate reverse effort, not full power -- this is a disengage maneuver, not a race
RECOVERY_DURATION_TICKS = 20  # ~1.0s of reversing before handing control back to the normal M5 decision
MAX_RECOVERY_ATTEMPTS = 3  # give up after this many tries in the same stuck spell rather than reverse/forward oscillating forever if genuinely boxed in
RECOVERY_RESET_SPEED_MPS = 1.0  # driving normally again at this speed clears the attempt counter, so a later, unrelated stuck event gets a fresh set of tries

# SECOND live incident (dense_market, after the fix above landed): the
# throttle-only trigger above deliberately excludes EMERGENCY_BRAKE/
# STALE_FALLBACK (throttle=0), on the assumption that braking is always
# either a legitimate, resolving safety stop or something a reverse
# maneuver shouldn't second-guess. A live run found the gap in that
# assumption: the FSM correctly emitted EMERGENCY_BRAKE against a
# stationary parked bicycle the ego was already touching, but braking
# doesn't create clearance the way accelerating away does -- once already
# in contact, brake=1.0 just holds the vehicle wedged there forever, with
# nothing to ever trigger an escape. Confirmed via the recorded control
# log: ego position frozen exactly, speed=0.00, throttle=0.00, brake=1.00
# for 493 consecutive ticks (24.65s, over half the scenario), generating
# 511 collision-sensor re-firings, with zero recovery attempts under the
# old trigger. The SAME run's genuinely legitimate EMERGENCY_BRAKE
# episodes (waiting out an actually-resolving hazard) all cleared on their
# own within 15-33 ticks (0.75-1.65s) -- STUCK_BRAKE_TICKS_THRESHOLD below
# is set well above that observed legitimate range specifically so this
# does not interrupt an ordinary, resolving safety stop.
STUCK_BRAKE_THRESHOLD = 0.9  # near-full brake -- EMERGENCY_BRAKE/STALE_FALLBACK both command brake close to 1.0
STUCK_BRAKE_TICKS_THRESHOLD = 60  # ~3.0s -- well above the 33-tick legitimate max observed live, far below the 493-tick pathological wedge

# THIRD live incident (dense_market, after both fixes above landed): the
# recovery maneuver itself worked -- each of 4 traced episodes achieved a
# genuine 1.09-1.23m of backward separation from a wall/curb (CARLA
# "static.static", invisible to M4/M5's obstacle pipeline entirely, so
# YIELD's straight-line-continuation plan had no reason to avoid driving
# right back into it) -- but RECOVERY_RESET_SPEED_MPS was being satisfied
# by the reverse maneuver's OWN residual momentum one tick after handing
# control back, since ego_speed_mps (hypot(vx,vy)) can't tell "still
# coasting backward" from "genuinely driving away". Traced tick-by-tick:
# speed crossed 1.0 m/s on the very first post-recovery tick, then
# collapsed to ~0.08 m/s just 2 ticks later as the vehicle's own momentum
# reversed -- attempts had already been cleared to 0 by then, so the FSM
# drove it straight back and recontacted 18-32 ticks later, repeating
# indefinitely. A genuinely successful departure (separately verified live)
# sustained >1.1 m/s for at least 5+ ticks without collapsing.
# RECOVERY_CLEAR_TICKS_REQUIRED below requires the reset speed to hold for
# a short sustained window instead of one sample -- comfortably above the
# ~2-tick collapse this incident showed, comfortably below the >=5-tick
# sustain a real departure demonstrated.
RECOVERY_CLEAR_TICKS_REQUIRED = 8  # ~0.4s of sustained speed above RECOVERY_RESET_SPEED_MPS before attempts actually clears

# FOURTH live incident (highway_merge): a genuinely different flavor of
# stuck than the three above -- not "not moving at all" (STUCK_SPEED_
# THRESHOLD_MPS's 0.15 m/s), but "moving, just nowhere near what it's
# actually trying to do". YIELD's target speed is a flat 2.0 m/s
# (fsm.TARGET_SPEED_BY_STATE) the whole time it holds that state; traced
# live, the ego correctly decelerated toward it, overshot slightly to
# ~1.0 m/s, then made sustained light contact with an adjacent, itself
# near-stationary LANE vehicle (a toyota.prius creeping at 0.32->0.06 m/s)
# -- the PID kept commanding throttle=0.76-0.91 trying to climb back to
# 2.0 m/s, but contact held speed at 0.41-0.52 m/s for 40+ ticks, well
# above the 0.15 m/s "not moving" threshold, so the existing throttle-side
# trigger never saw it as stuck at all. Confirmed via a live CARLA launch
# that a normal, genuinely-unimpeded 0-to-cruise launch clears this wider
# threshold in ~10 ticks (0.5s: 9 ticks of static-friction breakaway at
# exactly 0.0 m/s, then a fast ramp) -- half of STUCK_TICKS_THRESHOLD's
# 20-tick window, so this widened threshold still leaves a real launch
# comfortable margin before ever being mistaken for a wedge.
STUCK_PARTIAL_PROGRESS_SPEED_MPS = 0.6  # throttle-side only (see is_pushing_but_stuck) -- catches "blocked, not stopped", live incident topped out at 0.52 m/s


@dataclass
class _PID:
    kp: float
    ki: float
    kd: float
    _integral: float = field(default=0.0)
    _prev_error: float | None = field(default=None)

    def step(self, error: float, dt: float) -> float:
        self._integral = max(-INTEGRAL_CLAMP, min(INTEGRAL_CLAMP, self._integral + error * dt))
        derivative = 0.0 if self._prev_error is None or dt <= 0 else (error - self._prev_error) / dt
        self._prev_error = error
        return self.kp * error + self.ki * self._integral + self.kd * derivative


def _speed_scaled_lookahead(ego_speed: float) -> float:
    return max(LOOKAHEAD_MIN_M, min(LOOKAHEAD_MAX_M, LOOKAHEAD_MIN_M + LOOKAHEAD_TIME_S * ego_speed))


def _nearest_lookahead_waypoint(trajectory: list[dict], ego_x: float, ego_y: float, lookahead_m: float) -> dict | None:
    if not trajectory:
        return None
    for wp in trajectory:
        if math.hypot(wp["x"] - ego_x, wp["y"] - ego_y) >= lookahead_m:
            return wp
    return trajectory[-1]


def _lane_ahead_point(wp_here, lookahead_m: float) -> dict | None:
    """Real point on the lane centerline lookahead_m ahead of the ego,
    walked forward via CARLA's own waypoint graph (Waypoint.next()) from
    wp_here (the ego's OWN current-lane waypoint, already queried once per
    tick for the lane-width report).

    AVOID->YIELD spiral fix (live-run finding): the old non-AVOID
    correction snapped M5's own picked lookahead point onto the nearest
    road location, but that point is itself a straight-line extension of
    the ego's CURRENT heading (see module docstring's curb-strike fix --
    M5 has no lane data at all). Right after AVOID hands off to YIELD with
    the ego's heading still meaningfully off the lane's true direction,
    M5 recomputes that straight-line pick fresh every tick FROM the
    still-drifting heading, so snapping only its position onto the road
    doesn't stop the pick itself from continuing to rotate tick over tick
    -- confirmed live as a ~6s, near-full-lock steering spiral (YIELD,
    command_age_s ~0 the whole time, so not a stale-command issue) before
    EMERGENCY_BRAKE cut it off. Walking forward from the ego's OWN
    position along the ACTUAL lane sidesteps M5's heading entirely, so it
    can't inherit an error M5 has no way to know about."""
    if wp_here is None:
        return None
    try:
        ahead = wp_here.next(lookahead_m)
    except RuntimeError:
        return None
    if not ahead:
        return None
    loc = ahead[0].transform.location
    return {"x": loc.x, "y": loc.y}


def _clamp_lateral_offset_to_lane(raw_x: float, raw_y: float, wp, half_vehicle_width_m: float) -> tuple[dict, bool]:
    """Decomposes (raw_x, raw_y) relative to wp's own centerline into an
    along-lane and a cross-lane (lateral) component, clamps only the
    lateral one to +-(lane_width/2 - half_vehicle_width - margin), and
    reconstructs the point. The perpendicular basis's sign is arbitrary
    (clamping symmetrically to +-max makes which side is "positive"
    irrelevant) -- this deliberately does NOT need to match
    VehicleControl.steer's own sign convention, unlike the pure-pursuit
    math elsewhere in this file."""
    lane_yaw_rad = math.radians(wp.transform.rotation.yaw)
    perp = (-math.sin(lane_yaw_rad), math.cos(lane_yaw_rad))
    cx, cy = wp.transform.location.x, wp.transform.location.y
    dx, dy = raw_x - cx, raw_y - cy
    lateral = dx * perp[0] + dy * perp[1]
    max_lateral = max(0.0, wp.lane_width / 2.0 - half_vehicle_width_m - LANE_EDGE_MARGIN_M)
    clamped_lateral = max(-max_lateral, min(max_lateral, lateral))
    return {"x": cx + clamped_lateral * perp[0], "y": cy + clamped_lateral * perp[1]}, clamped_lateral != lateral


def _nudge_away_from_known_hazards(waypoint: dict, wp, half_vehicle_width_m: float,
                                    known_hazards: "list[tuple[float, float]]") -> dict:
    """Pushes `waypoint` laterally away from any remembered collision
    location it's currently near (see KNOWN_HAZARD_* constants' module-level
    comment) -- a safety net independent of M5's own trajectory, which has
    no way to know these locations exist at all. Only acts on hazards both
    laterally close (within KNOWN_HAZARD_RADIUS_M) AND longitudinally
    relevant (within KNOWN_HAZARD_LONGITUDINAL_WINDOW_M along the lane) --
    a wall 40m down the road shouldn't perturb the current target. Reuses
    _clamp_lateral_offset_to_lane's own centerline decomposition (kept
    separate, not shared, so this new/less-tested path can never affect
    that function's own already-verified behavior)."""
    if wp is None or not known_hazards:
        return waypoint
    lane_yaw_rad = math.radians(wp.transform.rotation.yaw)
    along = (math.cos(lane_yaw_rad), math.sin(lane_yaw_rad))
    perp = (-math.sin(lane_yaw_rad), math.cos(lane_yaw_rad))
    cx, cy = wp.transform.location.x, wp.transform.location.y

    def decompose(x: float, y: float) -> tuple[float, float]:
        dx, dy = x - cx, y - cy
        return dx * along[0] + dy * along[1], dx * perp[0] + dy * perp[1]

    target_along, target_lateral = decompose(waypoint["x"], waypoint["y"])
    max_lateral = max(0.0, wp.lane_width / 2.0 - half_vehicle_width_m - LANE_EDGE_MARGIN_M)

    for hx, hy in known_hazards:
        hazard_along, hazard_lateral = decompose(hx, hy)
        if abs(target_along - hazard_along) > KNOWN_HAZARD_LONGITUDINAL_WINDOW_M:
            continue
        if abs(target_lateral - hazard_lateral) >= KNOWN_HAZARD_RADIUS_M:
            continue
        candidates = (hazard_lateral + KNOWN_HAZARD_RADIUS_M, hazard_lateral - KNOWN_HAZARD_RADIUS_M)
        clamped = [max(-max_lateral, min(max_lateral, c)) for c in candidates]
        # Prefer whichever side still achieves real separation after the
        # lane's own width clamps it -- if the lane is too narrow to clear
        # the hazard at all, this at least picks the less-bad side rather
        # than an arbitrary one.
        target_lateral = max(clamped, key=lambda c: abs(c - hazard_lateral))

    return {"x": cx + target_along * along[0] + target_lateral * perp[0],
            "y": cy + target_along * along[1] + target_lateral * perp[1]}


def _query_max_steer_rad(ego) -> float:
    """Real per-vehicle front max-steer-lock angle (CARLA's own wheel
    order: wheels[0]=FL, wheels[1]=FR, confirmed against CARLA's shipped
    PythonAPI/util/vehicle_physics_tester.py), replacing a guessed
    constant. Falls back to MAX_STEER_RAD_FALLBACK only if physics_control
    ever reports a non-positive angle for both front wheels (not expected
    for a 4-wheeled car, defensive only)."""
    physics = ego.get_physics_control()
    front_angles = [w.max_steer_angle for w in physics.wheels[:2] if w.max_steer_angle > 1e-3]
    return math.radians(max(front_angles)) if front_angles else MAX_STEER_RAD_FALLBACK


HALF_WIDTH_FALLBACK_M = 0.95  # ~a Tesla Model 3's own half-width; only used if bounding_box ever reports zero


def _query_half_width_m(ego) -> float:
    """Real per-vehicle half-width from CARLA's own bounding box (Y is the
    ego's lateral axis, per CARLA's X-forward/Y-right convention), used by
    the AVOID lane-clamp so how much room is left inside the lane accounts
    for the actual spawned vehicle, not a guess."""
    half_width = ego.bounding_box.extent.y
    return half_width if half_width > 1e-3 else HALF_WIDTH_FALLBACK_M


class M5ControlExecutor:
    def __init__(self, port: int = M1_UDP_PORT) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # DecisionPacket is small (16 waypoints) and never chunked, but the
        # OS default receive buffer proved insufficient for M4->M5's much
        # larger packets under load (see M5_Pipeline/m5_server.py) -- set
        # defensively here too rather than assuming this side is exempt.
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self._sock.bind((M1_UDP_HOST, port))
        self._sock.setblocking(False)
        self._last_decision: dict | None = None
        self._last_received_time: float | None = None
        self._speed_pid = _PID(PID_KP, PID_KI, PID_KD)
        self._prev_tick_time: float | None = None
        # All three need a live `ego`/`world`, neither of which exists yet
        # at __init__ time (scenario3.py constructs this executor before
        # the ego is spawned) -- cached lazily on first apply() instead.
        self._max_steer_rad: float | None = None
        self._half_width_m: float | None = None
        self._map = None
        self._stuck_tick_count = 0
        self._brake_stuck_tick_count = 0
        self._recovery_ticks_remaining = 0
        self._recovery_attempts = 0
        self._recovery_clear_tick_count = 0
        # Own collision sensor, independent of whatever a scenario script
        # sets up for its own logging -- CARLA allows multiple listeners on
        # one actor. Also lazy: needs a live `ego`.
        self._collision_sensor = None
        self._known_hazard_locations: list[tuple[float, float]] = []
        # SPEC.md section 4's M1 -> M5 map interface -- plain UDP send
        # socket, no bind/blocking-mode needed (unlike self._sock, which
        # receives).
        self._lane_info_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def _on_collision(self, event) -> None:
        """Remembers where contact actually happened, world-frame -- see
        KNOWN_HAZARD_* constants' module-level comment. Deduplicates against
        already-known points (sustained contact re-fires this every tick)
        and bounds the list so a long run's memory can't grow unbounded."""
        loc = event.transform.location
        for hx, hy in self._known_hazard_locations:
            if math.hypot(loc.x - hx, loc.y - hy) < KNOWN_HAZARD_DEDUP_DIST_M:
                return
        if len(self._known_hazard_locations) >= KNOWN_HAZARD_MAX_REMEMBERED:
            self._known_hazard_locations.pop(0)
        self._known_hazard_locations.append((loc.x, loc.y))

    def _drain_latest(self) -> None:
        while True:
            try:
                data, _addr = self._sock.recvfrom(65536)
            except BlockingIOError:
                return
            try:
                self._last_decision = json.loads(data)
                self._last_received_time = time.time()
            except json.JSONDecodeError:
                continue

    def apply(self, ego, world) -> dict:
        """Reads the ego's current pose/speed from `ego`, applies control to
        it, and returns a log record (spec section 1b: "Logging per tick:
        command age, applied control and pose")."""
        import carla  # local import: this module is imported by M1 code that already depends on carla, keeps this file importable in isolation for tests without it installed

        self._drain_latest()

        if self._max_steer_rad is None:
            self._max_steer_rad = _query_max_steer_rad(ego)
        if self._half_width_m is None:
            self._half_width_m = _query_half_width_m(ego)
        if self._map is None:
            self._map = world.get_map()
        if self._collision_sensor is None:
            bp = world.get_blueprint_library().find("sensor.other.collision")
            self._collision_sensor = world.spawn_actor(bp, carla.Transform(), attach_to=ego)
            self._collision_sensor.listen(self._on_collision)

        now = time.time()
        dt = 0.0 if self._prev_tick_time is None else max(1e-3, now - self._prev_tick_time)
        self._prev_tick_time = now

        tf = ego.get_transform()
        ego_x, ego_y, ego_z = tf.location.x, tf.location.y, tf.location.z
        ego_yaw_rad = math.radians(tf.rotation.yaw)
        vel = ego.get_velocity()
        ego_speed = math.hypot(vel.x, vel.y)

        # SPEC.md section 4's M1 -> M5 map interface: report the CURRENT
        # lane's width every tick, independent of AVOID/CRUISE/etc -- M5
        # has no other source for this (M3 provides no lane/road-edge
        # information) and needs it fresh regardless of which state it's
        # in, not just while already mid-AVOID.
        try:
            wp_here = self._map.get_waypoint(
                carla.Location(x=ego_x, y=ego_y, z=ego_z), project_to_road=True, lane_type=carla.LaneType.Driving,
            )
        except RuntimeError:
            wp_here = None  # a map-query fault must never kill the only control writer
        if wp_here is not None:
            self._lane_info_sock.sendto(
                json.dumps({"lane_width_m": wp_here.lane_width}).encode("utf-8"),
                (M5_LANE_INFO_UDP_HOST, M5_LANE_INFO_UDP_PORT),
            )

        command_age = None if self._last_received_time is None else now - self._last_received_time
        is_stale = self._last_decision is None or command_age is None or command_age > STALENESS_TIMEOUT_S

        lookahead_m = _speed_scaled_lookahead(ego_speed)
        target_correction = "NONE"
        if is_stale:
            control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=STALE_BRAKE)
            applied_state = "STALE_FALLBACK"
        elif self._last_decision["emergency_stop"]:
            # Steer-while-braking fix (live-run finding): a pedestrian
            # occluded until very close can push M5 straight into
            # EMERGENCY_BRAKE with AVOID never getting a turn at all --
            # this used to hard-zero steer here regardless of what M5 sent,
            # so the ego could only ever stop dead straight, however close
            # that ended up being. Confirmed live as repeated contact while
            # cycling EMERGENCY_BRAKE (steer=0, holds until the brake-stuck
            # detector fires) -> a stuck-recovery reverse -> creeps forward
            # -> back into EMERGENCY_BRAKE range -> repeat, for most of a
            # run, never clearing the pedestrian. M5's planner.py now
            # computes a real lateral escape offset for EMERGENCY_BRAKE too
            # (see its own comment there), so reuse the same pure-pursuit
            # steering _compute_control already does for every other state
            # -- then force throttle/brake back to full stop regardless of
            # whatever its speed-PID would otherwise have picked, since
            # stopping is still non-negotiable here, only the steering
            # decision changes.
            control, target_correction = self._compute_control(ego_x, ego_y, ego_z, ego_yaw_rad, ego_speed, dt, lookahead_m, wp_here)
            control.throttle = 0.0
            control.brake = 1.0
            applied_state = "EMERGENCY_BRAKE"
        else:
            control, target_correction = self._compute_control(ego_x, ego_y, ego_z, ego_yaw_rad, ego_speed, dt, lookahead_m, wp_here)
            applied_state = self._last_decision["behavior_state"]

        control, applied_state = self._apply_stuck_recovery(control, applied_state, ego_speed, carla)

        ego.apply_control(control)

        return {
            "t": now, "command_age_s": command_age, "applied_state": applied_state,
            "throttle": control.throttle, "steer": control.steer, "brake": control.brake,
            "ego_x": ego_x, "ego_y": ego_y, "ego_yaw_deg": tf.rotation.yaw, "ego_speed_mps": ego_speed,
            "lookahead_m": lookahead_m, "target_correction": target_correction,
            "max_steer_deg": math.degrees(self._max_steer_rad), "half_width_m": self._half_width_m,
            "stuck_tick_count": self._stuck_tick_count, "recovery_attempts": self._recovery_attempts,
            "recovery_clear_tick_count": self._recovery_clear_tick_count,
        }

    def _apply_stuck_recovery(self, control, applied_state: str, ego_speed: float, carla):
        """Overrides `control` with a brief reverse maneuver once the
        vehicle has been wedged -- pushing real forward throttle without
        making real progress (STUCK_TICKS_THRESHOLD ticks; "real progress"
        means faster than STUCK_PARTIAL_PROGRESS_SPEED_MPS, not just
        "moving at all" -- see that constant's module-level comment for the
        live incident this widening addresses, where sustained contact held
        the vehicle at 0.4-0.5 m/s, well above a literal not-moving
        threshold, while it kept trying to reach a much higher target), or
        sitting near-full-braked without actually moving (STUCK_BRAKE_TICKS_
        THRESHOLD ticks, a longer window so an ordinary, resolving safety
        stop is never mistaken for a wedge) -- see the STUCK_*/
        STUCK_BRAKE_* constants' module-level comments for the live
        incidents this addresses. Deliberately does not touch WHY the
        vehicle got stuck (that's AVOID's offset/lane-clamp chain, or the
        FSM's own risk classification -- separate questions) -- purely a
        recovery behavior for once it already has.

        `recovery_attempts` only clears once ego_speed has stayed above
        RECOVERY_RESET_SPEED_MPS for RECOVERY_CLEAR_TICKS_REQUIRED
        consecutive ticks, not on a single sample -- see that constant's
        module-level comment for the live incident (a reverse maneuver's
        own residual momentum satisfying a one-sample check, clearing
        attempts before the vehicle had actually driven away) this fixes."""
        if self._recovery_ticks_remaining > 0:
            self._recovery_ticks_remaining -= 1
            return carla.VehicleControl(throttle=RECOVERY_THROTTLE, steer=0.0, brake=0.0, reverse=True), "STUCK_RECOVERY"

        if ego_speed > RECOVERY_RESET_SPEED_MPS:
            self._recovery_clear_tick_count += 1
            if self._recovery_clear_tick_count >= RECOVERY_CLEAR_TICKS_REQUIRED:
                self._recovery_attempts = 0
        else:
            self._recovery_clear_tick_count = 0

        is_pushing_but_stuck = control.throttle > STUCK_THROTTLE_THRESHOLD and ego_speed < STUCK_PARTIAL_PROGRESS_SPEED_MPS
        is_braked_but_stuck = control.brake >= STUCK_BRAKE_THRESHOLD and ego_speed < STUCK_SPEED_THRESHOLD_MPS
        self._stuck_tick_count = self._stuck_tick_count + 1 if is_pushing_but_stuck else 0
        self._brake_stuck_tick_count = self._brake_stuck_tick_count + 1 if is_braked_but_stuck else 0

        is_stuck = (
            self._stuck_tick_count >= STUCK_TICKS_THRESHOLD
            or self._brake_stuck_tick_count >= STUCK_BRAKE_TICKS_THRESHOLD
        )
        if is_stuck and self._recovery_attempts < MAX_RECOVERY_ATTEMPTS:
            self._stuck_tick_count = 0
            self._brake_stuck_tick_count = 0
            self._recovery_attempts += 1
            self._recovery_ticks_remaining = RECOVERY_DURATION_TICKS - 1  # this tick counts as the first
            return carla.VehicleControl(throttle=RECOVERY_THROTTLE, steer=0.0, brake=0.0, reverse=True), "STUCK_RECOVERY"

        return control, applied_state

    def _compute_control(self, ego_x: float, ego_y: float, ego_z: float, ego_yaw_rad: float,
                          ego_speed: float, dt: float, lookahead_m: float, wp_here=None):
        import carla

        decision = self._last_decision
        target_speed = decision["target_speed"]
        speed_error = target_speed - ego_speed
        pid_out = self._speed_pid.step(speed_error, dt)
        throttle, brake = (min(1.0, pid_out), 0.0) if pid_out >= 0 else (0.0, min(1.0, -pid_out))

        waypoint = _nearest_lookahead_waypoint(decision["trajectory"], ego_x, ego_y, lookahead_m)
        target_correction = "NONE"
        # EMERGENCY_BRAKE now carries a real lateral escape offset too (see
        # planner.py's own comment) whenever M5 finds room for one -- treat
        # it the same as AVOID here (CLAMP the offset to the real lane,
        # don't SNAP it back to centerline) so this call site being reused
        # for emergency braking (see apply()'s own comment) doesn't cancel
        # the very offset it exists to steer toward.
        is_avoid = decision["behavior_state"] in ("AVOID", "EMERGENCY_BRAKE")
        wp = None
        if waypoint is not None and self._map is not None:
            # M5 has no lane/map data (spec section 4) so its target is
            # either a naive straight-line continuation of current heading
            # (CRUISE/FOLLOW/YIELD) or a fixed ~2.5m lateral push
            # (AVOID) with no idea how wide the actual lane is --
            # M1's job alone (M4/M5 have no map source) to reconcile
            # either against the real road via CARLA's own map.
            try:
                wp = self._map.get_waypoint(
                    carla.Location(x=waypoint["x"], y=waypoint["y"], z=ego_z),
                    project_to_road=True, lane_type=carla.LaneType.Driving,
                )
            except RuntimeError:
                wp = None  # a map-query fault must never kill the only control writer
            if wp is not None:
                heading_delta = abs((math.degrees(ego_yaw_rad) - wp.transform.rotation.yaw + 180) % 360 - 180)
                if heading_delta <= MAX_SNAP_HEADING_DELTA_DEG:
                    if is_avoid:
                        # CLAMP, not snap: snapping back to centerline would
                        # cancel AVOID's deliberate offset entirely. Keeps
                        # the avoidance direction/magnitude, just bounded
                        # to what the real lane can actually accommodate.
                        waypoint, clamped = _clamp_lateral_offset_to_lane(waypoint["x"], waypoint["y"], wp, self._half_width_m)
                        target_correction = "CLAMPED" if clamped else "NONE"
                    else:
                        lane_point = _lane_ahead_point(wp_here, lookahead_m)
                        waypoint = lane_point if lane_point is not None else {
                            "x": wp.transform.location.x, "y": wp.transform.location.y,
                        }
                        target_correction = "SNAPPED"
                # else: near a junction/lane split where the nearest-waypoint
                # projection disagrees sharply with the ego's own heading --
                # skip the correction this tick, fall back to M5's raw target.

        if waypoint is not None and wp is not None and self._known_hazard_locations:
            waypoint = _nudge_away_from_known_hazards(waypoint, wp, self._half_width_m, self._known_hazard_locations)

        if waypoint is None:
            steer = 0.0
        else:
            dx, dy = waypoint["x"] - ego_x, waypoint["y"] - ego_y
            local_x = dx * math.cos(-ego_yaw_rad) - dy * math.sin(-ego_yaw_rad)
            local_y = dx * math.sin(-ego_yaw_rad) + dy * math.cos(-ego_yaw_rad)
            ld = math.hypot(local_x, local_y)
            if ld < 1e-3:
                steer = 0.0
            else:
                # Standard pure-pursuit curvature -> steering angle, then
                # normalized to CARLA's [-1, 1] steer range using the
                # vehicle's own real max-steer-lock angle.
                curvature = 2.0 * local_y / (ld * ld)
                steer_rad = math.atan(curvature * ld)
                steer = max(-1.0, min(1.0, steer_rad / self._max_steer_rad))

        return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake), target_correction
