"""M5 configuration: transport, planning rate, costmap/footprint geometry,
FSM thresholds, quintic-planner limits. Flat module, matching M2/M3/M4's
config.py style.

Frames/units, resolved (M5's design doc left these as open items to
settle): WORLD frame everywhere (matches M1/M3/M4's own convention end to
end -- zero silent conversion at any hop), meters, radians for yaw/heading.
M1's control executor converts to CARLA's own units only at the final
`carla.VehicleControl` call, never before.
"""
from __future__ import annotations

# --- Transport ---
M4_UDP_HOST = "0.0.0.0"
M4_UDP_PORT = 5004         # M4's output/m5_udp_sender.py sends here
M1_UDP_HOST = "127.0.0.1"
M1_UDP_PORT = 5006         # M1's control executor listens here

# SPEC.md section 4's "M1 -> M5 interface [that] does not exist yet and must
# be defined and built" -- the static map geometry M5 has no other source
# for (M3 provides no lane/road-edge information). Minimal first version:
# just the current lane's width, once per M1 tick -- enough for AVOID to
# recognize a required clearance the lane physically cannot provide BEFORE
# committing to it, rather than relying on M1's clamp to silently reduce an
# infeasible offset down to whatever fits (live incident: cattle_crossing/
# dense_market repeatedly committing to the same undersized-by-the-lane
# offset, colliding, recovering, and immediately repeating). Optional by
# design -- M5Engine.step()'s lane_width_m defaults to None, so M5 behaves
# exactly as before (unconstrained) until a value actually arrives.
M1_LANE_INFO_UDP_HOST = "0.0.0.0"
M1_LANE_INFO_UDP_PORT = 5007   # M1_Pipeline/m5_control.py sends the current lane width here

PLANNING_HZ = 10.0
PLANNING_PERIOD_S = 1.0 / PLANNING_HZ

# --- Recorder ---
LOG_DIR = "logs"
DECISIONS_LOG_FILE = "m5_decisions.jsonl"

# --- Sensor-artifact filter (live CARLA testing finding) ---
# A live smoke test found two persistent class="unknown", LiDAR-only
# confidence, near-zero-velocity tracks sitting 1.95m/3.55m from the ego
# from near the start of a run -- ground truth confirmed nothing real was
# within 11m. Consistent with a LiDAR self-detection artifact (a reflection
# off the ego's own body), not a real hazard: it caused a permanent false
# RED costmap reading, which froze the ego in EMERGENCY_BRAKE, which
# triggered scenario3.py's own ego-stuck recovery teleport -- which then
# dropped the ego next to a real vehicle, causing an actual collision. M3
# is frozen (M3_Pipeline/README.md) so this can't be filtered at the
# perception source -- filtered here instead, gated on THREE converging
# signals (see planner.py's _is_likely_sensor_artifact) so a genuine close
# call is never filtered regardless of distance.
#
# A SECOND live incident (same track_id) found the same ghost's reported
# velocity is fundamentally UNRELIABLE, not just "near zero": it climbed
# smoothly (0.19 -> 2.90 m/s) then decayed smoothly back down over 40+
# frames, INCLUDING a 20+ frame stretch where the ego was completely
# stationary the whole time -- i.e. the reading has no real relationship to
# the ego's own motion at all (a Kalman-filter noise-climb/decay artifact,
# the same pattern multimodal/features.py in M4_Pipeline already documents
# and fixes for M4's OWN prediction math, never applied to the raw velocity
# M4 forwards to M5). Checked and confirmed by replaying the exact recorded
# incident: NEITHER absolute-speed NOR relative-to-ego-speed ever drops
# below SENSOR_ARTIFACT_MAX_SPEED_MPS during the window this ghost is
# actually active -- no velocity-based check catches it. The one signal
# that DOES hold throughout: its DISTANCE to the ego stays within a narrow
# band (1.2-2.7m) for the entire 40+-frame window, regardless of what its
# velocity claims. SENSOR_ARTIFACT_STABILITY_* below implements that check
# (needs a short rolling history per track_id, kept in M5Engine itself --
# see planner.py's M5Engine._update_artifact_history). The velocity check
# is kept alongside it (a genuine simple case -- e.g. the FIRST incident's
# statics -- doesn't need to wait for a stability window to build up).
SENSOR_ARTIFACT_MAX_CONFIDENCE = 0.4     # M3's own "0.4 = LiDAR-only, no camera corroboration" marker
SENSOR_ARTIFACT_MAX_SPEED_MPS = 0.5      # "sitting there" RELATIVE TO THE EGO, not actively closing

# Live incident's ghost stayed within a ~1.5m band (1.2-2.7m) for 40+ CARLA
# frames (~2s at DT=0.05) -- window/min-samples are in M5Engine.step() TICKS
# (~10Hz, config.PLANNING_HZ), not CARLA frames, since that's the rate this
# history is actually built at.
SENSOR_ARTIFACT_STABILITY_WINDOW_TICKS = 15   # ~1.5s of M5 ticks
SENSOR_ARTIFACT_STABILITY_MIN_SAMPLES = 8     # require real history before judging "stable" -- don't flag a brand-new track on 1-2 samples
SENSOR_ARTIFACT_STABILITY_BAND_M = 2.0        # observed live range was a 1.5m band -- some margin above that
SENSOR_ARTIFACT_EXCLUSION_RADIUS_M = 4.0  # covers both observed instances (1.95m, 3.55m) with margin

# THIRD live incident (dense_market, class!='unknown' this time -- the two
# checks above never applied): track_id=58 was born class="car" the instant
# the ego first made contact with a real parked SLOW_CAR, with multi-sensor
# ("camera","lidar","radar") confidence 0.55 -- ABOVE SENSOR_ARTIFACT_MAX_
# CONFIDENCE, so the existing confidence/class gate never touches it. Its
# LiDAR-cluster extent (M3's own axis-aligned width/length) never exceeded
# 0.54m in its largest dimension across its full ~95-tick/15s life (a real
# car's bumper-corner fragment seen at point-blank/occluded range), yet it
# was picked as M5's single "nearest obstacle" the entire time, anchoring
# every AVOID offset calculation at a required clearance sized for a ~0.2m
# object instead of the real ~1.8m-wide car it was actually touching.
# max(width, length) below this threshold is physically implausible for any
# of VEHICLE_LIKE_CLASSES (planner.py) -- even a bicycle, the smallest of
# them, is ~1.7m in its long dimension -- so this check applies regardless
# of confidence/class-unknown status and needs no rolling history (unlike
# the checks above, the object was already implausibly small from birth).
SENSOR_ARTIFACT_MIN_VEHICLE_SIZE_M = 1.0

# --- TTC / Stateflow-equivalent FSM thresholds (spec section 3) ---
# TTC <= this forces EMERGENCY_BRAKE unconditionally, no hysteresis --
# section 3's own example value.
TTC_EMERGENCY_S = 1.5
TTC_AVOID_S = 4.0            # below this (and above EMERGENCY), AVOID is considered
TTC_FOLLOW_S = 8.0           # below this, FOLLOW (same-lane car ahead, no lateral conflict) is considered

# Hysteresis (section 3: "missing transition arrows... deliberately lock
# the vehicle in safety states to prevent... whiplash"): once in a more
# cautious state, TTC/costmap risk must clear by this margin before
# stepping back down one state, not just cross the raw threshold again.
HYSTERESIS_MARGIN_S = 0.5

# --- 2D local costmap (spec section 2) ---
COSTMAP_HALF_EXTENT_M = 40.0     # grid covers [-40, 40] around the ego in both axes
COSTMAP_RESOLUTION_M = 0.5       # meters per cell
COSTMAP_RED_RADIUS_M = 2.0       # obstacle inflation radius -> "Critical"
COSTMAP_YELLOW_RADIUS_M = 5.0    # -> "Medium"
COSTMAP_GREEN_RADIUS_M = 9.0     # -> "Low"

# Live testing (5-scenario validation pass) found the ego braking/creeping
# with nothing actually ahead of it: with no lane/drivable-space source
# (spec section 4, still deferred), a plain circular distance check made a
# roadside cone/sign/parked vehicle 1.6-2.8m to the SIDE read as the same
# risk as something that close directly AHEAD -- these scenarios
# deliberately place static clutter just off the lane. FOOTPRINT_LATERAL_
# STRETCH (see costmap.py's anisotropic_offset) scales up the effective
# distance of purely-lateral offsets before classifying risk, so something
# beside the car needs to be markedly closer than something ahead to read
# as the same risk level; a genuine head-on hazard's along-heading distance
# is never stretched, so this doesn't blunt real reactions at all.
FOOTPRINT_LATERAL_STRETCH = 2.5

# --- Ego 3-circle footprint (spec section 2) ---
# Front/center/rear circles along the vehicle's own heading axis, radius
# covering half the vehicle's width (a generic sedan-ish default; M1 can
# override per-vehicle-model if it knows its own bounding box).
EGO_LENGTH_M = 4.5
EGO_WIDTH_M = 2.0
EGO_FOOTPRINT_RADIUS_M = EGO_WIDTH_M / 2.0 * 1.15  # small margin over half-width
EGO_FOOTPRINT_OFFSETS_M = (-EGO_LENGTH_M / 2.0 + EGO_FOOTPRINT_RADIUS_M, 0.0, EGO_LENGTH_M / 2.0 - EGO_FOOTPRINT_RADIUS_M)

# --- AVOID's minimum lateral clearance ---
# Live 5-scenario validation (dense_market, 3 collisions) found AVOID's old
# offset formula (2.5m * sin(bearing angle to the nearest obstacle)) could
# produce as little as ~0.25-1.3m of actual lateral displacement -- nowhere
# near enough to clear an obstacle's own width plus the ego's, because
# sin(bearing) was never a clearance distance in the first place, just
# whatever fraction of 2.5m the geometry happened to produce. This is the
# real safety buffer added on top of "just barely not touching" when
# planner.py's _required_avoid_offset_m computes the new, size-aware
# minimum: obstacle_half_width + EGO_FOOTPRINT_RADIUS_M + this margin.
AVOID_SAFETY_MARGIN_M = 0.5
# Outer cap on AVOID's lateral offset -- M5 has no lane/map source of its
# own (spec section 4), so this is a conservative generic bound ("a
# typical lane's worth of room"), not a real lane-boundary query (that
# stays M1's job, via its own lane-clamp in m5_control.py). Unchanged
# from the value AVOID always used; a required clearance larger than this
# gets capped here, same as before.
AVOID_MAX_OFFSET_M = 2.5

# Live cattle_crossing finding: once the ego is already touching/wedged
# against an obstacle, _pick_safe_avoid_offset's candidate offset AND its
# mirror can BOTH read RED (the obstacle's own COSTMAP_RED_RADIUS_M=2.0m
# contact radius can cover both lateral points from the same near-zero
# ego-to-obstacle distance), so it fell straight through to 0.0 -- a
# dead-straight path back into the very obstacle it was trying to clear.
# Confirmed live: 146-346 repeated collisions with the SAME actor across a
# single run once wedged, recovering only via M1's capped stuck-recovery
# retries, not this function. Before giving up, _pick_safe_avoid_offset now
# also tries larger magnitudes (candidate_offset + N * this step, both
# sides) up to whatever's actually available (max_achievable_offset_m, or
# AVOID_MAX_OFFSET_M above when no lane data has arrived yet) -- the
# "required" offset only accounts for clearing the identified obstacle's
# own width, not the extra margin actually needed once already overlapping
# it. Half the RED radius: small enough that a couple of steps meaningfully
# probe the space between "required" and "the whole lane", not one giant
# jump that skips past a gap that would have worked.
AVOID_OFFSET_ESCALATION_STEP_M = 1.0

# Live cattle_crossing finding, per explicit user direction: swerving
# around a MOVING/LIVING obstacle (a pedestrian mid-crossing, cattle, or
# another vehicle) risks a lane change or clipping a second obstacle/the
# sidewalk to make room -- braking/slowing and letting it clear is the
# safer response for exactly these classes, unlike a stationary roadside
# hazard (a cone, pothole, sign) where steering around it is the only way
# to keep moving at all. classes.yaml's own 14-class taxonomy names every
# class this applies to; "unknown" (M3's own fallback for anything low-
# confidence or unclassified) is deliberately NOT included here -- an
# unidentified obstacle keeps the existing swerve-first behavior, since
# there's no classification to trust to make this call either way.
NO_SWERVE_CLASSES = frozenset({
    "pedestrian", "animal", "bicycle", "car", "motorcycle", "bus", "tempo", "truck", "rickshaw",
})
# A parked bicycle or a stopped car needs to be swerved around like any
# other stationary hazard -- it will never "clear" on its own the way an
# actually-moving one will, so NO_SWERVE_CLASSES only applies above this
# speed. Comfortably above sensor/prediction noise on a truly stationary
# track (tests/test_planner.py's own near-stationary regression case is
# ~0.02 m/s) and well below any real pedestrian walking (~1.4 m/s) or
# vehicle speed.
NO_SWERVE_MIN_SPEED_MPS = 0.3

# --- Target speeds per FSM state (MATLAB Function block equivalent --
# see fsm.py's docstring for why this is plain Python here, not Simulink) ---
CRUISE_SPEED_MPS = 10.0
FOLLOW_SPEED_MPS = 6.0
AVOID_SPEED_MPS = 4.0
YIELD_SPEED_MPS = 2.0
EMERGENCY_SPEED_MPS = 0.0

# --- Quintic polynomial trajectory sampler (spec section 2) ---
PLAN_HORIZON_S = 3.0
N_WAYPOINTS = 16   # spec section 1: "exactly 16 spatial-temporal waypoints"

# --- Decision engine backend (spec section 3) ---
# True (the project/demo default): M5Engine uses the real MATLAB/Simulink
# Stateflow chart (fsm_stateflow.StateflowFSM, backed by
# M5_Pipeline/matlab/behavior_fsm.slx) as the decision engine, matching
# spec's literal requirement. False: fsm.py's BehaviorFSM, a pure-Python
# state machine with identical decision-table logic.
#
# Constructing an M5Engine starts a MATLAB engine when this is True
# (10-30s) -- tests/conftest.py forces this back to False for the whole
# default test suite regardless of this setting (test_planner.py builds a
# fresh M5Engine 12 times; day-to-day `pytest tests/` needs to stay fast
# and MATLAB-free). tests/test_fsm_stateflow_parity.py is where the real
# chart gets exercised, via fsm_stateflow.StateflowFSM directly rather
# than through this flag, gated behind its own `matlab_parity` pytest
# marker. M5Engine falls back to BehaviorFSM automatically (loudly logged)
# if the engine/model fails to initialize even when this is True -- a
# decision-engine startup failure must never take down the whole planning
# loop.
USE_STATEFLOW_FSM = True
