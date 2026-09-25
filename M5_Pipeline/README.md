# M5 Pipeline — Behavior Decision & Path Planning

Built from `SPEC.md` (kept verbatim there). M5's job: turn M4's predicted
trajectories + uncertainty into a deterministic driving decision (behavior
state, target speed, a 16-waypoint path) for M1 to execute. M5 owns ALL
TTC/collision/risk assessment — M4 computes none of it (see
`M4_Pipeline/README.md`'s "Status" section for that boundary from M4's side).

## Architecture

```
M4 (m4_server.py)
   |  UDP :5004, ObstaclePacket (output/m5_udp_schema.py)
   v
M5 m5_server.py  (listens :5004, 10Hz planning loop)
   |  schema.py         : parse the UDP JSON into dataclasses
   |  ttc.py             : Time-To-Collision, relative kinematics
   |  costmap.py          : 2D EDT-based Red/Yellow/Green risk grid
   |  footprint.py         : ego 3-circle footprint projected onto the costmap
   |  fsm.py                : CRUISE/FOLLOW/AVOID/YIELD/EMERGENCY_BRAKE decision table + hysteresis
   |  quintic.py             : quintic-polynomial path sampler (16 waypoints)
   |  planner.py (M5Engine)   : joins all of the above into one step() call
   v
   UDP :5006, DecisionPacket
   v
M1 (scenario3.py --drive-mode m5, m5_control.py)
```

`m5_dummy_server.py`-equivalent for M4 lives in `M4_Pipeline/m5_dummy_server.py`
(now a UDP listener/logger, not the old HTTP stand-in) if you want to check
M4 is sending well-formed packets without running the real M5 engine.

## Running

```powershell
cd M5_Pipeline
py -3.11 -m venv venv
.\venv\Scripts\pip install -r requirements.txt
python m5_server.py       # listens on UDP :5004, broadcasts to UDP :5006
```

Then M4 (with `M5_UDP_HOST`/`M5_UDP_PORT` in `M4_Pipeline/config.py`
pointing here), M3, M2, and M1 with `--drive-mode m5` (see
`M1_Pipeline/m5_control.py`) — same overall order as M4_Pipeline/README.md's
"Running" section, M5 added between M4 and M1.

### Offline tools

- `python -m pytest tests/` — unit tests for every module (ttc/costmap/
  footprint/quintic/fsm) plus an integration test on the full `M5Engine`.
  No UDP, no CARLA, no other pipeline stage needed.

## Open items (SPEC.md section 5), resolved

**A. Static objects.** Resolved: M4's `output/m5_udp_schema.py` splits
`is_static` tracks into their own top-level `statics` list
(`track_id`/`class`/`pos_x`/`pos_y`/`width`/`length` only — no velocity or
trajectories, there's nothing to predict for something that isn't moving),
separate from `obstacles`. `planner.py` feeds `statics` into both the "now"
and "future" costmap checks (a static object matters at every future time
the same as now) but never into TTC (TTC needs a closing relative velocity,
which is meaningless for something that can't move).

**B. Uncertainty.** Resolved: used, not ignored. `output/m5_udp_schema.py`
adds a `sigma_m` array alongside each trajectory's `points` — one combined
per-point radius (`sqrt(along_sigma^2 + lateral_sigma^2) / 2`) rather than
M4's full oriented ellipse, since M5's costmap only needs a circular
inflation radius, not the ellipse orientation. Not yet wired into
`planner.py`'s risk checks (both currently use the fixed-radius footprint
only) — the field is there and parsed (`schema.py`'s `PredictedTrajectory.
sigma_m`) for a planner-side use to add next: inflate `EGO_FOOTPRINT_RADIUS_M`
by an obstacle's own `sigma_m` at the matched time when checking the future
costmap, so a wide-uncertainty prediction gets more clearance than a tight one.

**C. M5 → M1 frame/units.** Resolved: WORLD frame, meters, radians for
`yaw`. Matches M1/M3/M4's own convention throughout (M4 already reports
`ego_position`/`ego_yaw_deg` in world frame) — one frame/unit system across
the whole loop means no silent conversion bug at any hop. `m1_control.py`
converts to CARLA's own units (yaw in degrees, steer normalized to
`[-1, 1]`) only at the final `apply_control()` call.

**D. M4 → M5 frame/fields.** Resolved: WORLD frame (same reasoning as C).
`width`/`length` = M3's axis-aligned LiDAR-cluster extent (`size[1]`,
`size[0]` respectively) passed straight through, unchanged — M5 does not
treat it as an oriented box (SPEC.md's own note that it isn't one).

**E. TTC limitation.** Resolved by NOT relying on TTC alone:
`planner.py`'s `_future_footprint_risk` checks the ego's own nominal
(constant-velocity) future path against every predicted trajectory point at
matching time, independent of whether the raw TTC formula reads that
object as "closing" right now. A predicted future conflict promotes
straight to `YIELD` even with `min_ttc = None`. See `fsm.py`'s
`test_future_yellow_triggers_yield_not_avoid`.

## Design choices and why

| Question | Choice | Why |
|---|---|---|
| Decision engine implementation | Plain Python `fsm.py`, not MATLAB/Simulink `matlabengine` | No MATLAB/Simulink installation or license in this environment; a Stateflow chart here is a fixed threshold decision table, which a Python function expresses exactly as deterministically, with zero engine-startup latency — same reasoning M4's own README gives for not using Simulink anywhere in its own math |
| Hysteresis | Escalate immediately, de-escalate only with margin | Delaying a safety escalation to avoid state "flapping" is never the right trade; de-escalating too eagerly right at a threshold is exactly the whiplash spec section 3 warns about |
| Costmap | Single fresh grid per tick, EDT-based, ego-centered | Matches SPEC.md section 2 literally; a persistent/rolling map isn't needed since M5 has no static drivable-space source of its own (SPEC.md section 4 — that's M1's job) |
| "Future" conflict check | Direct point-to-point distance at matched time, not a grid per timestep | Same Red/Yellow/Green thresholds (`costmap.classify_distance`) without paying for `config.N_WAYPOINTS` separate EDT computations per tick |
| Quintic path target | Straight continuation (CRUISE/FOLLOW/YIELD) or a fixed lateral offset toward the far side of the nearest obstacle (AVOID) | Simplest policy that still produces a real jerk-limited avoidance maneuver; no map/lane data to plan a lane-change against (same limitation M4's own lateral-mode heuristic documents) |
| M4 → M5 transport | UDP, not HTTP | SPEC.md fixes this explicitly; also matches "fire-and-forget, never block the sender" the whole M1-M4 chain already uses, but even more directly — UDP `sendto()` needs no client, no timeout, no background task |

## Bugs found while testing (not theoretical — confirmed against real recorded data)

- **UDP's 65507-byte hard datagram limit**: a single frame's obstacle
  packet from a real dense scenario (`M4_Pipeline/logs/m4_predictions.jsonl`,
  ~29 objects/frame) serializes to ~140KB — more than double the limit.
  `sendto()` would raise on every such frame; M4->M5 delivery would have
  silently failed on 338 of 353 real recorded frames. Fixed by chunking
  (`M4_Pipeline/output/m5_udp_sender.py`'s `_pack_chunks` /
  `m5_server.py`'s `_ChunkReassembler`, `tests/test_reassembly.py`).
- **OS default UDP receive-buffer size**: even after chunking (each piece
  under the hard limit), replaying those same 100 real frames over actual
  loopback UDP dropped chunks silently on 28% of multi-chunk frames with
  the OS default receive buffer — never an exception, never a log line,
  the frame just never completed reassembly. Setting `SO_RCVBUF` to 1MB on
  every UDP listener in the loop (M5's `m5_server.py`, M4's
  `m5_dummy_server.py`, M1's `m5_control.py`) eliminated every drop across
  the same test. Neither of these would show up in the in-process unit
  tests (`tests/test_reassembly.py` calls `_ChunkReassembler` directly,
  no real socket) — both were only caught by actually sending real chunks
  over a real socket and checking what came out the other side.
- **A LiDAR self-detection ghost near the ego** (`config.py`'s
  `SENSOR_ARTIFACT_*` comment has the full incident writeup) forced a
  permanent false `RED` costmap reading, freezing the ego in
  `EMERGENCY_BRAKE` until `scenario3.py`'s own stuck-recovery teleported it
  into real traffic. Took two rounds to actually close: filtering on
  absolute world-frame velocity missed a ghost that inherited the ego's own
  speed; filtering on velocity relative to the ego then missed a SECOND
  live incident where the ghost's velocity reading was independently
  unreliable (a Kalman noise-climb/decay artifact uncorrelated with the
  ego's real motion — confirmed by it staying elevated through 20+ frames
  where the ego was completely stationary). The signal that finally held
  up: the ghost's DISTANCE to the ego stayed in a narrow band the entire
  time regardless of what its velocity claimed — `planner.py`'s
  `_distance_is_stable` / `M5Engine._update_artifact_history`.
- **Isotropic (circular) risk zones with no lane/drivable-space source**
  (see the "No map/lane/drivable-space input" limitation below) made a
  roadside cone/sign/parked-vehicle 1.6-2.8m to the SIDE read as the same
  `RED` risk as something that close directly AHEAD — full 5-scenario
  validation surfaced this as the ego visibly braking/creeping with
  nothing actually blocking its lane. Fixed by rotating the costmap into
  the ego's own heading frame and stretching the lateral axis before the
  distance transform runs (`costmap.py`'s `anisotropic_offset`,
  `config.FOOTPRINT_LATERAL_STRETCH`) — along-heading (genuine head-on)
  distances are completely unaffected, only purely-lateral proximity reads
  as farther than it is in isotropic terms.
- **YIELD starving AVOID for static obstacles directly ahead.** Full
  5-scenario validation found the ego repeatedly wedging into a static
  traffic cone/bicycle in `dense_market` (393 collision-sensor contacts)
  instead of steering around it. Root cause: `planner.py`'s
  `_future_footprint_risk` fed `packet.statics` into the future check
  alongside dynamic trajectories, and `fsm.py`'s `_raw_level` checks
  `footprint_future_risk` (-> YIELD) before `footprint_now_risk`'s AVOID
  branch. A static's future risk under the ego's own straight-line
  projection is essentially always >= its now risk, so YIELD (a
  straight-line path, per this doc's own design table above) kept winning
  over AVOID — and even where AVOID DID fire from TTC alone,
  `_nearest_risky_obstacle_direction` never looked at `packet.statics`
  either, so a static-only AVOID got `lateral_offset_m=0.0`, an "AVOID"
  with no actual swerve. Fixed both: statics dropped from
  `_future_footprint_risk` (already fully covered by `_now_footprint_risk`
  re-evaluated fresh every tick — no future-only information lost, only
  the FSM branch it was steering into) and added to
  `_nearest_risky_obstacle_direction` (`tests/test_planner.py`'s
  `test_regression_static_ahead_at_yellow_risk_triggers_avoid_not_yield`).
  **This alone was not enough**: re-running `dense_market` live afterward
  showed the SAME crawl-into-obstacle signature, worse (447 vs 393
  collision-sensor contacts) — the actual colliding object
  (`vehicle.bh.crossbike`, a parked bicycle) is tracked as a DYNAMIC
  obstacle (M3 sees velocity jitter, never flags `is_static`), so the
  statics-only fix above never touched it. The real, general root cause:
  `fsm.py`'s `_raw_level` checked `footprint_future_risk` before
  AVOID-eligibility (TTC or `footprint_now_risk`) at all, and for ANY
  roughly-stationary obstacle — static or a near-zero-velocity dynamic
  track alike — its CV-predicted future position tracks its current
  position, so `future_risk ~= now_risk` and YIELD kept winning regardless
  of classification. Fixed by reordering `_raw_level` to check
  AVOID-eligibility before the future-risk YIELD branch — a present,
  avoidable hazard now always gets the state that actually steers around
  it (`tests/test_planner.py`'s
  `test_regression_near_stationary_dynamic_obstacle_ahead_triggers_avoid_not_yield`).
  Re-verified live: `dense_market` now runs 0 collisions.
- **AVOID's offset steering INTO a second obstacle (or off-road).** Direct
  visual inspection (not log-based checks — this got past every automated
  check above) found the ego colliding and never recovering in
  `village_road`, `highway_merge`, and `cattle_crossing`, plus `scenario3`
  braking correctly for a merge then slamming into the curb right after.
  Root cause: `_plan_path`'s AVOID branch computes a fixed 2.5m lateral
  offset from the nearest obstacle's direction alone, with no check on
  whether that offset side is itself clear. `cattle_crossing`'s
  `m5_control_log` trace nailed it: ego yaw drifted steadily from 179.2° to
  172.7° over consecutive AVOID ticks — steering TOWARD
  `static.prop.streetsign01`, not away from the original obstacle — then
  EMERGENCY_BRAKE↔AVOID cycling repeated the identical bad swerve forever
  once stuck (206 collision-sensor contacts, 0/5 cattle-crossing events
  ever reached). Fixed by `_pick_safe_avoid_offset`: validates the
  candidate offset against the current costmap before committing, tries
  the mirrored side if blocked, and falls back to no offset (relying on
  EMERGENCY_BRAKE/YIELD, not more steering) if both sides are blocked
  (`tests/test_planner.py`'s `test_pick_safe_avoid_offset_flips_to_the_
  clear_mirrored_side` / `..._falls_back_to_zero_when_both_sides_blocked`).

## Known limitations

- **Uncertainty (`sigma_m`) is parsed but not yet used** to inflate the
  costmap risk radius — see open item B above.
- **No map/lane/drivable-space input** (SPEC.md section 4 — M1 → M5 for
  that does not exist yet). The anisotropic fix above (see "Bugs found")
  makes the risk check direction-aware, which resolved the specific
  "braking for roadside clutter" symptom, but M5 still has no actual
  concept of lane boundaries: AVOID's lateral offset is a fixed 2.5m toward
  the far side of the nearest obstacle, not a real lane-aware maneuver, and
  the "future" conflict check assumes the ego continues at constant
  velocity/heading, not along whatever path M5 last planned. A real map
  source would still be a strictly better fix than direction-based
  heuristics.
- **TTC's known blind spot (item E)** is mitigated, not eliminated: the
  future-conflict check only looks at the ego's own straight-line
  extrapolation, so a conflict that only appears once the ego itself
  changes heading (e.g. mid-AVOID-maneuver) isn't checked against.
- **M1's control executor (`m5_control.py`) has no equivalent CARLA-based
  test** in this pass — `carla.VehicleControl`/`carla.Actor` aren't
  meaningfully mockable without a running CARLA server, so it's covered by
  code review and the module docstring's reasoning, not a unit test the way
  every other M5 module is. A live CARLA smoke run is the real test for it.
