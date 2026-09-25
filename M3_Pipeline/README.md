# M3 Pipeline — LiDAR + Radar Fusion & Object Tracking

M3's job: turn M2's camera detections plus raw CARLA LiDAR/radar into a
persistent, tracked "world model" for M4 (motion prediction) and M5 (path
planning). M3 does **not** do detection (M2), prediction (M4), or planning
(M5) — see the strict boundaries at the bottom of this file.

## Architecture

```
CARLA (M1, scenario3.py)
   |  camera + lidar + radar + ego pose, per tick
   v
M2 perception_server.py  (port 8000)
   |  runs YOLO on the camera frame
   |  forwards {camera_detections, lidar raw bytes, radar raw bytes, ego pose}
   v
M3 m3_server.py  (port 9000)  <-- THIS PIPELINE
   |  lidar_processing.py   : decode -> ego frame -> ROI -> ground removal -> cluster
   |  radar_processing.py   : decode -> spherical->cartesian -> ego frame
   |  camera_lidar_fusion.py: projection + IoU association, ground-plane back-projection fallback
   |  camera_radar_fusion.py: direct camera<->radar fallback association
   |  sensor_fusion.py      : orchestrates the above into fused candidates
   |  tracker.py / kalman_filter.py: per-track constant-velocity KF, Mahalanobis-gated
   |    Hungarian assoc. + static-object cascade, track lifecycle
   |  forwards {tracked_objects} (world frame) to M4, fire-and-forget
   v
M4 (motion prediction) -- m4_dummy_server.py stands in until M4 exists
```

Every request is processed synchronously end-to-end from M1's side (M1 -> M2
-> M3 -> back), so there is at most one frame in flight through M3 at a time;
no extra concurrency handling was needed beyond a defensive `asyncio.Lock`.
The one exception is M3's own forward to M4, which is fire-and-forget (see
"Live CARLA test findings & fixes" below) so it never adds to that chain's
latency.

## Why everything here is Python, not MATLAB/Simulink

CARLA's Python client is the only place LiDAR/radar raw data and camera
detections actually arrive in this pipeline, and the pipeline must run in
lock-step with a live, synchronous-mode CARLA tick. Round-tripping every
frame's point cloud through a MATLAB engine bridge would add serialization
and process-boundary latency for no benefit — none of M3's steps (RANSAC
ground removal, clustering, projection, Kalman filtering, Hungarian
assignment) need anything MATLAB/Simulink offers that NumPy/SciPy don't
already provide. **Recommendation for the team:** keep the whole
perception -> fusion -> tracking chain (M1, M2, M3) in Python; reserve
MATLAB/Simulink for M5/M6 if Stateflow's authoring tools genuinely help
behavior-decision logic — M3's JSON/log output (below) is designed to be
consumed by anything, including MATLAB via its own JSON/HTTP support, so
that boundary is a live option, not a Python-only dead end.

## Coordinate frames

- **World frame**: CARLA's global left-handed, Z-up frame. Tracks are
  maintained here internally (see `tracker.py` docstring for why — tracking
  in the ego's own frame would inject fictitious acceleration into the
  constant-velocity model every time the ego brakes or turns).
- **Ego frame**: X-forward/Y-right/Z-up, origin at the ego actor, assumed to
  sit at ground level (`config.GROUND_PLANE_EGO_Z`). LiDAR/radar processing
  happens here; fused objects are converted to world frame only at the
  tracker boundary.
- All sensor mount transforms (`config.CAMERA_TRANSFORM` /
  `LIDAR_TRANSFORM` / `RADAR_TRANSFORM`) are copied from scenario3.py's
  constants of the same name. **If M1 changes sensor mounting, update
  config.py to match**, or 3D positions will be silently wrong — there is
  no calibration/extrinsics channel in the FramePacket to make this
  automatic.

## Running

```powershell
# one-off setup
cd M3_Pipeline
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt

# each run (from inside M3_Pipeline, with the venv active)
python m4_dummy_server.py      # terminal 1 -- port 9500, stands in for M4
python m3_server.py            # terminal 2 -- port 9000

# then start M2 (from M2_Pipeline, its own venv) and finally CARLA + scenario3.py (M1)
```

`logs/m3_tracks.jsonl` accumulates one JSON record per processed frame:
`{frame_id, timestamp, ego_position, ego_yaw_deg, num_camera_detections,
num_fused_candidates, num_tracks_total, tracked_objects}`.

### Offline tools

- `python tests/test_pipeline_dummy.py` — sanity-checks fusion + tracking
  with synthetic data, no CARLA/M2 required.
- `python visualize_tracks.py [logs/m3_tracks.jsonl]` — bird's-eye-view
  animation of tracked objects for a run.
- `python evaluation.py logs/scenario_stdout.log logs/m3_tracks.jsonl` —
  position-error / match-rate metrics against CARLA ground truth (ground
  truth is only ever used here, offline — never fed into the live pipeline).

## M3 -> M4 interface

Each `tracked_objects` entry:

```json
{
  "track_id": 17,
  "class": "pedestrian",
  "position": [x, y, z],
  "position_ego_relative": [x, y, z],
  "velocity": [vx, vy, vz],
  "speed_mps": 1.3,
  "heading_deg": 87.4,
  "size": [l, w, h],
  "confidence": 0.87,
  "age": 42,
  "hits": 39,
  "time_since_update": 0.05,
  "last_seen": 12.40,
  "sensor_sources": ["camera", "lidar", "radar"],
  "timestamp": 12.40,
  "track_status": "confirmed",
  "is_static": false,
  "last_match_pass": "primary",
  "last_match_cost": 1.42
}
```

`last_match_pass`/`last_match_cost` are diagnostic-only (added to attribute a
residual rare velocity-jump case to a specific association pass -- see "Run
#4 (planned diagnostic)" below) and not part of the frozen contract; they may
be dropped once that investigation is closed out.

`position`/`velocity` are in the CARLA **world** frame (stable across ego
maneuvers, matches CARLA ground truth 1:1 for evaluation);
`position_ego_relative` is provided as a convenience for anything that wants
it directly relative to the ego without re-deriving the transform. Only
`track_status == "confirmed"` tracks are forwarded (>= `TRACK_CONFIRM_HITS`
consecutive hits) — tentative tracks are kept internally to absorb one-frame
noise but are not exposed downstream. `is_static` (re-evaluated every hit —
see `track_types.Track.refresh_static_flag`) flags a track that has barely
moved since it was first seen; M5 can reasonably treat these differently
from moving agents (e.g. a curb, a genuine parked vehicle, a pothole prop)
without needing to infer it from a near-zero velocity itself.

## Live CARLA test findings & fixes

### Run #1

A full live run (CARLA Town05 -> M2 YOLO -> M3 -> M4-dummy, 265 frames)
validated the core math — zero exceptions, and tracked positions/velocities
matched CARLA ground truth (e.g. one long-lived track settled at ~6.87 m/s
against a real ~6.5-7 m/s same-lane vehicle) — but surfaced two real
association-layer bugs, since fixed:

1. **ID churn on static clutter**: 790 track_ids were minted over 265
   frames (92% stuck `"unknown"`), growing ~linearly instead of plateauing.
   Cause: LiDAR clustering has no temporal memory, so a large/extended
   static object's point pattern shifts frame to frame and repeatedly failed
   the flat 4m Euclidean re-association gate that used to exist, spawning a
   new track_id each time instead of reusing the existing one. **Fix**:
   `tracker.py` now runs a second, more permissive association pass
   specifically for tracks flagged `is_static` against unmatched
   `"unknown"`-class detections (`config.STATIC_CASCADE_DIST_FACTOR *
   TRACK_GATING_MAX_DIST_M` radius) — see `tests/test_pipeline_dummy.py`'s
   `test_static_track_absorbs_clustering_jitter_via_cascade`. Also bumped
   `config.LIDAR_CLUSTER_MIN_POINTS` (4 -> 7) to cut single/few-point noise
   blips. **Not applied**: narrowing the LiDAR ROI, even though it's
   plausibly part of the clutter source — the scenario that surfaced this
   is the unsignalized intersection, where legitimate cross-traffic can sit
   at large lateral offsets before a turn; narrow it further only after a
   live re-test confirms the cascade fix alone isn't enough.
2. **Radar velocity sawtooth**: on frames where a radar target
   (re)associated to a track, mean `|Δspeed|` was 1.27 m/s (max 6.16 m/s)
   vs. a 0.28 m/s baseline — because the old `apply_radar_los_correction`
   applied a fixed 50% blend every time it fired, and radar association was
   re-derived from scratch every frame with no persistence (79% of
   radar-touched tracks flickered attached/detached). **Fix**:
   `KalmanFilter6D.update_radar_los` now does a genuine scalar Kalman
   measurement update on the line-of-sight velocity component (weighted by
   the filter's own uncertainty, so it naturally shrinks as a track
   converges instead of always yanking by a fixed fraction), and
   `Track.radar_streak` requires `config.RADAR_MIN_STREAK` (2) consecutive
   radar hits on the *same* track before it's trusted at all — see
   `test_single_frame_radar_blip_does_not_move_velocity`.
3. **A related mis-association** (one track's speed jumped 1.38 -> 11.75 m/s
   in a single step) was traced to the same flat 4m Euclidean gate having no
   concept of a track's velocity/uncertainty. **Fix**: the primary
   association pass now gates on Mahalanobis distance using each track's own
   Kalman covariance (`config.TRACK_GATING_CHI2`), plus an absolute-distance
   safety cap (`config.TRACK_GATING_MAX_DIST_M`) so a very uncertain
   coasting track still can't match something absurdly far away.
4. **Throughput** (M2 only forwarded 265/800 CARLA ticks — effective ~6-7 Hz
   vs. CARLA's 20 Hz) is mostly a YOLO-inference/M1-queue issue outside M3's
   scope, but M3 was contributing one avoidable round-trip: `m3_server.py`
   used to `await` the POST to M4 before replying to M2. It's now
   fire-and-forget (`asyncio.create_task`, outcome logged not returned),
   removing that round-trip from the M1->M2->M3 critical path.

These were fixed at the code level in this session; CARLA isn't available in
the environment that made these fixes, so they were validated only against
the Run #1 numbers and synthetic tests until Run #2.

### Run #2

A 20s re-run (153 frames) with the Run #1 fixes in place confirmed both
mechanisms measurably helped, but neither fully solved the underlying
problem -- each fix addressed the *symptom* the numbers pointed at, not the
deeper cause. Also found and fixed: a real production bug in the Run #1
code itself.

- **Bug**: `Track.is_static` was computed as `python_bool and numpy_bool and
  numpy_bool`, which short-circuits to a raw `numpy.bool_` -- not
  JSON-serializable. This silently 500'd the tracks-log write for ~150
  consecutive frames and, since M4 forwarding is fire-and-forget, silently
  broke 100% of M3->M4 delivery with no visible error on M2's side. **Fix**:
  wrapped in `bool(...)` at the source (`track_types.Track.refresh_static_flag`),
  plus a defensive `_json_default` handler in `m3_server.py` as a second
  line of defense against the same class of bug from any future numpy leak.
- **ID churn, still not solved**: 456 track_ids over 153 frames (~3.3
  new/frame vs. Run #1's ~3.8/frame) -- a modest improvement, still growing
  ~linearly. Root cause: the Run #1 cascade was gated on `Track.is_static`,
  which requires `TRACK_CONFIRM_HITS` (3) hits already accumulated -- a
  chicken-and-egg problem where a track had to survive Pass 1 long enough to
  *prove* itself static before it could get help surviving Pass 1. Only
  0.7% of tracks ever reached that bar (median track lifespan 13.5 frames).
  Digging further: the *actual* numerical cause was `POSITION_MEASUREMENT_STD`
  (0.4, i.e. 16cm std) modeling a LiDAR cluster centroid as far more precise
  than it really is for a large/extended static object -- this collapsed
  each track's position covariance to near-R within 1-2 hits, so Pass 1's
  own Mahalanobis gate rejected the *second* observation of the very same
  clutter before the track could ever reach hit #3. **Fix**: (a) the cascade
  is now gated on `class_name == "unknown"` instead of `is_static`, removing
  the chicken-and-egg dependency (cascade radius factor also lowered 2.0 ->
  1.5, since "proven static, cannot have moved" is a weaker safety argument
  once eligibility isn't restricted to proven-static tracks); (b)
  `POSITION_MEASUREMENT_STD` raised 0.4 -> 0.8 to better match real
  clustering noise, so covariance stays wide enough for genuine
  re-observations to keep re-associating in Pass 1 in the first place.
- **Radar sawtooth reduced, but a worse tail case surfaced**: baseline
  smoothness held (median `|Δspeed|` 0.0 m/s). The Kalman-weighted update
  reduced typical-case radar jumps, but track_id 59 jumped 0.15 -> 40.27 m/s
  (144 km/h) in one step, exactly at the frame `RADAR_MIN_STREAK` first
  passed. Root cause: `sensor_fusion._nearest_radar()` proposes a radar
  candidate by proximity alone (nearest within `RADAR_ASSOC_MAX_DIST_M`),
  with no idea which track it's near -- a fast unrelated target (a racer
  bike) can be the "nearest" return to a slow/static, unrelated cluster. The
  Kalman weighting correctly trusts a noisy-but-*right* measurement more
  smoothly; it does nothing to protect against a confidently *wrong* one,
  and the streak gate alone just delayed the bad correction by 2 frames
  before applying it in a single larger jump instead of a small one. **Fix**:
  `KalmanFilter6D.radar_los_innovation` computes the candidate reading's
  Mahalanobis-style consistency against the *track's own* current velocity
  estimate and uncertainty; `tracker._apply_measurement` now gates on this
  (`config.RADAR_LOS_GATE_CHI2`) before a reading counts toward the streak
  at all -- a reading wildly inconsistent with a converged track is rejected
  outright and never starts a streak, while a still-uncertain track (young,
  or genuinely accelerating) keeps a naturally wider gate. See
  `test_radar_consistency_gate_rejects_wildly_inconsistent_reading`, which
  reproduces the exact track-59 scenario.

As with Run #1, these are code-level fixes made without live CARLA access in
this environment -- **re-validate live** again: track_id growth rate should
drop further (not just from ~3.8 to ~3.3/frame), `is_static` should fire on
a meaningfully larger fraction of long-lived unknown tracks, and no track
should show a single-step speed jump anywhere near the track-59 magnitude
regardless of how long a fast target's radar return stays nearest to it.

### Run #3

Both Run #2 fixes measurably worked as intended: track_id growth dropped
~62% (3.28 -> 1.25 new IDs/frame; 456 -> 184 distinct IDs over the same 161
frames; median track lifespan ~2x, 13.5 -> 26.0 frames) and the radar
sawtooth's worst case dropped ~77% (max Δspeed 40.12 -> 9.32 m/s). But
broadening Pass 2 (the cascade) to any `"unknown"`-class track -- the exact
change that fixed fragmentation -- reopened a hijack path the Mahalanobis
gate was built to close, just through the cascade's back door, since the
cascade itself never had any velocity/uncertainty check at all.

- **Cascade hijack**: a pure-LiDAR `"unknown"` track (no radar involved)
  jumped 0.29 -> 14.82 -> 29.8 m/s (108 km/h) across 3 frames. Pass 2 only
  gates on flat Euclidean distance (`STATIC_CASCADE_DIST_FACTOR *
  TRACK_GATING_MAX_DIST_M` = 12m) and then feeds a match into the same
  full-trust Kalman position update Pass 1 uses -- so a fast, unrelated
  cluster that happened to land within that radius of an old slow/static
  track got fully believed in one step, corrupting its velocity state.
  Radar on/off flicker also got worse (50.8% -> 82.8% of radar-touched
  tracks), though that's mostly downstream of more tracks now surviving
  longer (more track-lifetime for radar to flicker across), not a new bug.
- **Why not a hard distance/dt speed cutoff** (the natural first idea, and
  what was requested): genuine clustering jitter for a large/extended
  static object CAN itself be several meters within a single ~0.1-0.3s
  tick -- that's the entire reason the cascade radius is 12m in the first
  place. A fixed "implied speed" threshold can't tell "big one-off jitter,
  still the same static thing" apart from "a different fast object happened
  to land nearby" using distance/dt alone -- both produce an identical
  single-frame signature. See config.py's "Run #3 finding" comment for the
  full reasoning.
- **Fix**: `KalmanFilter6D.update()` now takes an optional measurement-noise
  override; `tracker._apply_measurement` passes a much larger one
  (`config.CASCADE_POSITION_MEASUREMENT_STD`, 3.0 vs. the normal 0.8) for
  cascade-origin matches specifically. A cascade match already failed the
  statistical Pass-1 test, so it's a genuinely less certain measurement, and
  the Kalman gain now reflects that: a wrong one-off match nudges position
  only slightly and barely touches velocity (which moves through the small
  position-velocity cross-covariance, not directly), while a real recurring
  static object's cluster still slowly converges over repeated
  re-associations -- degrading gracefully instead of a brittle cutoff.
  Verified directly: replaying the track-104-shaped scenario (a confirmed
  slow/static `"unknown"` track, then one fast cluster landing 8m away)
  produces an 11.65 m/s jump with the old full-trust update vs. 1.82 m/s
  with this fix -- an ~84% reduction, in the same range as the ~77%
  reduction the equivalent radar fix achieved in Run #2. See
  `test_cascade_match_does_not_hijack_velocity`.
- **Not fully solved**: a *sustained* run of consistent wrong cascade
  matches (the same fast object recurring near the same track for several
  frames in a row) still gradually pulls the track's velocity up, just far
  more slowly than before (reproduced test sequence: 0.32 -> 1.82 -> 3.53 ->
  5.56 m/s over 3 consecutive wrong hits, vs. the old code's single-hit jump
  to ~11-15 m/s). This is arguably correct Kalman behavior -- enough
  consistent evidence should eventually update a belief -- but it means the
  cascade's hijack risk is reduced, not eliminated, especially in denser
  scenes (dense-market) with more nearby-but-different unknown objects than
  this junction scenario had. If a future live run still shows this,
  consider adding a radar-style consistency streak requirement to Pass 2
  itself (require N consecutive cascade matches trending toward the same
  implied state before trusting them at full weight).

### Run #4

Live re-test confirmed the Run #3 cascade fix works as designed, with no
regression on the fragmentation fix: worst single-frame jump 14.99 -> 9.14
m/s (-39%), jumps >5 m/s down 72% (0.50% -> 0.14% of all deltas), mean/median
Δspeed down 36%/39%, radar-frame Δspeed mean/max down 46%/68%. Fragmentation
metrics stayed in the same much-improved band as Run #3 (track_id growth
1.25 -> 1.43/frame, still ~4x better than the pre-fix 3.28/frame), and
`is_static` fired more often (1.6% -> 5.2% of tracks) -- more tracks now
correctly settle into "proven static" instead of getting dragged off it by a
wrong one-off cascade match, itself evidence the softer update is working.

One tail case remains: a pure-LiDAR `"unknown"` track still jumped 3.43 ->
12.57 m/s (9.14 m/s single-step change) in one 0.1s tick. Since
`CASCADE_POSITION_MEASUREMENT_STD` only softens Pass-2 (cascade) matches,
this residual case is consistent with either (a) the softening not being
enough in this instance, or (b) the same wrong object slipping through Pass
1 itself (a full-trust Mahalanobis match, if that track's own covariance
happened to be wide enough at that moment) -- indistinguishable from the
numbers alone.

**Planned diagnostic, no algorithm change yet**: rather than guess at a
fourth fix, `Track.last_match_pass` ("spawn" | "primary" | "cascade") and
`Track.last_match_cost` (squared Mahalanobis distance for a primary match,
meters for a cascade match) are now recorded on every association in
`tracker._apply_measurement` and surfaced on every `tracked_objects` entry
(diagnostic-only, see the interface section above). The next live run should
filter `logs/m3_tracks.jsonl` for any track_id whose `speed_mps` jumps a lot
between two consecutive logged frames, then read that later frame's
`last_match_pass`/`last_match_cost` to attribute the jump to a specific pass
before deciding whether -- and how -- to fix it further (e.g. Pass 1 leaking
would point at `TRACK_GATING_CHI2`/`TRACK_GATING_MAX_DIST_M` being too loose
for this case; a persistent Pass-2 cascade cause would point at the
consistency-streak idea noted under Run #3's "Not fully solved").

## Known assumptions / limitations (read before trusting the numbers)

- **Radar velocity sign** (`config.RADAR_VELOCITY_SIGN`): CARLA's docs say
  RadarDetection.velocity is "towards the sensor" (positive = approaching),
  which is what this pipeline assumes, but it has not been empirically
  verified against a live closing/opening scenario. If M4's risk estimates
  look inverted, flip this constant.
- **Ground level assumption**: the ego actor's own transform is assumed to
  be at ground level. True for standard CARLA vehicle blueprints; if M1
  switches ego vehicle type, sanity-check `config.GROUND_PLANE_EGO_Z`.
  Ground removal falls back to a flat-plane RANSAC-failure heuristic using
  this same assumption.
- **Clustering is Euclidean-radius, not learned**: tightly packed objects
  (e.g. a pedestrian group in the dense-market scenario) can merge into one
  cluster. Tune `config.LIDAR_CLUSTER_EPS_M` if this shows up in testing.
- **Camera-only detections** (no LiDAR/radar support) get a rough
  ground-plane-backprojected position and a confidence penalty — treat
  these as lower-quality fixes, not LiDAR-grade positions.

## Strict responsibility boundaries

M3 owns: LiDAR/radar acquisition+processing, coordinate transforms,
camera/LiDAR/radar association, sensor fusion, object tracking (position,
velocity, heading, confidence, track lifecycle).

M3 does **not** do: camera object detection (M2), trajectory prediction or
risk estimation (M4), behavior/path planning (M5), vehicle control (M6), or
CARLA scenario design (M1).
