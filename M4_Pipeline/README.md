# M4 Pipeline — Motion Prediction + Uncertainty Estimation

M4's job: turn M3's tracked objects into short-term, multi-modal motion
predictions with uncertainty/confidence, and forward that to M5. M4
deliberately does **not** compute TTC, time-to-conflict, collision
probability, or a risk score/level — that was explicitly cut from scope
(see "Status" below); M5 (or a later stage) derives risk itself from M4's
raw predicted trajectories if/when needed. M4 does **not** do tracking (M3),
behavior/path planning (M5), or vehicle control (M1's control executor).

## Architecture

```
M3 (m3_server.py)
   |  POST /api/v1/tracks  (FramePacket, port 9500)
   v
M4 m4_server.py  (port 9500)  <-- THIS PIPELINE
   |  recorder.py            : records the raw packet verbatim (replay source)
   |  tracking_input/        : adapter (M3 wire format -> internal shapes),
   |                           packet ordering, input-quality scoring,
   |                           bounded per-track history
   |  predictor.py           : orchestrates one frame through:
   |    models/              : class-specific constant-velocity / constant-
   |                           acceleration nominal trajectory
   |    uncertainty/         : growing along/lateral std-dev per predicted step
   |    multimodal/          : nominal + stop + lateral candidate trajectories
   |  output/                : M4 -> M5 wire schema + fire-and-forget sender
   v
M5 (does not exist yet) -- m5_dummy_server.py stands in, port 10000
```

Every request is processed synchronously end-to-end within M4's own lock
(mirrors M3_Pipeline/m3_server.py's own `runtime.lock` pattern), and the
forward to M5 is a separate fire-and-forget `asyncio.create_task` — M4 never
blocks on M5, the same way M3 never blocks on M4.

## Verified against M3 (do not re-derive from the mentor-notes doc — trust this)

Read directly from the current `M3_Pipeline` code (`m3_server.py`,
`tracker.py`, `track_types.py`, `config.py`) before writing anything here:

- M3's actual working environment in this repo is **Python 3.11.9**, not
  3.14 — the venv's `pyvenv.cfg` pointed at a Python 3.14 install from a
  different machine (`C:\Users\Raman\...`) that doesn't exist here; it had
  to be rebuilt with 3.11 to run at all. **M4 targets Python 3.11** to match.
- M3 has now been live-tested across all five required scenarios in this
  repo (village road/Town07, urban intersection via scenario3.py/Town05,
  highway merge/Town06, dense market/Town02, cattle crossing/Town05), not
  only the intersection scenario — `M1_Pipeline/common.py`'s sensor
  transforms are confirmed identical to `M3_Pipeline/config.py`'s.
- The exact M3→M4 wire format (`tracking_input/schema.py`) was copied
  field-for-field from `m3_server.py`'s `_to_track_dict` and the
  `_forward_to_m4` payload, including the diagnostic `last_match_pass`/
  `last_match_cost` fields M3 added while chasing its own velocity-jump bug.
- `track_status` is always `"confirmed"` on anything M3 sends (only
  `tracker.confirmed_tracks()` is forwarded); `hits` counts TOTAL successful
  matches, not a consecutive streak (confirmed by reading `tracker.py`:
  misses never reset `hits`).
- `is_static` is still reported per-track but is **no longer used by M3's
  own tracker** as its association gate (an earlier version gated the
  clutter-cascade on it; M3's current code gates that cascade on
  `class_name == "unknown"` instead, per its own in-code "Run #3 finding"
  comment) — M4 can still use `is_static` as a shortcut signal, this is
  just no longer telling you anything about M3's internal association logic.

## Running

```powershell
# one-off setup
cd M4_Pipeline
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt

# each run (from inside M4_Pipeline, with the venv active)
python m5_dummy_server.py      # terminal 1 -- port 10000, stands in for M5
python m4_server.py            # terminal 2 -- port 9500

# then start M3, M2, and finally CARLA + a scenario script (M1) — same
# order as M3_Pipeline/README.md's "Running" section.
```

`logs/m4_input_packets.jsonl` accumulates the RAW JSON M3 sends, one line per
frame, before any parsing — the replay source (`replay.py`), independent of
M3's own `m3_tracks.jsonl` (which omits `ego_velocity`).
`logs/m4_predictions.jsonl` accumulates M4's own output, one line per frame.

### Offline tools (no CARLA/M3/M2 needed)

- `python replay.py [logs/m4_input_packets.jsonl] [--out logs/m4_predictions.jsonl]`
  — replays recorded packets through the exact same `Predictor` core the
  live server uses.
- `python visualization/visualize_predictions.py [logs/m4_predictions.jsonl]`
  — bird's-eye animation: predicted trajectories (solid=nominal,
  dotted=stop, dashed=lateral), uncertainty ellipses, class-colored markers
  annotated with confidence.
- `python -m pytest tests/` — pure-math unit tests only (CV/CA kinematics,
  packet ordering). **Not a substitute for real-data testing** — see the
  project brief's explicit "no dummy data" rule; these tests exist only to
  pin down formulas on hand-computed cases.
- `python -c "from evaluation.ground_truth import parse_ground_truth; from evaluation.metrics import ...; ..."`
  — ADE/FDE/miss-rate against a recorded `[GROUND TRUTH]` stdout log (see
  `evaluation/metrics.py`).

## M4 → M3 interface (input, verified — see `tracking_input/schema.py`)

Every processed frame, M3 POSTs one `FramePacket` (see `output/schema.py`'s
neighbor for the mirrored M4→M5 shape):

```json
{
  "frame_id": 1234,
  "timestamp": 12.40,
  "ego_position": [x, y, z],
  "ego_velocity": [vx, vy, vz],
  "ego_yaw_deg": 90.0,
  "tracked_objects": [
    {
      "track_id": 17, "class": "pedestrian",
      "position": [x, y, z], "position_ego_relative": [x, y, z],
      "velocity": [vx, vy, vz], "speed_mps": 1.3, "heading_deg": 87.4,
      "size": [l, w, h], "confidence": 0.87, "age": 42, "hits": 39,
      "time_since_update": 0.0, "last_seen": 12.40,
      "sensor_sources": ["camera", "lidar", "radar"], "timestamp": 12.40,
      "track_status": "confirmed", "is_static": false,
      "last_match_pass": "primary", "last_match_cost": 1.42
    }
  ]
}
```

No history, no acceleration, no covariance, no oriented box, no map/lane
info. `tracking_input/history.py` builds M4's own bounded per-track history;
`models/constant_acceleration.py` derives acceleration from it.

## M4 → M5 interface (output — M5_Pipeline now exists; this is FROZEN to its design)

**Superseded**: this used to be a proposed HTTP contract on port 10000 with
no real M5 to confirm it against. M5_Pipeline's own design (see
`M5_Pipeline/SPEC.md`) fixed the actual contract as **UDP port 5004**
instead — `output/m5_udp_schema.py` builds it, `output/m5_udp_sender.py`
sends it (a single non-blocking `sendto()`, no background task, unlike
M2→M3/M3→M4's HTTP fire-and-forget). `output/schema.py`'s `PredictionPacket`
is still used internally (recording/replay/evaluation — Phase 5 depends on
it), it's just no longer what goes out over the wire to M5.

World frame throughout, with `ego_position`/`ego_yaw_deg` included so M5
can convert to ego-relative itself. Two top-level lists: `obstacles`
(dynamic tracks — current state + up to 3 candidate `trajectories`, each
with points, a per-point combined `sigma_m` radius, and a probability) and
`statics` (`is_static` tracks — position/size only, no velocity or
trajectories: M5_Pipeline's own README section "Open items, resolved (A)"
covers why they're split out). M5 never sees which motion model or how
many modes produced any of it (section 41). No TTC/collision-probability/
risk-score/risk-level — M4 does not compute risk (see "Status" below); M5
owns all of that (see `M5_Pipeline/README.md`).

## Design choices and why (short version — see each module's own docstring for the full reasoning)

| Question | Choice | Why |
|---|---|---|
| Primary motion model | Constant velocity | Consistent with M3's own constant-velocity Kalman filter; CA's acceleration estimate is inherently noisier (2nd derivative of already-noisy positions) |
| Uncertainty representation | Growing along/lateral std-dev, oriented by heading | The one thing that actually matters (uncertainty grows with horizon, and unevenly between along-track vs. cross-track) without needing a full covariance/particle-cloud machinery M5 doesn't need |
| Multimodal | 3 rule-based modes: nominal/stop/lateral | Section 52: no learned mixture model: the simplest branching that still covers "continues", "stops", "turns/crosses" |
| Risk (TTC/collision-probability/risk-score) | Not implemented in M4 | Explicit scope cut — see "Status" below; M5 derives risk itself from M4's raw trajectories if/when needed |
| Config location | Flat `config.py`, not a `config/` package | Matches M2/M3's existing flat-config style |
| MATLAB/Simulink | Not used anywhere in this pass | The whole pipeline (M1-M3) is already Python/FastAPI; nothing in M4's math needs Simulink, and a cross-language bridge would only add latency to the M3→M4→M5 critical path for no benefit — see `training/__init__.py`'s docstring for the one place (an optional future LSTM) where this could be revisited |

## Status (roadmap phases, see the project brief's section 46)

Done: Phase 1 (server + input interface), Phase 2 (recorder + replay),
Phase 3 (CV baseline), Phase 4 (CA model), Phase 6 (class-specific), Phase 7
(uncertainty), Phase 8 (multimodal), Phase 13 (M4→M5 API — proposed, not yet
frozen: no M5 exists to confirm it against).

**Phase 5 (baseline-vs-ground-truth) has now been run** against a full live
5-stage capture (`scenario3_fuzzy_final7.log` / `logs/m4_predictions.jsonl`,
178 frames, ~28 simultaneous actors, dense traffic). `evaluation/run_eval.py`
is the runner (`python -m evaluation.run_eval [predictions.jsonl]
[ground_truth.log]`). Three real bugs were found and fixed while getting a
trustworthy number out of it, all in the evaluation harness, not the model:

1. **Off-by-one time indexing**: `points[0]` in a trajectory is `t=0` (the
   current position, see `models/trajectory.py`'s `sample_times`), not one
   `time_step_s` into the future — the first eval runner scored every point
   against the wrong ground-truth frame.
2. **Duplicate identity claims in dense frames**: matching each of an
   object's 3 trajectory-modes independently (`match_actor`) lets two
   different M4 track_ids claim the same nearby ground-truth actor — this
   scenario averages ~2 OTHER actors within the 5m match threshold of any
   given one, confirmed happening in practice (two different track_ids
   matched to the same actor_id in the same frame). Fixed by matching once
   per object per frame with a unique greedy nearest-pair assignment
   (`evaluation/metrics.py`'s `match_actors_unique`).
3. **Per-mode-only scoring overstates failure**: M4 is a multimodal
   predictor by design (nominal/stop/lateral) — a single mode "missing" is
   expected (that's why the others exist). Scoring each mode independently
   roughly DOUBLES the apparent error versus the standard minADE/minFDE
   ("best of the k candidate modes per object", the metric the trajectory-
   prediction literature, e.g. nuScenes' prediction challenge, actually
   uses for multimodal output) — `run_eval.py` now reports both, labeled
   `BEST-OF-3` vs. per-mode.

After those three fixes, plus replacing `multimodal/features.py`'s
mean-of-raw-velocity-readings smoothing with a least-squares fit of
position-vs-time (M3's own per-frame velocity reading can OSCILLATE in sign
frame-to-frame for a genuinely, steadily-moving object, which a mean cancels
toward ~0 — see that module's docstring), the corrected result on this run:
BEST-OF-3 ADE ≈ 4.0-4.5m, FDE ≈ 7.7-8.9m, miss rate (FDE > 2.0m) ≈ 85%,
regardless of input_quality bucket. The miss rate is still high, but a fixed
2m threshold over a car's 4-second horizon (15-30m of real displacement) is
a strict bar for any non-learned CV/CA baseline — comparable published
constant-velocity baselines on similar multimodal-prediction benchmarks
report similarly high miss rates, so this is plausibly close to what a
rule-based model of this kind should score here, not obviously a remaining
bug. Bringing it down further would most likely mean either a
horizon/speed-scaled miss threshold (2m is arguably too strict a constant
across pedestrian vs. truck horizons) or a genuine modeling upgrade (learned
motion model) rather than another evaluation fix — see `training/__init__.py`
for why that's deliberately deferred.

**Phases 9-12 (TTC, time-to-conflict, collision probability, risk score) are
explicitly OUT OF SCOPE for this pass**, per the project owner's decision to
keep M4 focused on prediction/uncertainty and defer risk computation
entirely (to a later stage, or to M5) "unless explicitly required later" —
this was a deliberate cut, not unfinished work; an earlier version of this
pipeline did implement all four (a `risk/` package with `ttc.py`,
`conflict.py`, `collision_probability.py`, `risk_score.py`) and was removed
wholesale. If risk estimation is required later, that prior design (TTC via
swept-circle trajectory intersection, time-to-conflict via cross-time
spatial-overlap search, collision probability as the sum of colliding
multimodal modes' probabilities, a weighted+normalized risk score with a
hard safety override) is a reasonable starting point to reintroduce, ideally
as its own downstream module rather than back inside M4.

Phases 14-21 (dataset generation, optional ML, M5 integration, five-scenario
runs, latency/accuracy measurement, API freeze) are not started — see
`dataset/__init__.py` and `training/__init__.py` for why ML is deliberately
deferred.

## Known limitations (read before trusting the numbers)

- **Lateral-mode turn direction is a conservative heuristic, not a
  forecast** (`multimodal/modes.py`): with no map/lane data, M4 picks the
  lateral branch's turn direction toward the ego unless a real turning
  trend is visible in recent history. This deliberately biases toward
  flagging conflicts, not toward predicting what the object will actually do.
- **No map/lane/drivable-area input.** M4 was designed to not need one
  (section 11), but this also means it cannot tell "off-road" from
  "on-road" when picking a lateral-mode direction or judging plausibility.
- **No risk/danger assessment at all** (by design, see "Status" above) —
  M4's output is purely descriptive (where might this object go, how
  uncertain is that), not evaluative (is that dangerous). Anything
  downstream that needs to know "should I be worried about this" has to
  compute it from the raw trajectories/uncertainty M4 provides.
