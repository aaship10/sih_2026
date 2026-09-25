# M5 (Behavior Decision & Path Planning) — Design Spec

This is the design document M5_Pipeline was built from, kept verbatim for
reference. See `README.md` for what was actually implemented, including
how each of this doc's "Open Items" (section 5) was resolved.

---

This overview details the current architecture, data contracts, and calculation mechanisms of the M5 (Behavior Decision & Path Planning) module. Review these specifications to ensure complete alignment across the perception/prediction subsystems (M3/M4) and M1's control executor.

Pipeline: M1 -> M2 -> M3 -> M4 -> M5 -> back to M1. There is NO M6 stage. M5's output goes directly back into M1's simulation loop, which applies it to the ego vehicle in CARLA.

Ownership: M4 provides predicted trajectories and uncertainty ONLY. M5 owns ALL Time-To-Collision, collision and risk assessment. M4 computes none of it.

## 1. M5 Input / Output Interfaces (The Data Contract)

M5 is the bridge between predicted traffic and vehicle motion. It expects kinematic and predicted-motion data from M4 and outputs deterministic driving commands directly to M1.

Status of the links (verify each in the code before relying on it):
* M3 -> M4: HTTP POST to port 9500 (/api/v1/tracks). This is implemented and verified in M3's code.
* M4 -> M5: UDP port 5004, as M5 currently expects. M4 has NO such output today. It is a NEW M4 output adapter that must be built.
* M5 -> M1: UDP port 5006. M1 has NO listener or control executor for it today. Both are NEW work in M1.

**Input Expected from M4 (UDP Port 5004, per M5's current design; verify in the M5 code):**
M5 expects a synchronized JSON payload containing a timestamp and an array of `obstacles`. For each obstacle, M4 must provide:

* **Current State:** `track_id`, `class`, `confidence`, `pos_x`, `pos_y`, `vel_x`, `vel_y`, `width`, `length`.
* **Predicted State:** `trajectories` array containing `mode`, `probability`, and `points` (x, y, t).
* **Crucial Note for M4:** M4 sends **zero** risk, collision, or Time-To-Collision (TTC) values. M5 handles all safety assessments internally to maintain a deterministic pipeline.

**Output Broadcast to M1 (UDP Port 5006):**
M5 outputs a JSON payload containing the final, collision-checked driving instructions:

* **Control Flags:** `behavior_state` (e.g., "AVOID", "CRUISE"), `emergency_stop` (Boolean override).
* **Target Metrics:** `target_speed` (calculated optimal speed in m/s).
* **Path Data:** `trajectory` array containing exactly 16 spatial-temporal waypoints (`x`, `y`, `v_target`, `yaw`, `t`).
* **Diagnostics:** `replan_triggered` and `computation_time_ms`.

M5 does NOT compute steering, throttle or brake. It only decides behavior, speed and path.

## 1b. What M1 Must Do With M5's Output (replaces the old M6 role; new work)

M1 hosts a small control executor inside its simulation loop. It is not a separate stage or process.

* **Single writer:** the executor is the ONLY place that applies control to the ego vehicle. Any autopilot or scripted ego control in the scenario scripts must be disabled while M5 is driving. Two writers make the vehicle flip between "go" and "stop" every tick.
* **Receiving:** a non-blocking UDP listener on port 5006. Each simulation tick, the loop takes the newest M5 message. M5 runs at 10 Hz and the CARLA tick is faster, so the last command is held between updates.
* **Staleness:** if no fresh command arrives within a timeout (tuned in CARLA), the executor falls back to a controlled brake, never to "keep going".
* **Emergency:** `emergency_stop = true` means full brake and zero throttle immediately, overriding everything else until M5 clears it.
* **Longitudinal control:** track `target_speed` / `v_target` with a speed controller (for example PID) that outputs throttle or brake.
* **Lateral control:** follow the 16 waypoints using `x`, `y`, `yaw` (for example pure pursuit or Stanley), converted to a normalized steering command.
* **After applying the command,** the loop continues the scenario as before.
* **Logging per tick:** command age, applied control and pose, so the evaluation metrics (replanning latency, path smoothness, minimum obstacle distance, collisions, completion rate) can be computed.

## 2. Core Working Mechanism & Calculations (Python Layer)

The Python layer of M5 acts as the spatial mapping and mathematical verification engine, running continuously at 10Hz.

* **Time-to-Collision (TTC):** Calculated strictly based on relative kinematics. Python computes the Euclidean distance ($d$) between the Ego vehicle and the obstacle, and divides it by the relative closing velocity ($v_{rel}$).

$$TTC = \frac{d}{v_{rel}}$$

* **2D Local Costmap & Risk Evaluation:** M5 maps the raw coordinates from M4 onto a localized 2D grid. Using Euclidean Distance Transforms, obstacles are "inflated" into Red (Critical), Yellow (Medium), and Green (Low) risk zones. Risk is evaluated by projecting the Ego vehicle's 3-circle physical footprint onto this grid to check for zone intersections.
* **Trajectory Generation:** If an avoidance maneuver is required, Python utilizes a **Quintic Polynomial Sampler**. A 5th-degree polynomial is used because it satisfies 6 boundary conditions (Start/Target Position, Velocity, and Acceleration), mathematically guaranteeing a smooth, jerk-limited path that preserves vehicle stability.

## 3. The Decision Engine (MATLAB / Simulink Layer)

M5 utilizes a "Split-Brain" architecture. While Python handles the spatial math, the actual driving decisions are outsourced to a MATLAB/Simulink bridge (`matlabengine`).

* **Stateflow Finite State Machine (FSM):** The core logic is housed in a deterministic Stateflow chart (`behavior_fsm.slx`). It evaluates the TTC and Costmap flags provided by Python.
* **State Transitions:** The vehicle moves between states (`CRUISE`, `FOLLOW`, `AVOID`, `YIELD`, `EMERGENCY_BRAKE`) based on strict threshold rules. For example, if $TTC \le 1.5$ seconds, the FSM guarantees an immediate `EMERGENCY_BRAKE` command, which M1 executes directly as `emergency_stop`. Missing transition arrows (hysteresis) deliberately lock the vehicle in safety states to prevent control oscillation (whiplash).
* **MATLAB Functions:** Embedded MATLAB functions within the Stateflow calculate the exact `target_speed` required for the active state before passing the final decision back to Python.

## 4. Hardcoded vs. Dynamic Elements (Action Items)

To prepare for unstructured road conditions and ensure full integration, the team must address the current hardcoded elements within M5.

* **Fully Dynamic (Live):** Ego-vehicle tracking, dynamic obstacle states (from M4), TTC calculations, risk mapping, Stateflow decision-making, and Quintic Polynomial generation.
* **Currently Hardcoded (Needs API Integration):** The `drivable_space` (lane boundaries/road edges) and `statics` (parked cars, walls, permanent infrastructure) are currently pulled from a mock initialization function (`get_scenario_data(4)`).

**Decision:** M4 is NOT responsible for static map geometry. M3 provides no lane or road-edge information, so M4 has no source for it. The static map geometry (drivable space, road edges) comes to M5 directly from M1's CARLA connection. This M1 -> M5 interface does not exist yet and must be defined and built.

## 5. Open Items To Settle Before Integration

A. **Static objects from perception.** Whether M4 forwards static tracks (M3 tracks flagged `is_static`) to M5 as a separate `statics` list, or M5 derives statics some other way, is a separate design choice. It must be decided and documented. Either way it is independent of the map geometry in section 4.

B. **Uncertainty is not in M5's input schema.** The `trajectories` schema has `mode`, `probability` and `points` (x, y, t), but no field for prediction uncertainty. Decide whether M5 uses it (for example a per-point sigma or covariance to inflate obstacles in the costmap) or ignores it. If M5 uses it, add the field to the schema.

C. **Frames and units of M5's output.** State whether `x`, `y` and `yaw` in the trajectory are in the CARLA world frame or the ego frame, and whether yaw is in degrees or radians. M1 needs this to steer correctly. CARLA's frames are left-handed (ego frame: X forward, Y right).

D. **Frames and fields of M4's input.** State whether `pos_x`, `pos_y`, `vel_x`, `vel_y` are expected in the world or ego frame. Also state what `width` and `length` mean: M3 sends axis-aligned extents along the ego axes, not an oriented box.

E. **TTC limitation (M5 now owns TTC alone).** TTC = d / v_rel is only meaningful when the object is actually closing on the ego. For crossing or offset paths it can be large or undefined even though a conflict is predicted. Since M4 no longer provides any time-to-conflict, M5 should check the costmap/footprint against the predicted trajectories, not only this formula, before the FSM decides.
