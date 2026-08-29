# M5: Adaptive Path Planning & Behavior Engine

## Overview
The M5 subsystem acts as the tactical brain of the autonomous vehicle. It takes predicted obstacle data (from M4), calculates Time-to-Collision (TTC), and uses a Stateflow FSM to output behavior states (`CRUISE`, `AVOID`, `EMERGENCY_BRAKE`). It outputs a safe, jerk-limited trajectory evaluated against a 2D Local Costmap.

## Environment Setup (Critical for M6)
To run this module, the host machine **MUST** have MATLAB installed.

1. Install dependencies:
   `pip install numpy matplotlib scipy matlabengine`
2. **Important:** The Python script automatically starts the MATLAB engine in the background and loads `behavior/m5_planner_core.slx`.

## Outputs to M6 (at 10Hz)
M5 provides a JSON payload to M6 containing:
1. `trajectory`: Array of 16 (X, Y) spatial waypoints.
2. `target_speed_ms`: Optimal velocity to maintain.
3. `emergency_brake`: Boolean override to halt the vehicle.