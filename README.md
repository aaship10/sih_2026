# SIH 2026: Autonomous Driving Pipeline

**Team Name:** Mathletics (Team ID: 131459)  
**Theme:** Smart Vehicles  
**PS Category:** Software  

Welcome to the **SIH 2026** project repository! This project implements a comprehensive, modular Autonomous Driving pipeline designed to operate within the **CARLA Simulator**. 

The system architecture is broken down into six sequential modules (M1 to M6), spanning from raw perception to low-level vehicle control. The pipeline is designed for real-time performance, communicating across nodes using ZeroMQ (ZMQ) and UDP.

## 🏗️ System Architecture

The project is structured into distinct pipelines, representing the standard lifecycle of an autonomous vehicle system:

### 1. **M1 Pipeline: Perception**
Processes raw camera streams using deep learning (YOLOv8/custom models) to detect dynamic and static objects (vehicles, pedestrians, traffic signs, etc.).
- **Key Components:** `Carla Object Detection/`, `perception_server.py`

### 2. **M2 Pipeline: Advanced Perception / Pre-processing**
Works in tandem with M1, extending detection capabilities and preparing raw bounding boxes and sensor metadata for the fusion layer.

### 3. **M3 Pipeline: Sensor Fusion**
Fuses multi-modal sensor data (**Camera, LiDAR, Radar**) to create a unified understanding of the vehicle's environment.
- **Key Components:** `fusion.py`, `kalman_filter.py`, `tracker.py`
- Handles coordinate transformations and persistent tracking of dynamic objects.

### 4. **M4 Pipeline: Prediction**
Predicts the future trajectories and behaviors of tracked dynamic objects.
- **Key Components:** `multimodal.py`, `motion_models.py`, `predictor.py`
- Outputs dynamic predictions that inform safe decision-making.

### 5. **M5 Pipeline: Behavior & Local Planning**
The core decision-making module. It evaluates the environment (drivable space, static obstacles, and predicted dynamic trajectories) to generate a safe, local trajectory.
- **Master Node:** `main_planner_node.py`
- **Key Components:** 
  - **Behavior FSM (`behavior/`):** Finite State Machine for state decisions (Yield, Avoid, Emergency Stop, Cruise).
  - **Costmap (`costmap/`):** Generates local cost grids incorporating obstacle footprints and risks.
  - **Trajectory Generator (`trajectory_generator/`):** Polynomial trajectory sampling and cost-based evaluation.

### 6. **M6 Pipeline: Control & Actuation**
Translates the planned trajectory into actionable low-level vehicle commands (throttle, brake, steering) in the CARLA simulator.
- **Master Node:** `m6_control/main_controller_node.py`
- **Controllers:** Stanley Controller (Lateral Control) and PID Controller (Longitudinal Control).
- Features a Safety Supervisor to trigger emergency stops if data streams time out or constraints are violated.

## 📂 Project Structure

```text
sih_2026/
├── Carla Object Detection/  # YOLO-based object detection models and scripts
├── M1_Pipeline/             # Perception and scenario definitions
├── M2_Pipeline/             # Secondary perception logic
├── M3_Pipeline/             # Sensor Fusion (LiDAR, Camera, Radar)
├── M4_Pipeline/             # Trajectory Prediction & Motion Models
├── M5_Pipeline/             # Standalone planning experiments and specs
├── m6_control/              # Low-level actuation (PID, Stanley)
├── behavior/                # M5 Behavior FSM and Simulink models
├── costmap/                 # M5 Local Costmap generation
├── interfaces/              # Shared data types (EgoState, DrivableSpace) and network adapters
├── trajectory_generator/    # M5 Polynomial trajectory sampling
├── tests/                   # Integration and unit tests
├── main_planner_node.py     # Master executable for the M5 Planning node
└── README.md                # Project documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- [CARLA Simulator](https://carla.org/) (Compatible with 0.9.13/0.9.14)
- ZeroMQ (`zmq`)
- NumPy, PyTorch (for perception)

### Running the Pipeline

The system is designed to run synchronously with CARLA. Typical execution involves spinning up the modules in sequence:

1. **Start CARLA Server:**
   ```bash
   ./CarlaUE4.sh
   ```

2. **Launch Perception & Fusion (M1 - M3):**
   ```bash
   python M1_Pipeline/perception_server.py
   python M3_Pipeline/main.py
   ```

3. **Launch Prediction (M4):**
   ```bash
   python M4_Pipeline/m4_server.py
   ```

4. **Launch M5 Planner (Master Node):**
   *(Expects ZMQ communication on `tcp://127.0.0.1:5555`)*
   ```bash
   python main_planner_node.py
   ```

5. **Launch M6 Controller:**
   *(Connects to CARLA Client and listens to M5)*
   ```bash
   python m6_control/main_controller_node.py
   ```

## 🧪 Testing and Evaluation

- The repository includes mocked scenarios and unit tests for offline development.
- Run `tests/run_simulink_loop.py` or execute `pytest` in respective directories for component-level verification.
- Extensive debugging logs and metrics are emitted during runtime, which are crucial for tuning PID gains and costmap thresholds.

---
*Built for the Smart India Hackathon (SIH) 2026.*
