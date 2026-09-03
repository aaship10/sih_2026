"""
Master Node for M5 (Behavior & Local Planning).
Simulates a 10 Hz real-time planning loop with dynamic replanning triggers.
"""

import time
import numpy as np
import zmq

# 1. Put this at the top of main_planner_node.py (before the loop starts)
context = zmq.Context()
pub_socket = context.socket(zmq.PUB)
pub_socket.bind("tcp://127.0.0.1:5555")

from interfaces.data_types import EgoState, DrivableSpace, TrackedObject, DynamicPrediction, PlanningOutput
from interfaces.m4_adapter import M4Adapter
from behavior.behavior_fsm import BehaviorFSM
from costmap.local_costmap import LocalCostMap
from trajectory_generator.polynomial_sampler import PolynomialTrajectorySampler
from trajectory_generator.cost_evaluator import CostEvaluator
from tests.mock_scenarios import get_scenario_data

class M5PlannerNode:
    def __init__(self):
        self.fsm = BehaviorFSM()
        self.costmap_gen = LocalCostMap()
        self.sampler = PolynomialTrajectorySampler(time_horizon=3.0, dt=0.2)
        self.evaluator = CostEvaluator(self.costmap_gen)
        
        self.current_path = None
        self.replan_threshold = 150  # Cost at which the current path is considered unsafe

    def check_path_safety(self, path, costmap_grid) -> bool:
        if not path:
            return False
        # Quickly check if the existing path collides with the updated cost map
        for pt in path:
            gx, gy = self.costmap_gen.world_to_grid(pt.x, pt.y)
            if 0 <= gx < self.costmap_gen.nx and 0 <= gy < self.costmap_gen.ny:
                if costmap_grid[gx, gy] >= self.replan_threshold:
                    return False
        return True

    def run_step(self, ego: EgoState, drivable: DrivableSpace, 
                 tracked_objects: list, predictions: list, dt: float) -> PlanningOutput:
        
        start_time = time.perf_counter()
        replan_triggered = False

        # 1. Evaluate Behavior
        state, target_speed, emergency = self.fsm.evaluate(ego, tracked_objects, predictions, dt)

        if emergency:
            # Bypass path planning entirely to save milliseconds
            calc_time = (time.perf_counter() - start_time) * 1000
            return PlanningOutput(state, [], 0.0, True, True, calc_time, ego.timestamp)

        # 2. Build Updated Cost Map
        costmap_grid = self.costmap_gen.build_costmap(drivable, tracked_objects, predictions)

        # 3. Check if we need to replan
        path_is_safe = self.check_path_safety(self.current_path, costmap_grid)
        
        if not path_is_safe or self.current_path is None:
            replan_triggered = True
            # Generate new candidates
            candidates = self.sampler.generate_candidates(ego, target_speed)
            # Evaluate against new cost map
            best_path, _ = self.evaluator.evaluate_paths(candidates, costmap_grid)
            self.current_path = best_path

        calc_time = (time.perf_counter() - start_time) * 1000

        return PlanningOutput(
            behavior_state=state,
            trajectory=self.current_path if self.current_path else [],
            target_speed=target_speed,
            replan_triggered=replan_triggered,
            emergency_stop=False,
            computation_time_ms=calc_time,
            timestamp=ego.timestamp
        )

def simulate_real_time_loop():
    print("--- Starting M5 Real-Time Replanning Node ---")
    planner = M5PlannerNode()
    
    # We will simulate Scenario 4 (Dense Market)
    ego, drivable, statics, _ = get_scenario_data(4)
    dt = 0.1

    # Create M4 UDP receiver once
    m4_receiver = M4Adapter.create_receiver()

    # Start with no dynamic predictions
    dynamics = []

    print("Waiting for first M4 packet...")

    while True:
        first_data = M4Adapter.receive_latest(m4_receiver)

        if first_data is not None:
            dynamics = first_data["predictions"]
            ego = first_data["ego"]
            break

        time.sleep(0.01)

    print("First M4 packet received. Starting planner.")

    step = 0
    while True:
        new_data = M4Adapter.receive_latest(m4_receiver)

        if new_data is not None:
            dynamics = new_data["predictions"]
            ego = new_data["ego"]

        if new_data is not None:
            dynamics = new_data["predictions"]
            ego = new_data["ego"]

            print(
                f"[M5 EGO] x={ego.x:.2f}, y={ego.y:.2f}, "
                f"vx={ego.vx:.2f}, vy={ego.vy:.2f}, "
                f"yaw={ego.yaw:.2f}, t={ego.timestamp:.2f}"
            )
            
        # Run the planner iteration
        output = planner.run_step(ego, drivable, statics, dynamics, dt)
        
        # 2. Right after M5 generates its trajectory dictionary (e.g., `planned_data`), add:
        planned_data = {
            "behavior_state": output.behavior_state,
            "trajectory": [{"x": pt.x, "y": pt.y, "v_target": pt.v_target, "yaw": pt.yaw, "t": pt.t} for pt in output.trajectory],
            "target_speed": output.target_speed,
            "replan_triggered": output.replan_triggered,
            "emergency_stop": output.emergency_stop,
            "computation_time_ms": output.computation_time_ms,
            "timestamp": output.timestamp
        }
        pub_socket.send_json(planned_data)
        
        # print(f"Step {step} (t={ego.timestamp:.1f}s): "
        #       f"State=[{output.behavior_state}], "
        #       f"TargetSpd={output.target_speed:.1f}m/s, "
        #       f"Replanned={output.replan_triggered}, "
        #       f"Latency={output.computation_time_ms:.2f}ms")

        print("\n========== M5 OUTPUT ==========")
        print(f"Behavior State    : {output.behavior_state}")
        print(f"Target Speed      : {output.target_speed:.2f} m/s")
        print(f"Trajectory Points : {len(output.trajectory)}")
        print(f"Replan Triggered  : {output.replan_triggered}")
        print(f"Emergency Stop    : {output.emergency_stop}")
        print(f"Computation Time  : {output.computation_time_ms:.2f} ms")

        if output.trajectory:
            print("First Trajectory Point:")
            pt = output.trajectory[0]
            print(
                f"  x={pt.x:.2f}, y={pt.y:.2f}, "
                f"v_target={pt.v_target:.2f}, yaw={pt.yaw:.2f}"
            )

        print("================================\n")
              
        step += 1
        time.sleep(dt)

if __name__ == "__main__":
    simulate_real_time_loop()