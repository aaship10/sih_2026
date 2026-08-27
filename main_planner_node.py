"""
Master Node for M5 (Behavior & Local Planning).
Simulates a 10 Hz real-time planning loop with dynamic replanning triggers.
"""

import time
import numpy as np
from interfaces.data_types import EgoState, DrivableSpace, TrackedObject, DynamicPrediction, PlanningOutput
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
    ego, drivable, statics, dynamics = get_scenario_data(4)
    dt = 0.1  # 10 Hz clock
    
    for step in range(5):
        ego.timestamp += dt
        
        # Simulate ego moving forward slightly
        if step > 0:
            ego.x += ego.vx * dt
            
        # Run the planner iteration
        output = planner.run_step(ego, drivable, statics, dynamics, dt)
        
        print(f"Step {step} (t={ego.timestamp:.1f}s): "
              f"State=[{output.behavior_state}], "
              f"TargetSpd={output.target_speed:.1f}m/s, "
              f"Replanned={output.replan_triggered}, "
              f"Latency={output.computation_time_ms:.2f}ms")

if __name__ == "__main__":
    simulate_real_time_loop()