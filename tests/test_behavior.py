"""
Test script to verify the Behavior FSM transitions.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from behavior.behavior_fsm import BehaviorFSM
from interfaces.data_types import EgoState, TrackedObject, DynamicPrediction

def run_fsm_test():
    fsm = BehaviorFSM()
    ego = EgoState(x=0, y=0, vx=6.0, vy=0, yaw=0, accel=0, steering=0, timestamp=0)
    
    print("--- Testing Scenario: Sudden Cattle Crossing ---")
    
    # Tick 1: Clear road
    state, speed, emg = fsm.evaluate(ego, [], [], dt=0.1)
    print(f"Tick 1 (Clear): State={state}, TargetSpeed={speed}m/s, Emergency={emg}")
    
    # Tick 2: M4 predicts critical cattle crossing
    cattle_pred = DynamicPrediction(
        track_id=20, trajectories=[], risk_level="CRITICAL", collision_probability=0.9
    )
    state, speed, emg = fsm.evaluate(ego, [], [cattle_pred], dt=0.1)
    print(f"Tick 2 (Cattle Appears): State={state}, TargetSpeed={speed}m/s, Emergency={emg}")

if __name__ == "__main__":
    run_fsm_test()