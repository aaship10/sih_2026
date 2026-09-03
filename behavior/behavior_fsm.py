"""
Behavior Decision Finite State Machine for M5.
Evaluates risk, TTC (Time-to-Collision), and obstacle proximity.
"""

import math
from typing import List
from interfaces.data_types import EgoState, TrackedObject, DynamicPrediction

class BehaviorFSM:
    def __init__(self):
        self.current_state = "CRUISE"
        self.state_timer = 0.0
        self.min_state_time = 0.5  # Hysteresis: Hold state for at least 0.5s to prevent chattering
        self.speed_limit = 8.0     # m/s (~30 km/h for unstructured roads)

    def calculate_ttc(self, ego: EgoState, obj_pos: tuple, obj_vel: tuple) -> float:
        # Simple longitudinal Time-To-Collision calculation
        rel_x = obj_pos[0]  # Assuming object position is relative to ego (ego is at 0,0)
        rel_vx = ego.vx - obj_vel[0]
        
        if rel_vx <= 0.1:
            return float('inf')  # Not catching up
        return rel_x / rel_vx

    def evaluate(self, ego: EgoState, tracked_objects: List[TrackedObject], 
                 predictions: List[DynamicPrediction], dt: float) -> tuple[str, float, bool]:
        
        self.state_timer += dt
        
        # 1. Check for immediate critical threats (Bypasses hysteresis)
        for pred in predictions:
            if pred.risk_level == "CRITICAL" or pred.collision_probability > 0.8:
                # Sudden cattle crossing or pedestrian jump
                self._transition("EMERGENCY_BRAKE")
                return self.current_state, 0.0, True

        # If we just entered a state, don't change it immediately (unless it's an emergency)
        if self.state_timer < self.min_state_time:
            return self.current_state, self._get_target_speed(ego), False

        # 2. Evaluate environment for normal state transitions
        min_ttc = float('inf')
        closest_obj_dist = float('inf')
        lead_speed = self.speed_limit

        for obj in tracked_objects:
            # Check objects straight ahead (within a 3m lateral corridor)
            if abs(obj.position[1]) < 1.5 and obj.position[0] > 0:
                ttc = self.calculate_ttc(ego, obj.position, obj.velocity)
                min_ttc = min(min_ttc, ttc)
                
                if obj.position[0] < closest_obj_dist:
                    closest_obj_dist = obj.position[0]
                    lead_speed = max(0.0, obj.velocity[0])

        # State Transition Logic
        if min_ttc < 2.5:
            # Threat is approaching
            if closest_obj_dist < 8.0:
                self._transition("YIELD")
            else:
                self._transition("FOLLOW")
        elif min_ttc < 5.0 and self.current_state != "YIELD":
            # Check if we have space to bypass (simplified logic for now)
            self._transition("AVOID")
        else:
            self._transition("CRUISE")

        return self.current_state, self._get_target_speed(ego, lead_speed), False

    def _transition(self, new_state: str):
        if self.current_state != new_state:
            self.current_state = new_state
            self.state_timer = 0.0  # Reset timer on state change

    def _get_target_speed(self, ego: EgoState, lead_speed: float = None) -> float:
        if self.current_state == "CRUISE":
            return self.speed_limit
        elif self.current_state == "FOLLOW":
            return lead_speed if lead_speed is not None else self.speed_limit * 0.8
        elif self.current_state == "AVOID":
            return self.speed_limit * 0.6  # Slow down during avoidance
        elif self.current_state == "YIELD":
            return 0.0
        elif self.current_state == "EMERGENCY_BRAKE":
            return 0.0
        return self.speed_limit