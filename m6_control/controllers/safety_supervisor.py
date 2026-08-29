import time

class SafetySupervisor:
    def __init__(self, timeout_threshold=0.5):
        self.state = "WAIT_FOR_PLAN"
        self.timeout_threshold = timeout_threshold

    def evaluate_state(self, current_trajectory):
        if not current_trajectory or not current_trajectory.points:
            return "WAIT_FOR_PLAN"

        if current_trajectory.target_behavior == "EMERGENCY":
            return "EMERGENCY_BRAKE"

        time_since_plan = time.time() - current_trajectory.timestamp
        if time_since_plan > self.timeout_threshold:
            print("[WARNING] Planner Timeout! Triggering Emergency Brake.")
            return "EMERGENCY_BRAKE"

        if not current_trajectory.is_valid:
            print("[WARNING] Invalid Trajectory Received! Triggering Emergency Brake.")
            return "EMERGENCY_BRAKE"

        return "TRACKING"