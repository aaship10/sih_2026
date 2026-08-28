import numpy as np

class LongitudinalPID:
    def __init__(self, kp=1.0, ki=0.1, kd=0.05, max_throttle=1.0, max_brake=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_throttle = max_throttle
        self.max_brake = max_brake
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.dt = 0.02  # Assuming 50Hz control loop

    def calculate_throttle_brake(self, target_speed, current_speed):
        error = target_speed - current_speed
        
        # Anti-windup: only integrate if we aren't saturating the actuators
        if -2.0 < error < 2.0: 
            self.integral += error * self.dt
            # Clamp integral
            self.integral = np.clip(self.integral, -5.0, 5.0)
        else:
            self.integral = 0.0

        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        control_effort = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        throttle = 0.0
        brake = 0.0

        if control_effort > 0:
            throttle = np.clip(control_effort, 0.0, self.max_throttle)
        else:
            brake = np.clip(abs(control_effort), 0.0, self.max_brake)

        return float(throttle), float(brake), error