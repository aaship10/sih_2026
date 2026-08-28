import time
from interfaces.carla_interface import CarlaInterface
from testing.dummy_m5 import DummyM5
from controllers.lateral_stanley import StanleyController
from controllers.longitudinal_pid import LongitudinalPID
from controllers.safety_supervisor import SafetySupervisor
from utils.coordinate_transforms import get_closest_waypoint_index
from utils.metrics_logger import MetricsLogger

def main():
    print("Initializing M6 Controller Node...")
    
    carla_iface = CarlaInterface()
    if not carla_iface.ego_vehicle:
        print("Waiting for ego vehicle to be spawned by M1...")
        return

    # Initialize M6 Components
    stanley = StanleyController(k_gain=6.0)
    pid = LongitudinalPID(kp=1.0, ki=0.1, kd=0.05)
    supervisor = SafetySupervisor(timeout_threshold=1.0)
    logger = MetricsLogger()
    dummy_planner = DummyM5()

    # Control Loop Frequency (50 Hz)
    dt = 0.02
    
    # Get initial position for dummy trajectory
    ego_x, ego_y, ego_yaw, ego_speed = carla_iface.get_ego_state()
    current_trajectory = dummy_planner.generate_curved_trajectory(ego_x, ego_y, ego_yaw, target_v=6.0)

    try:
        while True:
            start_time = time.time()

            # 1. Get Ego State
            ego_x, ego_y, ego_yaw, ego_speed = carla_iface.get_ego_state()

            # --- THE FIX: Keep dummy trajectory fresh ---
            current_trajectory.timestamp = time.time()
            # --------------------------------------------

            # 2. Safety Supervision
            state = supervisor.evaluate_state(current_trajectory)

            if state == "EMERGENCY_BRAKE":
                carla_iface.apply_control(steer=0.0, throttle=0.0, brake=1.0, hand_brake=True)
                
            elif state == "WAIT_FOR_PLAN":
                carla_iface.apply_control(steer=0.0, throttle=0.0, brake=1.0)

            elif state == "TRACKING":
                # Find closest point
                idx = get_closest_waypoint_index(ego_x, ego_y, current_trajectory.points)
                
                # Check if we reached the end of the dummy path
                if idx >= len(current_trajectory.points) - 2:
                    print("Reached end of dummy trajectory. Stopping.")
                    carla_iface.apply_control(steer=0.0, throttle=0.0, brake=1.0)
                    break # Exit the loop to see metrics

                target_point = current_trajectory.points[idx]

                # Lookahead for target speed (prevents braking too late)
                lookahead_idx = min(idx + 3, len(current_trajectory.points) - 1)
                target_speed = current_trajectory.points[lookahead_idx].velocity

                # Calculate Controls
                steer, cte = stanley.calculate_steering(ego_x, ego_y, ego_yaw, ego_speed, target_point)
                throttle, brake, speed_error = pid.calculate_throttle_brake(target_speed, ego_speed)

                # Log metrics
                logger.log_step(cte, speed_error)

                # Apply to CARLA
                carla_iface.apply_control(steer, throttle, brake)

            # Enforce loop rate
            elapsed = time.time() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\nStopping M6 Controller Node.")
        carla_iface.apply_control(steer=0.0, throttle=0.0, brake=1.0)
        logger.print_summary()

if __name__ == '__main__':
    main()