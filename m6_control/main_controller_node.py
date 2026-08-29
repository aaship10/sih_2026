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
    stanley = StanleyController(k_gain=4.0) 
    pid = LongitudinalPID(kp=1.0, ki=0.1, kd=0.05)
    supervisor = SafetySupervisor(timeout_threshold=1.0)
    logger = MetricsLogger()
    dummy_planner = DummyM5()

    # Control Loop Frequency (50 Hz)
    dt = 0.02
    
    # Get initial position
    ego_x, ego_y, ego_yaw, ego_speed = carla_iface.get_ego_state()
    
    # Start with a normal straight trajectory
    current_trajectory = dummy_planner.generate_straight_trajectory(ego_x, ego_y, ego_yaw, target_v=6.0)
    
    # Track when we started so we can trigger the replan event
    mission_start_time = time.time()
    has_replanned = False

    try:
        while True:
            start_time = time.time()

            # 1. Get Ego State
            ego_x, ego_y, ego_yaw, ego_speed = carla_iface.get_ego_state()

            # --- PHASE 14: TRIGGER REPLAN AFTER 2 SECONDS ---
            if not has_replanned and (time.time() - mission_start_time > 2.0):
                print("\n[EVENT] Sudden Obstacle! Generating Avoidance Path...\n")
                # Generate new path from the vehicle's CURRENT position
                current_trajectory = dummy_planner.generate_avoidance_trajectory(ego_x, ego_y, ego_yaw, target_v=6.0)
                has_replanned = True
            # ------------------------------------------------

            # Keep dummy trajectory fresh so SafetySupervisor doesn't trigger a timeout
            current_trajectory.timestamp = time.time()

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
                    # Hold the stop so we can observe the final behavior
                    carla_iface.apply_control(steer=0.0, throttle=0.0, brake=1.0)
                    print("Reached end of dummy trajectory. Holding stop. Press Ctrl+C to view metrics.")
                else:
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
                    
                    # Live terminal output for debugging
                    print(f"Target V: {target_speed:.1f} m/s | Ego V: {ego_speed:.1f} m/s | Steer: {steer:.3f} | CTE: {cte:.3f} m")

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