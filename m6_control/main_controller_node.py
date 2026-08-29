import sys
import os
# Ensure Python can find the root modules when running from inside m6_control
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import zmq
from m6_control.interfaces.carla_interface import CarlaInterface
from m6_control.interfaces.m5_adapter import M5Adapter
from m6_control.controllers.lateral_stanley import StanleyController
from m6_control.controllers.longitudinal_pid import LongitudinalPID
from m6_control.controllers.safety_supervisor import SafetySupervisor
from m6_control.utils.coordinate_transforms import get_closest_waypoint_index
from m6_control.utils.metrics_logger import MetricsLogger

def main():
    print("Initializing M6 Controller Node (Integrated)...")
    
    carla_iface = CarlaInterface()
    
    # Wait continuously until the vehicle is spawned
    while not carla_iface.ego_vehicle:
        print("Waiting for ego vehicle to be spawned by M1/CARLA...")
        time.sleep(2.0)
        # Re-poll the CARLA world for the vehicle
        carla_iface = CarlaInterface()
        
    print("Ego vehicle found! Connecting to M5...")

    # Setup ZeroMQ Subscriber to listen to M5
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://127.0.0.1:5555")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    stanley = StanleyController(k_gain=4.0) 
    pid = LongitudinalPID(kp=1.0, ki=0.1, kd=0.05)
    supervisor = SafetySupervisor(timeout_threshold=1.0)
    logger = MetricsLogger()

    dt = 0.02
    current_trajectory = None

    try:
        while True:
            start_time = time.time()
            ego_x, ego_y, ego_yaw, ego_speed = carla_iface.get_ego_state()

            # 2. Pull Data from M5
            try:
                m5_msg = socket.recv_json(flags=zmq.NOBLOCK)
                print(f"\n[NETWORK SUCCESS] Received message with {len(m5_msg.get('trajectory', []))} points!")
                m5_msg["timestamp"] = time.time() 
                current_trajectory = M5Adapter.parse_message(m5_msg, ego_x, ego_y, ego_yaw)
            except zmq.Again:
                # No message this tick
                pass 
            except Exception as e:
                print(f"[NETWORK ERROR] Failed to parse message: {e}")

            # 3. Raw Execution
            if not current_trajectory or not current_trajectory.points:
                # We will only print this once every 50 ticks to not spam the console
                if int(time.time() * 10) % 10 == 0:
                    print("DEBUG: Waiting for M5 data over ZMQ...")
                carla_iface.apply_control(steer=0.0, throttle=0.0, brake=1.0)
            else:
                idx = get_closest_waypoint_index(ego_x, ego_y, current_trajectory.points)
                target_point = current_trajectory.points[idx]
                lookahead_idx = min(idx + 3, len(current_trajectory.points) - 1)
                target_speed = current_trajectory.points[lookahead_idx].v_target

                steer, cte = stanley.calculate_steering(ego_x, ego_y, ego_yaw, ego_speed, target_point)
                throttle, brake, speed_error = pid.calculate_throttle_brake(target_speed, ego_speed)

                print(f"DEBUG | EgoV: {ego_speed:.1f} | TargetV: {target_speed:.1f} | Thr: {throttle:.2f} | Brk: {brake:.2f} | CTE: {cte:.2f}")
                carla_iface.apply_control(steer, throttle, brake)

            elapsed = time.time() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\nStopping M6 Controller Node.")
        carla_iface.apply_control(steer=0.0, throttle=0.0, brake=1.0)

if __name__ == '__main__':
    main()