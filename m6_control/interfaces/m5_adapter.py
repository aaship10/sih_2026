import math
import time
from interfaces.data_types import PlannedTrajectory, TrajectoryPoint

class M5Adapter:
    @staticmethod
    def parse_message(m5_dict, ego_x, ego_y, ego_yaw):
        traj = PlannedTrajectory()
        traj.timestamp = time.time()  # Override with system time to avoid timeout

        
        if m5_dict.get("emergency_brake", False):
            traj.target_behavior = "EMERGENCY"
        else:
            traj.target_behavior = m5_dict.get("behavior_state", "TRACK")
            
        target_v = m5_dict.get("target_speed", m5_dict.get("target_speed_ms", 0.0))

        raw_points = m5_dict.get("trajectory", [])
        
        # We need a temporary list of global points to calculate heading properly
        global_points = []
        
        for pt in raw_points:
            local_x = pt["x"]
            local_y = pt["y"]
            
            # --- The Fix: Transform Local to Global Coordinates ---
            global_x = ego_x + (local_x * math.cos(ego_yaw)) - (local_y * math.sin(ego_yaw))
            global_y = ego_y + (local_x * math.sin(ego_yaw)) + (local_y * math.cos(ego_yaw))
            
            global_points.append((global_x, global_y))
            
        # Now create the trajectory points with correct global headings
        for i in range(len(global_points)):
            pt_x, pt_y = global_points[i]
            
            if i < len(global_points) - 1:
                next_x, next_y = global_points[i+1]
                heading = math.atan2(next_y - pt_y, next_x - pt_x)
            else:
                heading = traj.points[-1].yaw if traj.points else ego_yaw
                
            pt_v_target = raw_points[i].get("v_target", target_v)
                
            traj.points.append(
                TrajectoryPoint(x=pt_x, y=pt_y, yaw=heading, v_target=pt_v_target)
            )
            
        traj.is_valid = len(traj.points) > 0
        return traj