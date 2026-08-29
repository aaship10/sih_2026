import time
import math
from interfaces.data_types import PlannedTrajectory, TrajectoryPoint

class DummyM5:

    def generate_avoidance_trajectory(self, start_x, start_y, start_yaw, target_v=5.0):
        traj = PlannedTrajectory()
        
        for i in range(50):
            # Normal forward progress
            forward_dist = i * 1.0 
            
            # Shift 3 meters left gradually over the first 10 meters
            lateral_shift = 0.0
            if i < 10:
                lateral_shift = (i / 10.0) * 3.0
            else:
                lateral_shift = 3.0
                
            # Calculate world coordinates with the lateral shift
            pt_x = start_x + (forward_dist * math.cos(start_yaw)) - (lateral_shift * math.sin(start_yaw))
            pt_y = start_y + (forward_dist * math.sin(start_yaw)) + (lateral_shift * math.cos(start_yaw))

            traj.points.append(
                TrajectoryPoint(x=pt_x, y=pt_y, heading=start_yaw, velocity=target_v)
            )
        traj.is_valid = True
        return traj

    def generate_straight_trajectory(self, start_x, start_y, start_yaw, target_v=5.0):
        traj = PlannedTrajectory()
        for i in range(50):
            # Generate points 1 meter apart
            pt_x = start_x + (i * 1.0) * math.cos(start_yaw)
            pt_y = start_y + (i * 1.0) * math.sin(start_yaw)
            traj.points.append(
                TrajectoryPoint(x=pt_x, y=pt_y, heading=start_yaw, velocity=target_v)
            )
        traj.is_valid = True
        return traj
    
    def generate_curved_trajectory(self, start_x, start_y, start_yaw, target_v=5.0):
        traj = PlannedTrajectory()
        
        # We will create a trajectory that goes straight for 10 meters, 
        # then does a U-turn/curve, then goes straight again.
        for i in range(100):
            if i < 20:
                # Go straight
                yaw = start_yaw
                pt_x = start_x + (i * 1.0) * math.cos(yaw)
                pt_y = start_y + (i * 1.0) * math.sin(yaw)
            else:
                # Turn left gradually (0.05 radians per meter)
                yaw = start_yaw + ((i - 20) * 0.05)
                # Calculate new x,y based on the previous point and new yaw
                prev_pt = traj.points[-1]
                pt_x = prev_pt.x + 1.0 * math.cos(yaw)
                pt_y = prev_pt.y + 1.0 * math.sin(yaw)

            traj.points.append(
                TrajectoryPoint(x=pt_x, y=pt_y, heading=yaw, velocity=target_v)
            )
        traj.is_valid = True
        return traj
    
    def generate_stop_trajectory(self, start_x, start_y, start_yaw):
        traj = PlannedTrajectory()
        
        for i in range(40):
            pt_x = start_x + (i * 1.0) * math.cos(start_yaw)
            pt_y = start_y + (i * 1.0) * math.sin(start_yaw)
            
            # Create a speed profile
            if i < 15:
                vel = 6.0  # Cruise at 6 m/s
            elif i < 30:
                vel = 6.0 - ((i - 15) * 0.4)  # Smoothly decelerate
            else:
                vel = 0.0  # Full stop for the remaining points

            traj.points.append(
                TrajectoryPoint(x=pt_x, y=pt_y, heading=start_yaw, velocity=vel)
            )
        traj.is_valid = True
        return traj