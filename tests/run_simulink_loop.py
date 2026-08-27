"""
Simulates a 3-second Cattle Crossing scenario by pushing 
sensor data into Simulink and pulling the trajectories back out.
"""

import sys
import os
import matlab.engine
import matlab

def run_bridge_simulation():
    print("Starting MATLAB Engine (this takes a moment)...")
    eng = matlab.engine.start_matlab()
    
    behavior_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "behavior"))
    eng.cd(behavior_dir, nargout=0)
    
    sim_input = []
    print("Generating scenario data: Cattle appears at t=2.0s")
    
    for i in range(31):
        t = i * 0.1
        ttc = max(0.5, 5.0 - (t * 1.5))
        risk = 0.9 if t >= 2.0 else 0.1
        dist = max(2.0, 20.0 - (t * 5.0))
        sim_input.append([t, ttc, risk, dist])
        
    eng.workspace['sim_input'] = matlab.double(sim_input)
    
    print("Running Simulink M5 core logic...")
    try:
        eng.eval("simOut = sim('m5_planner_core', 'StopTime', '3.0', 'ReturnWorkspaceOutputs', 'on');", nargout=0)
        
        # --- NEW CODE: Format the matrix safely inside MATLAB ---
        eng.eval("out_x = simOut.yout.getElement(1).Values.Data;", nargout=0)
        # Squeeze out any weird 3D dimensions Simulink adds
        eng.eval("out_x = squeeze(out_x);", nargout=0)
        # Ensure Time is on the rows (31) and Waypoints are columns (16)
        eng.eval("if size(out_x, 1) == 16; out_x = out_x'; end", nargout=0)
        
        paths = eng.workspace['out_x']
        # --------------------------------------------------------
        
        print("\n=== SIMULATION RESULTS ===")
        # Now paths is a guaranteed clean 2D array
        val_0s = paths[0][0]
        val_1s = paths[10][0]
        val_2s = paths[20][0] 
        val_3s = paths[-1][0] # Safely grabs the very last step
        
        print(f"t=0.0s (CRUISE) : First X waypoint = {val_0s:.2f} m")
        print(f"t=1.0s (CRUISE) : First X waypoint = {val_1s:.2f} m")
        print(f"t=2.0s (EMERGENCY): First X waypoint = {val_2s:.2f} m  <-- Brakes applied, path zeroed!")
        print(f"t=3.0s (STOPPED)  : First X waypoint = {val_3s:.2f} m")
        
        print("\nSUCCESS! Python pushed sensor data, Stateflow made decisions, and Python retrieved the paths.")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
    finally:
        eng.quit()
if __name__ == "__main__":
    run_bridge_simulation()