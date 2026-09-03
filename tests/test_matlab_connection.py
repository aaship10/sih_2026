"""
Tests the MATLAB Engine API connection and attempts to load the M5 Simulink model.
"""

import sys
import os
import time

print("Importing MATLAB Engine API...")
try:
    import matlab.engine
except ImportError:
    print("ERROR: matlabengine is not installed.")
    print("Run: python -m pip install matlabengine")
    sys.exit(1)

def run_connection_test():
    print("Starting MATLAB Engine in the background...")
    print("(This usually takes 10-20 seconds on the first run...)")
    
    start_time = time.time()
    eng = matlab.engine.start_matlab()
    startup_time = time.time() - start_time
    
    print(f"SUCCESS: MATLAB Engine started in {startup_time:.1f} seconds!")
    
    # Send a simple command to MATLAB to prove it works
    eng.eval("disp('Hello from Python! The bridge is active.');", nargout=0)
    
    # Locate your behavior folder where the .slx is saved
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    behavior_dir = os.path.join(project_root, "behavior")
    
    print(f"Changing MATLAB working directory to: {behavior_dir}")
    eng.cd(behavior_dir, nargout=0)
    
    print("Loading the Simulink model (m5_planner_core.slx)...")
    try:
        # load_system loads the model into MATLAB memory without opening the heavy GUI
        eng.eval("load_system('m5_planner_core');", nargout=0)
        print("SUCCESS: Simulink model loaded perfectly!")
    except Exception as e:
        print(f"ERROR loading model. Make sure 'm5_planner_core.slx' is saved in the 'behavior' folder.")
        print(f"Details: {e}")
    finally:
        print("Closing MATLAB Engine...")
        eng.quit()

if __name__ == "__main__":
    run_connection_test()