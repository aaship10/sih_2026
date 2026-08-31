"""
dummy_ego.py
============
Stub ego_state generator, matched in style to M3's dummy_data.py, so
M4 can be developed and tested before M1 (CARLA) or M5 (planner) exist.

Real integration later: M1/M5 will provide ego_state = {position,
velocity, heading, timestamp} each frame -- just swap the call site in
main.py / interface.py, nothing else in M4 changes (same pattern M3
used for its own dummy_data.py -> real CARLA swap).
"""


def generate_dummy_ego(t, speed=8.0, heading=0.0, start_xy=(0.0, 0.0)):
    """
    Ego moving at constant speed/heading from start_xy -- same
    constant-velocity assumption M3's dummy_data.py pedestrian uses,
    kept intentionally simple since ego motion realism isn't M4's job.
    """
    import math
    x = start_xy[0] + speed * math.cos(heading) * t
    y = start_xy[1] + speed * math.sin(heading) * t
    return {
        "position": [x, y],
        "velocity": speed,
        "heading": heading,
        "timestamp": round(t, 2),
    }