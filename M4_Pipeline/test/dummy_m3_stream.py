"""
dummy_m3_stream.py
===================
Section 38: a dummy tracked-object generator so M4 can be built and
tested BEFORE M3's real tracker is wired in -- without importing
M3_Pipeline at all. Produces frames in exactly M3's to_output_dict()
format (verified against tracker.py), so swapping this for the real
M3 stream later means changing only main.py's import, nothing in
buffer.py/derive.py/interface.py/etc.

Scenario generated: a pedestrian starts 15m ahead, 2m to the side,
walking sideways at -1 m/s toward the ego's path (test case for
Section 32's "sudden pedestrian" scenario), plus a stationary
auto-rickshaw further ahead -- deliberately similar to M3's own
dummy_data.py scenario so results are directly comparable once real
M3 output is plugged in.
"""


def generate_dummy_tracked_objects(n_frames=15, dt=0.2):
    """
    Yields (timestamp, tracked_objects) per frame, tracked_objects
    matching M3's Track.to_output_dict() schema exactly:
        track_id, class, position [x,y,z], velocity [vx,vy,0.0],
        size [l,w,h], confidence, age, last_seen, sensor_sources, timestamp
    """
    ped_x, ped_y = 15.0, 2.0
    ped_vx, ped_vy = 0.0, -1.0
    rick_x, rick_y = 25.0, -3.0

    t = 12.0
    for i in range(n_frames):
        tracked_objects = [
            {
                "track_id": 17,
                "class": "pedestrian",
                "position": [round(ped_x, 2), round(ped_y, 2), 0.0],
                "velocity": [round(ped_vx, 2), round(ped_vy, 2), 0.0],
                "size": [0.5, 0.5, 1.7],
                "confidence": 0.9,
                "age": i + 1,
                "last_seen": round(t, 2),
                "sensor_sources": ["camera", "lidar", "radar"],
                "timestamp": round(t, 2),
            },
            {
                "track_id": 23,
                "class": "rickshaw",
                "position": [round(rick_x, 2), round(rick_y, 2), 0.0],
                "velocity": [0.0, 0.0, 0.0],
                "size": [2.6, 1.4, 1.8],
                "confidence": 0.87,
                "age": i + 1,
                "last_seen": round(t, 2),
                "sensor_sources": ["camera", "lidar", "radar"],
                "timestamp": round(t, 2),
            },
        ]
        yield t, tracked_objects

        ped_x += ped_vx * dt
        ped_y += ped_vy * dt
        t += dt