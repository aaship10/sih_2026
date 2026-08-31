# """
# radar_processing.py
# ====================
# Radar's superpower is velocity: it directly measures how fast a target
# is moving toward/away from the sensor, which neither camera nor LiDAR
# can do on their own. This file just cleans/standardizes raw radar
# detections into a simple format the fusion step can use.

# CARLA's real radar sensor reports each detection as:
#     depth (range), azimuth, altitude, velocity (radial, i.e. toward/
#     away from the sensor)
# which you then convert to (x, y) position using basic trigonometry:
#     x = depth * cos(altitude) * cos(azimuth)
#     y = depth * cos(altitude) * sin(azimuth)
# This file's `from_carla_format()` shows that conversion. Our dummy
# data already hands you (x, y, vx, vy) directly to keep things simple
# while you're developing without CARLA connected yet.
# """

# import numpy as np


# def from_carla_format(range_m: float, azimuth_rad: float, altitude_rad: float,
#                        radial_velocity: float, sensor_offset_xyz=(0, 0, 0)):
#     """
#     Converts one raw CARLA radar detection (range/azimuth/altitude/
#     velocity) into ego-frame (x, y, z) position + a rough (vx, vy)
#     velocity estimate.

#     NOTE: radar only measures speed ALONG the line from sensor to
#     target (this is called "radial velocity") -- not full 2D/3D
#     velocity. For objects moving mostly toward/away from the car (the
#     most safety-critical case, e.g. crossing traffic, oncoming
#     vehicles) this radial estimate is usually good enough for a
#     student project. A more advanced system would combine several
#     radar returns over time to recover full velocity, but that's
#     beyond what's needed here.
#     """
#     x = range_m * np.cos(altitude_rad) * np.cos(azimuth_rad)
#     y = range_m * np.cos(altitude_rad) * np.sin(azimuth_rad)
#     z = range_m * np.sin(altitude_rad)

#     ox, oy, oz = sensor_offset_xyz
#     position = [x + ox, y + oy, z + oz]

#     # crude split of radial velocity into x/y using the same angle
#     # (good enough approximation for near-forward objects)
#     vx = radial_velocity * np.cos(azimuth_rad)
#     vy = radial_velocity * np.sin(azimuth_rad)

#     return {"position": position[:2], "relative_velocity": [vx, vy]}


# def process_radar_frame(raw_detections):
#     """
#     Standardizes a list of radar detections (already in ego-frame (x,y)
#     + (vx,vy) format -- either from dummy_data.py directly, or produced
#     by from_carla_format() above for real CARLA data) into a clean list
#     ready for fusion.

#     Returns: list of dicts:
#         {"position": [x, y], "velocity": [vx, vy]}
#     """
#     cleaned = []
#     for det in raw_detections:
#         cleaned.append({
#             "position": list(det["position"]),
#             "velocity": list(det["relative_velocity"]),
#         })
#     return cleaned

"""
radar_processing.py
====================
Radar's superpower is velocity: it directly measures how fast a target
is moving toward/away from the sensor, which neither camera nor LiDAR
can do on their own. This file just cleans/standardizes raw radar
detections into a simple format the fusion step can use.

CARLA's real radar sensor reports each detection as:
    depth (range), azimuth, altitude, velocity (radial, i.e. toward/
    away from the sensor)
which you then convert to (x, y) position using basic trigonometry:
    x = depth * cos(altitude) * cos(azimuth)
    y = depth * cos(altitude) * sin(azimuth)
This file's `from_carla_format()` shows that conversion. Our dummy
data already hands you (x, y, vx, vy) directly to keep things simple
while you're developing without CARLA connected yet.
"""

import numpy as np


def from_carla_format(range_m: float, azimuth_rad: float, altitude_rad: float,
                       radial_velocity: float, sensor_offset_xyz=(0, 0, 0)):
    """
    Converts one raw CARLA radar detection (range/azimuth/altitude/
    velocity) into ego-frame (x, y, z) position + a rough (vx, vy)
    velocity estimate.

    NOTE: radar only measures speed ALONG the line from sensor to
    target (this is called "radial velocity") -- not full 2D/3D
    velocity. For objects moving mostly toward/away from the car (the
    most safety-critical case, e.g. crossing traffic, oncoming
    vehicles) this radial estimate is usually good enough for a
    student project. A more advanced system would combine several
    radar returns over time to recover full velocity, but that's
    beyond what's needed here.
    """
    x = range_m * np.cos(altitude_rad) * np.cos(azimuth_rad)
    y = range_m * np.cos(altitude_rad) * np.sin(azimuth_rad)
    z = range_m * np.sin(altitude_rad)

    ox, oy, oz = sensor_offset_xyz
    position = [x + ox, y + oy, z + oz]

    # crude split of radial velocity into x/y using the same angle
    # (good enough approximation for near-forward objects)
    vx = radial_velocity * np.cos(azimuth_rad)
    vy = radial_velocity * np.sin(azimuth_rad)

    return {"position": position[:2], "relative_velocity": [vx, vy]}


def process_radar_frame(raw_detections):
    """
    Standardizes a list of radar detections (already in ego-frame (x,y)
    + (vx,vy) format -- either from dummy_data.py directly, or produced
    by from_carla_format() above for real CARLA data) into a clean list
    ready for fusion.

    Returns: list of dicts:
        {"position": [x, y], "velocity": [vx, vy]}
    """
    cleaned = []
    for det in raw_detections:
        cleaned.append({
            "position": list(det["position"]),
            "velocity": list(det["relative_velocity"]),
        })
    return cleaned