# """
# transforms.py
# ==============
# All coordinate-frame math lives here, in one place, so the rest of the
# code never has to think about rotations/projections directly.

# FRAMES USED IN THIS PROJECT
# ----------------------------
# 1. EGO FRAME (our "internal" frame -- everything M3 outputs uses this)
#        origin  = center of the car (roughly rear axle)
#        x = forward
#        y = left
#        z = up
#    This is the ONE frame every module (lidar, radar, fusion, tracker)
#    agrees to use internally. M4 also expects objects in this frame.

# 2. SENSOR FRAMES (LiDAR's own frame, radar's own frame, camera's own
#    frame) -- each sensor reports raw data relative to ITSELF, not the
#    car. In CARLA, every sensor actor has a known, fixed offset
#    (x, y, z, roll, pitch, yaw) relative to the ego vehicle, because you
#    define that offset yourself when you spawn the sensor. So the
#    sensor->ego transform is just simple, FIXED, KNOWN geometry --
#    nothing to calibrate or guess, unlike a real physical car.

# 3. IMAGE / PIXEL FRAME (2D, camera's flat picture)
#        Used only when projecting a 3D point onto the camera image, to
#        check whether it falls inside M2's bounding box.

# WHY THIS MATTERS
# -----------------
# LiDAR points and radar detections both arrive already in (or are
# converted here into) the EGO frame, so fusion.py can directly compare
# "where LiDAR says the object is" with "where radar says the object is"
# without unit or frame mismatches.
# """

# import numpy as np


# def sensor_to_ego(points_sensor_frame: np.ndarray, sensor_offset_xyz):
#     """
#     Transforms points from a sensor's own frame into the ego (car) frame.

#     In CARLA, `sensor_offset_xyz` is just the (x, y, z) you passed when
#     spawning the sensor relative to the vehicle, e.g. LiDAR mounted
#     0.5m forward, 0m sideways, 1.8m up: offset = (0.5, 0, 1.8).
#     (For simplicity this ignores rotation -- if your sensor is mounted
#     at an angle, add a rotation matrix multiply before the translation.
#     Most roof-mounted LiDARs are mounted flat, so this is often enough
#     for a student project.)

#     points_sensor_frame: (N, 3) array
#     sensor_offset_xyz: (3,) array-like, the sensor's mounting position
#                         relative to the ego vehicle's center

#     Returns: (N, 3) array of points in the ego frame.
#     """
#     offset = np.asarray(sensor_offset_xyz, dtype=float)
#     return points_sensor_frame + offset


# def project_point_to_image(point_ego_xyz, camera_intrinsics, camera_extrinsics):
#     """
#     Projects one 3D point (in ego frame) onto the 2D camera image, so we
#     can check "does this LiDAR point fall inside M2's bounding box?"

#     camera_intrinsics: dict with fx, fy (focal lengths in pixels) and
#                         cx, cy (image center in pixels). CARLA gives you
#                         these for free once you know the camera's FOV
#                         and image resolution -- no calibration needed
#                         since it's simulated:
#                             fx = fy = image_width / (2*tan(FOV/2))
#                             cx = image_width / 2
#                             cy = image_height / 2
#     camera_extrinsics: dict with the camera's (x, y, z) offset from the
#                         ego frame (same idea as sensor_offset_xyz above).

#     Returns: (u, v) pixel coordinates, or None if the point is behind
#              the camera (can't be seen).
#     """
#     x, y, z = point_ego_xyz
#     ox, oy, oz = camera_extrinsics["offset"]

#     # move point into the camera's own frame
#     xc, yc, zc = x - ox, y - oy, z - oz

#     # camera convention: forward = xc, so a point behind the camera
#     # (xc <= 0) cannot be projected
#     if xc <= 0.01:
#         return None

#     fx, fy = camera_intrinsics["fx"], camera_intrinsics["fy"]
#     cx, cy = camera_intrinsics["cx"], camera_intrinsics["cy"]

#     # standard pinhole camera projection.
#     # NOTE: image "right" = -y (since our y is left-positive) and
#     # image "down" = -z (since our z is up-positive) -- this sign flip
#     # is the #1 source of bugs people hit, so it's called out here.
#     u = cx + fx * (-yc / xc)
#     v = cy + fy * (-zc / xc)
#     return (u, v)


# def make_default_camera_intrinsics(image_width=1280, image_height=720, fov_deg=90):
#     """
#     Builds a plausible camera_intrinsics dict the way you would from a
#     CARLA RGB camera's blueprint attributes (image_size_x, image_size_y,
#     fov). Use this instead of hand-guessing fx/fy.
#     """
#     import math
#     fov_rad = math.radians(fov_deg)
#     fx = fy = image_width / (2 * math.tan(fov_rad / 2))
#     return {
#         "fx": fx, "fy": fy,
#         "cx": image_width / 2, "cy": image_height / 2,
#     }

"""
transforms.py
==============
All coordinate-frame math lives here, in one place, so the rest of the
code never has to think about rotations/projections directly.

FRAMES USED IN THIS PROJECT
----------------------------
1. EGO FRAME (our "internal" frame -- everything M3 outputs uses this)
       origin  = center of the car (roughly rear axle)
       x = forward
       y = left
       z = up
   This is the ONE frame every module (lidar, radar, fusion, tracker)
   agrees to use internally. M4 also expects objects in this frame.

2. SENSOR FRAMES (LiDAR's own frame, radar's own frame, camera's own
   frame) -- each sensor reports raw data relative to ITSELF, not the
   car. In CARLA, every sensor actor has a known, fixed offset
   (x, y, z, roll, pitch, yaw) relative to the ego vehicle, because you
   define that offset yourself when you spawn the sensor. So the
   sensor->ego transform is just simple, FIXED, KNOWN geometry --
   nothing to calibrate or guess, unlike a real physical car.

3. IMAGE / PIXEL FRAME (2D, camera's flat picture)
       Used only when projecting a 3D point onto the camera image, to
       check whether it falls inside M2's bounding box.

WHY THIS MATTERS
-----------------
LiDAR points and radar detections both arrive already in (or are
converted here into) the EGO frame, so fusion.py can directly compare
"where LiDAR says the object is" with "where radar says the object is"
without unit or frame mismatches.
"""

import numpy as np


def sensor_to_ego(points_sensor_frame: np.ndarray, sensor_offset_xyz):
    """
    Transforms points from a sensor's own frame into the ego (car) frame.

    In CARLA, `sensor_offset_xyz` is just the (x, y, z) you passed when
    spawning the sensor relative to the vehicle, e.g. LiDAR mounted
    0.5m forward, 0m sideways, 1.8m up: offset = (0.5, 0, 1.8).
    (For simplicity this ignores rotation -- if your sensor is mounted
    at an angle, add a rotation matrix multiply before the translation.
    Most roof-mounted LiDARs are mounted flat, so this is often enough
    for a student project.)

    points_sensor_frame: (N, 3) array
    sensor_offset_xyz: (3,) array-like, the sensor's mounting position
                        relative to the ego vehicle's center

    Returns: (N, 3) array of points in the ego frame.
    """
    offset = np.asarray(sensor_offset_xyz, dtype=float)
    return points_sensor_frame + offset


def project_point_to_image(point_ego_xyz, camera_intrinsics, camera_extrinsics):
    """
    Projects one 3D point (in ego frame) onto the 2D camera image, so we
    can check "does this LiDAR point fall inside M2's bounding box?"

    camera_intrinsics: dict with fx, fy (focal lengths in pixels) and
                        cx, cy (image center in pixels). CARLA gives you
                        these for free once you know the camera's FOV
                        and image resolution -- no calibration needed
                        since it's simulated:
                            fx = fy = image_width / (2*tan(FOV/2))
                            cx = image_width / 2
                            cy = image_height / 2
    camera_extrinsics: dict with the camera's (x, y, z) offset from the
                        ego frame (same idea as sensor_offset_xyz above).

    Returns: (u, v) pixel coordinates, or None if the point is behind
             the camera (can't be seen).
    """
    x, y, z = point_ego_xyz
    ox, oy, oz = camera_extrinsics["offset"]

    # move point into the camera's own frame
    xc, yc, zc = x - ox, y - oy, z - oz

    # camera convention: forward = xc, so a point behind the camera
    # (xc <= 0) cannot be projected
    if xc <= 0.01:
        return None

    fx, fy = camera_intrinsics["fx"], camera_intrinsics["fy"]
    cx, cy = camera_intrinsics["cx"], camera_intrinsics["cy"]

    # standard pinhole camera projection.
    # NOTE: image "right" = -y (since our y is left-positive) and
    # image "down" = -z (since our z is up-positive) -- this sign flip
    # is the #1 source of bugs people hit, so it's called out here.
    u = cx + fx * (-yc / xc)
    v = cy + fy * (-zc / xc)
    return (u, v)


def make_default_camera_intrinsics(image_width=1280, image_height=720, fov_deg=90):
    """
    Builds a plausible camera_intrinsics dict the way you would from a
    CARLA RGB camera's blueprint attributes (image_size_x, image_size_y,
    fov). Use this instead of hand-guessing fx/fy.
    """
    import math
    fov_rad = math.radians(fov_deg)
    fx = fy = image_width / (2 * math.tan(fov_rad / 2))
    return {
        "fx": fx, "fy": fy,
        "cx": image_width / 2, "cy": image_height / 2,
    }