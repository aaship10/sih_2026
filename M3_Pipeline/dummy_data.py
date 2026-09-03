"""
dummy_data.py
=============
Generates FAKE camera / LiDAR / radar data so you can build and test the
whole M3 pipeline before CARLA or M2's real output are ready.

Story we simulate:
  A pedestrian starts 15m ahead of the car and slowly walks sideways.
  An auto-rickshaw sits further ahead, roughly stationary.

Every function here returns data in the SAME FORMAT that the real
CARLA / M2 pipeline will eventually produce, so later you can delete
this file's calls and plug in the real sensors WITHOUT changing any
other file (lidar_processing.py, fusion.py, tracker.py stay untouched).

Coordinate frame used everywhere in this project (see transforms.py
for a full explanation):
    x = forward from the car (meters)
    y = left/right from the car (meters), positive = left
    z = up (meters)
"""

import numpy as np
from transforms import project_point_to_image, make_default_camera_intrinsics

# camera calibration used only by the static-object scenario functions
# below (get_pothole_scenario, get_speed_bump_scenario, etc.) to build
# realistic bounding boxes -- main.py's own scenario uses hand-placed
# boxes instead and doesn't need this.
_STATIC_CAMERA_INTRINSICS = make_default_camera_intrinsics(image_width=1280, image_height=720, fov_deg=90)
_STATIC_CAMERA_EXTRINSICS = {"offset": (1.5, 0.0, 1.4)}

# Fixed positions for the static/ground-surface objects that now appear
# alongside the pedestrian and auto-rickshaw in EVERY frame of
# generate_scenario(). Chosen to sit at different x positions so they
# don't spatially overlap with the pedestrian (stays near x=15) or the
# auto-rickshaw (x=25, y=-3).
POTHOLE_POSITION = (8.0, 0.0)          # in ego's own lane, close to car
POTHOLE_DEPTH = 0.08
SPEED_BUMP_X = 20.0                     # a strip spanning the full road width
SPEED_BUMP_HEIGHT = 0.10
# Barricade/static_obstacle: positioned so their (fixed) camera angle
# doesn't cross the pedestrian's angular sweep as it walks from y=2.0
# to y=-0.84 at x=15.01 (angle range roughly -0.056 to +0.133 rad).
# Without this separation, the pedestrian can momentarily project to
# the SAME pixel region as one of these distant static objects (a real
# depth-ambiguity issue -- see match_lidar_to_camera's docstring),
# which a real YOLO detector wouldn't produce (its bbox reflects actual
# image content each frame), but our fixed hand-placed dummy bbox does.
BARRICADE_POSITION = (30.0, -6.0)      # angle ratio -0.20, clear of pedestrian's sweep
STATIC_OBSTACLE_POSITION = (35.0, 8.0)  # angle ratio +0.23, clear of pedestrian's sweep


def _make_camera_bbox_for_point(point_3d, half_width_px=45, half_height_px=25):
    """Projects a 3D point to pixels and builds a bbox around it -- stands
    in for what a real YOLO detection box would look like."""
    u, v = project_point_to_image(point_3d, _STATIC_CAMERA_INTRINSICS, _STATIC_CAMERA_EXTRINSICS)
    return [u - half_width_px, v - half_height_px, u + half_width_px, v + half_height_px]


def get_camera_detections(frame_id: int, t: float, ped_y: float, ped_x: float, rick_x: float, rick_y: float):
    """
    Fakes M2's output format exactly as M2 sends it:
    {
        "class": ...,
        "confidence": ...,
        "bbox": [x_min, y_min, x_max, y_max],   # pixel coords
        "timestamp": ...,
        "frame_id": ...
    }

    We don't actually run a detector here -- we just hand-place a
    plausible bounding box for the pedestrian (its position depends on
    ped_x, ped_y so the box moves realistically as the pedestrian moves).
    The auto-rickshaw's box is properly PROJECTED from its real 3D
    position (rick_x, rick_y) using the same math as the static
    objects below -- important: an earlier version used a disconnected
    hardcoded pixel box here, which could coincidentally overlap a
    genuinely different object's projected position and cause a
    mismatch (see fusion.py's docstring on depth ambiguity for why).
    """
    # crude fake "projection": closer/more-central objects get bigger/centered boxes
    # (this is NOT the real projection math -- that lives in transforms.py and
    # is used for real 3D->2D projection of LiDAR points, not for faking bboxes)
    cx = 640 - ped_y * 40          # sideways position -> pixel column
    box_half_w = max(20, 300 - ped_x * 10)   # closer pedestrian -> wider box
    box_half_h = box_half_w * 1.3

    detections = [
        {
            "class": "pedestrian",
            "confidence": 0.91,
            "bbox": [
                cx - box_half_w, 360 - box_half_h,
                cx + box_half_w, 360 + box_half_h,
            ],
            "timestamp": round(t, 2),
            "frame_id": frame_id,
        },
        {
            "class": "auto_rickshaw",
            "confidence": 0.87,
            "bbox": _make_camera_bbox_for_point([rick_x, rick_y, 0.8], half_width_px=75, half_height_px=65),
            "timestamp": round(t, 2),
            "frame_id": frame_id,
        },
        # --- static / ground-surface classes: fixed position every frame ---
        {
            "class": "pothole",
            "confidence": 0.83,
            "bbox": _make_camera_bbox_for_point(
                [POTHOLE_POSITION[0], POTHOLE_POSITION[1], -POTHOLE_DEPTH]),
            "timestamp": round(t, 2),
            "frame_id": frame_id,
        },
        {
            "class": "speed_bump",
            "confidence": 0.79,
            "bbox": _make_camera_bbox_for_point(
                [SPEED_BUMP_X, 0.0, SPEED_BUMP_HEIGHT], half_width_px=90, half_height_px=15),
            "timestamp": round(t, 2),
            "frame_id": frame_id,
        },
        {
            "class": "barricade",
            "confidence": 0.88,
            "bbox": _make_camera_bbox_for_point(
                [BARRICADE_POSITION[0], BARRICADE_POSITION[1], 0.5], half_width_px=70, half_height_px=40),
            "timestamp": round(t, 2),
            "frame_id": frame_id,
        },
        {
            "class": "static_obstacle",
            "confidence": 0.7,
            "bbox": _make_camera_bbox_for_point(
                [STATIC_OBSTACLE_POSITION[0], STATIC_OBSTACLE_POSITION[1], 0.3], half_width_px=35, half_height_px=30),
            "timestamp": round(t, 2),
            "frame_id": frame_id,
        },
    ]
    return detections


def get_lidar_pointcloud(ped_x: float, ped_y: float, rick_x: float, rick_y: float,
                          n_ground_points: int = 800):
    """
    Fakes a raw LiDAR point cloud in the EGO (car) frame, already in meters.

    In real CARLA, you get points in the LiDAR sensor's own frame and
    must transform them into the ego frame first (see transforms.py).
    Here we skip that step and generate points directly in ego frame
    for simplicity -- but the rest of the pipeline works identically
    either way.

    Returns: numpy array of shape (N, 3) -> columns are x, y, z
    """
    rng = np.random.default_rng(42)
    points = []

    # 1) Ground points: a big flat sheet at z ~ 0, scattered around the car.
    #    This is what ground_removal() in lidar_processing.py must delete.
    gx = rng.uniform(0, 40, n_ground_points)
    gy = rng.uniform(-10, 10, n_ground_points)
    gz = rng.uniform(-0.05, 0.05, n_ground_points)  # near-zero height = road surface
    points.append(np.stack([gx, gy, gz], axis=1))

    # 2) Pedestrian body: ~40 points clustered around (ped_x, ped_y),
    #    roughly shin-to-head height (0.3m to 1.7m). Real LiDAR often
    #    gets weak/no returns right at ankle height due to the grazing
    #    beam angle, so starting at 0.3m (not 0.0m) is both more
    #    realistic AND keeps these points clearly separated from
    #    speed_bump's height band (see fusion.py's
    #    ROAD_SURFACE_HEIGHT_RANGES) -- otherwise a moving pedestrian's
    #    very lowest points could occasionally be mistaken for road
    #    texture just by falling in the same height range.
    n_ped = 40
    px = ped_x + rng.normal(0, 0.15, n_ped)
    py = ped_y + rng.normal(0, 0.15, n_ped)
    pz = rng.uniform(0.3, 1.7, n_ped)
    points.append(np.stack([px, py, pz], axis=1))

    # 3) Auto-rickshaw: ~90 points clustered in a bigger box shape.
    n_rick = 90
    rx = rick_x + rng.normal(0, 0.6, n_rick)
    ry = rick_y + rng.normal(0, 0.5, n_rick)
    rz = rng.uniform(0.0, 1.6, n_rick)
    points.append(np.stack([rx, ry, rz], axis=1))

    # 4) A handful of far-away junk points (e.g. a distant wall) that
    #    the ROI filter should discard.
    n_far = 30
    fx = rng.uniform(60, 90, n_far)
    fy = rng.uniform(-20, 20, n_far)
    fz = rng.uniform(0, 3, n_far)
    points.append(np.stack([fx, fy, fz], axis=1))

    # 5) Pothole: a shallow depression AT road height (negative z dip).
    #    This is exactly why naive ground removal would wipe it out --
    #    see fusion.py's ROAD_SURFACE_CLASSES handling.
    n_pothole = 20
    hx = POTHOLE_POSITION[0] + rng.uniform(-0.3, 0.3, n_pothole)
    hy = POTHOLE_POSITION[1] + rng.uniform(-0.3, 0.3, n_pothole)
    hz = rng.uniform(-POTHOLE_DEPTH - 0.02, -POTHOLE_DEPTH + 0.02, n_pothole)
    points.append(np.stack([hx, hy, hz], axis=1))

    # 6) Speed bump: a raised strip spanning the road width, only ~10cm tall.
    n_bump = 60
    sx = SPEED_BUMP_X + rng.uniform(-0.25, 0.25, n_bump)
    sy = rng.uniform(-3.5, 3.5, n_bump)
    sz = rng.uniform(SPEED_BUMP_HEIGHT - 0.02, SPEED_BUMP_HEIGHT + 0.02, n_bump)
    points.append(np.stack([sx, sy, sz], axis=1))

    # 7) Barricade: solid, clearly raised -- goes through the normal
    #    clustered LiDAR path (ground removal correctly leaves it alone).
    n_barricade = 45
    bax = BARRICADE_POSITION[0] + rng.normal(0, 0.4, n_barricade)
    bay = BARRICADE_POSITION[1] + rng.normal(0, 0.8, n_barricade)
    baz = rng.uniform(0.1, 1.0, n_barricade)
    points.append(np.stack([bax, bay, baz], axis=1))

    # 8) Static obstacle: generic catch-all raised object.
    n_obstacle = 35
    ox = STATIC_OBSTACLE_POSITION[0] + rng.normal(0, 0.3, n_obstacle)
    oy = STATIC_OBSTACLE_POSITION[1] + rng.normal(0, 0.3, n_obstacle)
    oz = rng.uniform(0.05, 0.6, n_obstacle)
    points.append(np.stack([ox, oy, oz], axis=1))

    return np.vstack(points)


def get_radar_detections(ped_x: float, ped_y: float, ped_vx: float, ped_vy: float,
                          rick_x: float, rick_y: float):
    """
    Fakes radar output in the ego frame.

    Real CARLA radar gives you range/azimuth/relative-velocity per
    detection; we convert straight to (x, y, vx, vy) here for
    simplicity, and radar_processing.py shows how to do that conversion
    for real CARLA data.

    Returns: list of dicts, each one detection.
    """
    detections = [
        {
            "position": [ped_x, ped_y],
            "relative_velocity": [ped_vx, ped_vy],
        },
        {
            "position": [rick_x, rick_y],
            "relative_velocity": [0.0, 0.0],   # auto-rickshaw roughly stationary
        },
    ]
    return detections


def generate_scenario(n_frames: int = 15, dt: float = 0.2):
    """
    Generates a full synthetic scenario: a pedestrian starting 15m ahead,
    2m to the side, walking sideways (crossing) at -1 m/s (moving toward
    y=0, i.e. toward the car's path). An auto-rickshaw sits ~25m ahead,
    stationary.

    Yields one (timestamp, camera_dets, lidar_points, radar_dets) tuple
    per frame -- exactly what main.py loops over.
    """
    ped_x, ped_y = 15.0, 2.0
    ped_vx, ped_vy = 0.0, -1.0   # walking sideways, toward the car's path
    rick_x, rick_y = 25.0, -3.0

    t = 12.0
    for frame_id in range(1800, 1800 + n_frames):
        cam = get_camera_detections(frame_id, t, ped_y, ped_x, rick_x, rick_y)
        lidar = get_lidar_pointcloud(ped_x, ped_y, rick_x, rick_y)
        radar = get_radar_detections(ped_x, ped_y, ped_vx, ped_vy, rick_x, rick_y)

        yield t, cam, lidar, radar

        # move the pedestrian for the next frame
        ped_x += ped_vx * dt
        ped_y += ped_vy * dt
        t += dt


# =======================================================================
# STATIC / GROUND-SURFACE CLASS SCENARIOS
# =======================================================================
# These four classes behave differently from normal moving traffic:
#   - pothole, speed_bump: sit AT road-surface height (see fusion.py's
#     ROAD_SURFACE_CLASSES -- they use the raw, non-ground-removed
#     point cloud instead of the normal clustered path)
#   - barricade, static_obstacle: normal raised objects, just with
#     zero velocity, so they go through the usual clustered path
# Each function below returns (camera_det, lidar_points, radar_dets)
# for ONE single frame -- unlike generate_scenario() above, these don't
# move over time, since none of these four classes move at all.
# =======================================================================

def _road_background(n_points=600, rng=None):
    """A generic flat road surface, z ~ 0, same idea as get_lidar_pointcloud() above."""
    rng = rng or np.random.default_rng(7)
    x = rng.uniform(0, 40, n_points)
    y = rng.uniform(-10, 10, n_points)
    z = rng.uniform(-0.03, 0.03, n_points)
    return np.stack([x, y, z], axis=1)


def get_pothole_scenario(pothole_center=(10.0, 0.0), depth=0.08, timestamp=5.0, frame_id=1001):
    """
    A shallow depression AT road height (negative z dip). This is
    exactly why naive ground removal (anything near z=0 -> delete)
    would wipe this out -- a real pothole often sits even lower than
    the road, not higher, making it easy to confuse with "just more
    ground."
    """
    rng = np.random.default_rng(1)
    road = _road_background(rng=rng)

    n_pothole_pts = 20
    px = pothole_center[0] + rng.uniform(-0.3, 0.3, n_pothole_pts)
    py = pothole_center[1] + rng.uniform(-0.3, 0.3, n_pothole_pts)
    pz = rng.uniform(-depth - 0.02, -depth + 0.02, n_pothole_pts)
    pothole_points = np.stack([px, py, pz], axis=1)

    lidar_points = np.vstack([road, pothole_points])

    camera_det = {
        "class": "pothole",
        "confidence": 0.83,
        "bbox": _make_camera_bbox_for_point([pothole_center[0], pothole_center[1], -depth]),
        "timestamp": timestamp,
        "frame_id": frame_id,
    }

    radar_dets = []  # potholes don't move and give no useful radar return
    return camera_det, lidar_points, radar_dets


def get_speed_bump_scenario(bump_center_x=15.0, bump_height=0.10, timestamp=5.0, frame_id=1002):
    """
    A raised strip AT road height (small positive z bump). Speed bumps
    stretch across the width of the road (a strip, not a small blob
    like a pothole), and are only slightly raised (8-12cm is typical)
    -- same "would get deleted by ground removal" problem as potholes.
    """
    rng = np.random.default_rng(2)
    road = _road_background(rng=rng)

    n_bump_pts = 60
    bx = bump_center_x + rng.uniform(-0.25, 0.25, n_bump_pts)   # narrow in driving direction
    by = rng.uniform(-3.5, 3.5, n_bump_pts)                      # spans the road width
    bz = rng.uniform(bump_height - 0.02, bump_height + 0.02, n_bump_pts)
    bump_points = np.stack([bx, by, bz], axis=1)

    lidar_points = np.vstack([road, bump_points])

    camera_det = {
        "class": "speed_bump",
        "confidence": 0.79,
        "bbox": _make_camera_bbox_for_point([bump_center_x, 0.0, bump_height], half_width_px=90, half_height_px=15),
        "timestamp": timestamp,
        "frame_id": frame_id,
    }

    radar_dets = []
    return camera_det, lidar_points, radar_dets


def get_barricade_scenario(position=(18.0, -1.0), timestamp=5.0, frame_id=1003):
    """
    Static, raised, road-construction style barrier. A solid raised
    object (like a car in that sense), so it goes through the NORMAL
    clustered path, not the raw-point path -- ground removal correctly
    leaves these points alone since they're well above road height.
    """
    rng = np.random.default_rng(3)
    road = _road_background(rng=rng)

    n_pts = 45
    bx = position[0] + rng.normal(0, 0.4, n_pts)
    by = position[1] + rng.normal(0, 0.8, n_pts)   # barricades are often wide/horizontal
    bz = rng.uniform(0.1, 1.0, n_pts)
    barricade_points = np.stack([bx, by, bz], axis=1)

    lidar_points = np.vstack([road, barricade_points])

    camera_det = {
        "class": "barricade",
        "confidence": 0.88,
        "bbox": _make_camera_bbox_for_point([position[0], position[1], 0.5], half_width_px=70, half_height_px=40),
        "timestamp": timestamp,
        "frame_id": frame_id,
    }

    radar_dets = []  # static objects usually give little/no radar return
    return camera_det, lidar_points, radar_dets


def get_static_obstacle_scenario(position=(22.0, 2.5), timestamp=5.0, frame_id=1004):
    """Generic catch-all static object (fallen debris, parked cart, etc.)."""
    rng = np.random.default_rng(4)
    road = _road_background(rng=rng)

    n_pts = 35
    ox = position[0] + rng.normal(0, 0.3, n_pts)
    oy = position[1] + rng.normal(0, 0.3, n_pts)
    oz = rng.uniform(0.05, 0.6, n_pts)
    obstacle_points = np.stack([ox, oy, oz], axis=1)

    lidar_points = np.vstack([road, obstacle_points])

    camera_det = {
        "class": "static_obstacle",
        "confidence": 0.7,   # generic catch-all class -- often lower confidence
        "bbox": _make_camera_bbox_for_point([position[0], position[1], 0.3], half_width_px=35, half_height_px=30),
        "timestamp": timestamp,
        "frame_id": frame_id,
    }

    radar_dets = []
    return camera_det, lidar_points, radar_dets


def get_all_static_class_scenarios():
    """Returns all 4 scenarios as a list of (name, camera_det, lidar_points, radar_dets)."""
    return [
        ("pothole", *get_pothole_scenario()),
        ("speed_bump", *get_speed_bump_scenario()),
        ("barricade", *get_barricade_scenario()),
        ("static_obstacle", *get_static_obstacle_scenario()),
    ]