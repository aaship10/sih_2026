"""
test_static_objects.py
========================
Runs the pothole / speed_bump / barricade / static_obstacle dummy
scenarios through your ACTUAL fusion pipeline (lidar_processing.py +
fusion.py) and prints what comes out -- so you can see with your own
eyes that the raw-point path correctly finds pothole/speed_bump
(which the normal path would miss), while barricade/static_obstacle
correctly go through the normal clustered path.

RUN WITH:
    python test_static_objects.py
"""

from dummy_data import get_all_static_class_scenarios, _STATIC_CAMERA_INTRINSICS as CAMERA_INTRINSICS, _STATIC_CAMERA_EXTRINSICS as CAMERA_EXTRINSICS
from lidar_processing import process_lidar_frame, get_raw_roi_points
from fusion import fuse_frame, ROAD_SURFACE_CLASSES


def run():
    print("=" * 78)
    print(f"{'Class':<18} {'Path used':<20} {'Result'}")
    print("=" * 78)

    for name, camera_det, lidar_points, radar_dets in get_all_static_class_scenarios():
        clusters = process_lidar_frame(lidar_points)
        raw_roi_points = get_raw_roi_points(lidar_points)

        fused = fuse_frame(
            [camera_det], clusters, radar_dets,
            CAMERA_INTRINSICS, CAMERA_EXTRINSICS, raw_roi_points=raw_roi_points,
        )
        result = fused[0]

        path_used = "raw points" if name in ROAD_SURFACE_CLASSES else "clustered"
        if result["position"] is None:
            outcome = "NOT FOUND (camera-only, no LiDAR match)"
        else:
            pos = result["position"]
            outcome = f"found at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"

        print(f"{name:<18} {path_used:<20} {outcome}")

    print("=" * 78)
    print("\nExpected: all 4 should show 'found at (...)' -- if pothole or")
    print("speed_bump show 'NOT FOUND', the raw-point matching isn't being")
    print("triggered correctly (check ROAD_SURFACE_CLASSES in fusion.py).")


if __name__ == "__main__":
    run()