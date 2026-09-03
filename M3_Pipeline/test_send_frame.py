"""
test_send_frame.py
====================
Sends ONE fake-but-valid frame straight to M2's real endpoint
(/api/v1/perception), using the test.jpg already sitting in your
Carla Object Detection folder, plus synthetic LiDAR/radar bytes
standing in for what M1/CARLA would normally send.

This lets you verify the FULL chain (M2 -> M3 -> M4) works end-to-end
WITHOUT needing CARLA running yet.

BEFORE RUNNING:
    1. M2 must be running:  python perception_server.py   (port 8000)
    2. M3 must be running:  python m3_server.py            (port 9000)
    3. Update TEST_JPG_PATH below to point at your actual test.jpg

RUN WITH:
    python test_send_frame.py
"""

import numpy as np
import requests

M2_URL = "http://localhost:8000/api/v1/perception"
TEST_JPG_PATH = "Carla Object Detection/test.jpg"  # adjust path if needed

# Fake LiDAR bytes: CARLA's standard (x, y, z, intensity) float32 format,
# a handful of points scattered in front of the "vehicle" -- enough to
# prove the bytes travel through M2 -> M3 and decode without crashing.
fake_lidar_points = np.random.uniform(
    low=[-5, -5, -0.1, 0.0], high=[30, 5, 2.0, 1.0], size=(500, 4)
).astype(np.float32)
fake_lidar_bytes = fake_lidar_points.tobytes()

# Fake radar bytes: CARLA's (velocity, azimuth, altitude, depth) format,
# one detection roughly ahead of the vehicle.
fake_radar_points = np.array(
    [[-2.5, 0.0, 0.0, 15.0]], dtype=np.float32  # closing at 2.5 m/s, 15m ahead
)
fake_radar_bytes = fake_radar_points.tobytes()


def send_test_frame():
    with open(TEST_JPG_PATH, "rb") as image_file:
        image_bytes = image_file.read()

    files = {
        "image": ("test.jpg", image_bytes, "image/jpeg"),
        "lidar_file": ("lidar.bin", fake_lidar_bytes, "application/octet-stream"),
        "radar_file": ("radar.bin", fake_radar_bytes, "application/octet-stream"),
    }
    data = {
        "sensor_id": "test_sensor",
        "frame_id": 1,
        "timestamp": 12.0,
        "ego_speed_mps": 12.0,
    }

    print(f"Sending test frame to {M2_URL} ...")
    response = requests.post(M2_URL, files=files, data=data, timeout=30)

    print(f"\nHTTP {response.status_code}")
    print(response.json())

    if response.status_code == 200:
        body = response.json()
        if body.get("forwarded_to_m3"):
            print("\n✅ M2 successfully forwarded this frame to M3.")
            print("Check M3's terminal logs now -- you should see a line like:")
            print('   "[frame 1] tracked_objects=... predictions=..."')
            print("The predictions count confirms M4 ran too (M4 has no server of its own --")
            print("it runs as function calls inside the M3 process).")
        else:
            print("\n⚠️  M2 did NOT successfully forward to M3 -- check M2's logs for the error.")


if __name__ == "__main__":
    send_test_frame()