#!/usr/bin/env python3
"""SIH 2026 - Scenario 1 CARLA client with RGB/LiDAR/radar -> M2.

The simulator remains synchronous: exactly one world.tick() occurs in step().
Sensor callbacks only place data into frame-indexed buffers. After the tick,
step() collects the matching RGB, LiDAR, and radar samples and queues one
multipart/form-data request for M2.

The M2 endpoint expected by this client is:
    POST http://127.0.0.1:8000/api/v1/perception

Form fields:
    sensor_id, frame_id, timestamp, ego_speed_mps
Files:
    image       PNG encoded RGB camera image
    lidar_file  CARLA raw LiDAR bytes (float32 x,y,z,intensity tuples)
    radar_file  CARLA raw radar bytes (float32 velocity,altitude,azimuth,depth tuples)
"""

from __future__ import annotations

import argparse
import csv
import math
import queue
import random
import threading
import time
from pathlib import Path
from typing import Any

import carla
import cv2
import httpx
import numpy as np

TOWN = "Town07"
FIXED_DELTA_SECONDS = 0.05
LOG_DIR = Path("logs")
M2_URL = "http://127.0.0.1:8000/api/v1/perception"
M2_TIMEOUT_SECONDS = 5.0
TRAFFIC_MANAGER_PORT = 8001  # M2 uses port 8000, so do not use CARLA's default TM port.
SENSOR_WAIT_SECONDS = 2.0
UPLOAD_QUEUE_SIZE = 4
DROP_OLD_FRAMES_WHEN_BUSY = True
SENSOR_ID = "carla-scenario1-ego"

# Sensor placement relative to the ego vehicle.
CAMERA_TRANSFORM = carla.Transform(
    carla.Location(x=1.5, z=2.2),
    carla.Rotation(pitch=-5.0),
)
LIDAR_TRANSFORM = carla.Transform(carla.Location(x=0.0, z=2.3))
RADAR_TRANSFORM = carla.Transform(
    carla.Location(x=2.0, z=1.2),
    carla.Rotation(pitch=0.0),
)

EGO_SPEED_MULTIPLIER = 10.0
PEDESTRIAN_TRIGGER_DISTANCE_M = 15.0
CATTLE_TRIGGER_DISTANCE_M = 20.0
PEDESTRIAN_CROSSING_TIME_S = 4.0
CATTLE_CROSSING_TIME_S = 3.0


class FrameSensorBuffer:
    """Thread-safe frame-indexed buffer for CARLA sensor callbacks."""

    def __init__(self, name: str, max_frames: int = 32) -> None:
        self.name = name
        self.max_frames = max_frames
        self._condition = threading.Condition()
        self._frames: dict[int, Any] = {}

    def put(self, frame: int, data: Any) -> None:
        with self._condition:
            self._frames[int(frame)] = data
            while len(self._frames) > self.max_frames:
                self._frames.pop(min(self._frames))
            self._condition.notify_all()

    def get(self, frame: int, timeout: float) -> Any:
        deadline = time.monotonic() + timeout
        with self._condition:
            while int(frame) not in self._frames:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for {self.name} frame {frame}")
                self._condition.wait(remaining)
            return self._frames.pop(int(frame))


class M2Uploader:
    """Background HTTP uploader using binary multipart fields instead of Base64."""

    def __init__(self, url: str) -> None:
        self.url = url
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=UPLOAD_QUEUE_SIZE)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="m2-uploader", daemon=True)
        self.sent = 0
        self.failed = 0
        self.dropped = 0
        self._thread.start()

    def submit(self, packet: dict[str, Any]) -> bool:
        try:
            self._queue.put_nowait(packet)
            return True
        except queue.Full:
            if not DROP_OLD_FRAMES_WHEN_BUSY:
                self.dropped += 1
                return False
            try:
                self._queue.get_nowait()  # Drop the oldest queued frame.
                self.dropped += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(packet)
                return True
            except queue.Full:
                self.dropped += 1
                return False

    def close(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=M2_TIMEOUT_SECONDS + 1.0)

    def _run(self) -> None:
        with httpx.Client(timeout=M2_TIMEOUT_SECONDS) as client:
            while not self._stop.is_set():
                try:
                    packet = self._queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if packet is None:
                    break
                frame_id = packet["frame_id"]
                files = {
                    "image": (f"rgb_{frame_id}.png", packet["image"], "image/png"),
                    "lidar_file": (f"lidar_{frame_id}.bin", packet["lidar"], "application/octet-stream"),
                    "radar_file": (f"radar_{frame_id}.bin", packet["radar"], "application/octet-stream"),
                }
                data = {
                    "sensor_id": packet["sensor_id"],
                    "frame_id": str(frame_id),
                    "timestamp": repr(packet["timestamp"]),
                    "ego_speed_mps": repr(packet["ego_speed_mps"]),
                }
                try:
                    response = client.post(self.url, data=data, files=files)
                    response.raise_for_status()
                    self.sent += 1
                    if self.sent % 20 == 0:
                        print(f"[M2] sent={self.sent} failed={self.failed} dropped={self.dropped}", flush=True)
                except (httpx.HTTPError, OSError) as exc:
                    self.failed += 1
                    print(f"[M2][WARN] frame {frame_id} upload failed: {exc}", flush=True)


class Scenario1VillageRoad:
    def __init__(self, host: str, port: int, ego_spawn_index: int, m2_url: str, tm_port: int) -> None:
        self.host, self.port, self.ego_spawn_index = host, port, ego_spawn_index
        self.m2_url, self.tm_port = m2_url, tm_port
        self.client = self.world = self.tm = self.map = None
        self.ego = self.bike = self.pushcart = self.pedestrian = self.cattle = None
        self.collision_sensor = None
        self.rgb_camera = self.lidar_sensor = self.radar_sensor = None
        self.pedestrian_controller = self.cattle_controller = None
        self.rgb_frames = FrameSensorBuffer("RGB")
        self.lidar_frames = FrameSensorBuffer("LiDAR")
        self.radar_frames = FrameSensorBuffer("radar")
        self.uploader: M2Uploader | None = None
        self.collision_events: list[dict[str, Any]] = []
        self._collision_keys: set[tuple[int, int]] = set()
        self.trajectory_log: list[dict[str, Any]] = []
        self.frame_count = 0
        self.start_time = None
        self.pedestrian_triggered = self.cattle_triggered = False
        self.pedestrian_crossing_progress = self.cattle_crossing_progress = 0.0
        self.pedestrian_start_loc = self.pedestrian_target_loc = None
        self.cattle_start_loc = self.cattle_target_loc = None
        self.cattle_rotation = carla.Rotation()
        self.pedestrian_rotation = carla.Rotation()

    def setup(self) -> None:
        print("[SETUP] Connecting to CARLA...")
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(TOWN)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self.world.apply_settings(settings)
        self.tm = self.client.get_trafficmanager(self.tm_port)
        self.tm.set_synchronous_mode(True)
        self.world.set_weather(carla.WeatherParameters(cloudiness=40.0, sun_altitude_angle=55.0, fog_density=15.0, fog_distance=60.0))
        self.map = self.world.get_map()
        bp_lib = self.world.get_blueprint_library()
        spawn_points = self.map.get_spawn_points()
        if self.ego_spawn_index >= len(spawn_points):
            raise ValueError(f"Invalid spawn index {self.ego_spawn_index}")
        ego_bp = bp_lib.find("vehicle.tesla.model3")
        ego_bp.set_attribute("role_name", "ego_vehicle")
        self.ego = self.world.spawn_actor(ego_bp, spawn_points[self.ego_spawn_index])

        collision_bp = bp_lib.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.ego)
        self.collision_sensor.listen(self._on_collision)
        self._setup_perception_sensors(bp_lib)

        ahead_wp = self._forward_waypoint(spawn_points[self.ego_spawn_index], 15.0)
        self.bike = self._spawn_bike(ahead_wp)
        if self.bike:
            self.bike.set_autopilot(True, self.tm.get_port())
            self.tm.vehicle_percentage_speed_difference(self.bike, -20)
        self.pedestrian = self._spawn_pedestrian(ahead_wp)
        if self.pedestrian:
            self.pedestrian_start_loc = self._copy_location(self.pedestrian.get_location())
            self.pedestrian_rotation = self.pedestrian.get_transform().rotation
        self.pushcart = self._spawn_pushcart(ahead_wp)
        self.cattle = self._spawn_cattle(ahead_wp)
        if self.cattle:
            self.cattle_start_loc = self._copy_location(self.cattle.get_location())
            self.cattle_rotation = self.cattle.get_transform().rotation
        self.start_time = time.time()
        self.uploader = M2Uploader(self.m2_url)
        print(f"[SETUP] RGB + LiDAR + radar attached; M2={self.m2_url}", flush=True)

    def _setup_perception_sensors(self, bp_lib) -> None:
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "1280")
        camera_bp.set_attribute("image_size_y", "720")
        camera_bp.set_attribute("fov", "90")
        camera_bp.set_attribute("sensor_tick", str(FIXED_DELTA_SECONDS))
        self.rgb_camera = self.world.spawn_actor(camera_bp, CAMERA_TRANSFORM, attach_to=self.ego)
        self.rgb_camera.listen(lambda image: self.rgb_frames.put(image.frame, image))

        lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", "32")
        lidar_bp.set_attribute("range", "80.0")
        lidar_bp.set_attribute("points_per_second", "640000")
        lidar_bp.set_attribute("rotation_frequency", str(1.0 / FIXED_DELTA_SECONDS))
        lidar_bp.set_attribute("upper_fov", "10.0")
        lidar_bp.set_attribute("lower_fov", "-30.0")
        lidar_bp.set_attribute("sensor_tick", str(FIXED_DELTA_SECONDS))
        self.lidar_sensor = self.world.spawn_actor(lidar_bp, LIDAR_TRANSFORM, attach_to=self.ego)
        self.lidar_sensor.listen(lambda scan: self.lidar_frames.put(scan.frame, bytes(scan.raw_data)))

        radar_bp = bp_lib.find("sensor.other.radar")
        radar_bp.set_attribute("horizontal_fov", "35.0")
        radar_bp.set_attribute("vertical_fov", "20.0")
        radar_bp.set_attribute("range", "80.0")
        radar_bp.set_attribute("points_per_second", "1500")
        radar_bp.set_attribute("sensor_tick", str(FIXED_DELTA_SECONDS))
        self.radar_sensor = self.world.spawn_actor(radar_bp, RADAR_TRANSFORM, attach_to=self.ego)
        self.radar_sensor.listen(lambda scan: self.radar_frames.put(scan.frame, bytes(scan.raw_data)))

    @staticmethod
    def _encode_png(image) -> bytes:
        bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        ok, encoded = cv2.imencode(".png", bgra[:, :, :3])
        if not ok:
            raise RuntimeError("Could not encode CARLA RGB image as PNG")
        return encoded.tobytes()

    def _collect_and_send(self, carla_frame: int, snapshot) -> None:
        try:
            image = self.rgb_frames.get(carla_frame, SENSOR_WAIT_SECONDS)
            lidar = self.lidar_frames.get(carla_frame, SENSOR_WAIT_SECONDS)
            radar = self.radar_frames.get(carla_frame, SENSOR_WAIT_SECONDS)
            packet = {
                "sensor_id": SENSOR_ID,
                "frame_id": int(carla_frame),
                "timestamp": float(snapshot.timestamp.elapsed_seconds),
                "ego_speed_mps": self._ego_speed_mps(),
                "image": self._encode_png(image),
                "lidar": lidar,
                "radar": radar,
            }
            if self.uploader and not self.uploader.submit(packet):
                print(f"[M2][WARN] frame {carla_frame} could not be queued", flush=True)
        except TimeoutError as exc:
            print(f"[SENSOR][WARN] {exc}", flush=True)

    def _ego_speed_mps(self) -> float:
        velocity = self.ego.get_velocity()
        return float(math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))

    def _update_spectator(self) -> None:
        ego_tf = self.ego.get_transform()
        forward = ego_tf.get_forward_vector()
        up = ego_tf.get_up_vector()
        cam_location = ego_tf.location - forward * 8.0 + up * 4.0  # 8m behind, 4m above
        cam_rotation = carla.Rotation(pitch=-15.0, yaw=ego_tf.rotation.yaw)
        self.world.get_spectator().set_transform(carla.Transform(cam_location, cam_rotation))

    def step(self) -> None:
        carla_frame = self.world.tick()
        self.frame_count += 1
        self._update_spectator()
        snapshot = self.world.get_snapshot()
        ego_loc = self.ego.get_location()
        self.trajectory_log.append({"frame": carla_frame, "t": snapshot.timestamp.elapsed_seconds, "x": ego_loc.x, "y": ego_loc.y, "z": ego_loc.z})
        self._collect_and_send(carla_frame, snapshot)
        if self.pedestrian and not self.pedestrian_triggered and ego_loc.distance(self.pedestrian.get_location()) < PEDESTRIAN_TRIGGER_DISTANCE_M:
            self._trigger_pedestrian()
        if self.cattle and not self.cattle_triggered and ego_loc.distance(self.cattle.get_location()) < CATTLE_TRIGGER_DISTANCE_M:
            self._trigger_cattle()
        if self.pedestrian_triggered:
            self._update_pedestrian_kinematic()
        if self.cattle_triggered:
            self._update_cattle_kinematic()
        self._check_scripted_collisions()

    def finish(self) -> None:
        LOG_DIR.mkdir(exist_ok=True)
        with (LOG_DIR / "scenario1_trajectory.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "t", "x", "y", "z"]); writer.writeheader(); writer.writerows(self.trajectory_log)
        if self.uploader:
            self.uploader.close()
            print(f"[M2] final sent={self.uploader.sent} failed={self.uploader.failed} dropped={self.uploader.dropped}", flush=True)
        if self.world:
            try:
                settings = self.world.get_settings(); settings.synchronous_mode = False; self.world.apply_settings(settings)
                if self.tm is not None:
                    self.tm.set_synchronous_mode(False)
            except RuntimeError:
                pass
        for actor in [self.rgb_camera, self.lidar_sensor, self.radar_sensor, self.collision_sensor, self.ego, self.bike, self.pedestrian, self.pushcart, self.cattle]:
            if actor:
                try:
                    if hasattr(actor, "stop"): actor.stop()
                    actor.destroy()
                except RuntimeError:
                    pass

    def _on_collision(self, event) -> None:
        key = (int(event.frame), int(event.other_actor.id))
        if key not in self._collision_keys:
            self._collision_keys.add(key); self.collision_events.append({"frame": int(event.frame), "other_actor": event.other_actor.type_id})

    def _check_scripted_collisions(self) -> None:
        ego_loc = self.ego.get_location()
        for label, actor, radius in (("pedestrian", self.pedestrian, 2.4), ("cattle", self.cattle, 2.8), ("bike", self.bike, 2.5), ("pushcart", self.pushcart, 2.8)):
            if actor and actor.is_alive and ego_loc.distance(actor.get_location()) <= radius:
                key = (self.frame_count, actor.id)
                if key not in self._collision_keys:
                    self._collision_keys.add(key); self.collision_events.append({"frame": self.frame_count, "other_actor": f"{actor.id}:{label}:proximity"})

    def _forward_waypoint(self, transform, distance):
        wp = self.map.get_waypoint(transform.location); travelled = 0.0
        while travelled < distance:
            nxt = wp.next(2.0)
            if not nxt: break
            wp = nxt[0]; travelled += 2.0
        return wp

    def _spawn_bike(self, ahead_wp):
        bp = self.world.get_blueprint_library().find("vehicle.kawasaki.ninja")
        wp = ahead_wp
        for _ in range(3):
            nxt = wp.next(5.0)
            if not nxt: break
            wp = nxt[0]
        tf = wp.transform; tf.location.z += 0.3; tf.rotation.yaw = (tf.rotation.yaw + 180.0) % 360.0
        bp.set_attribute("role_name", "oncoming_bike")
        return self.world.try_spawn_actor(bp, tf)

    def _spawn_pedestrian(self, ahead_wp):
        bps = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        if not bps: return None
        wp = ahead_wp
        for _ in range(7):
            nxt = wp.next(5.0)
            if not nxt: break
            wp = nxt[0]
        tf = wp.transform; right = tf.get_right_vector(); tf.location.x += right.x * 4.0; tf.location.y += right.y * 4.0; tf.location.z += 0.5
        return self.world.try_spawn_actor(random.choice(bps), tf)

    def _spawn_pushcart(self, ahead_wp):
        bp = self.world.get_blueprint_library().find("vehicle.micro.microlino")
        wp = ahead_wp; travelled = 0.0
        while travelled < 25.0:
            nxt = wp.next(2.0)
            if not nxt: break
            wp = nxt[0]; travelled += 2.0
        tf = wp.transform; right = tf.get_right_vector(); tf.location.x += right.x * 4.5; tf.location.y += right.y * 4.5; tf.location.z += 0.1
        cart = self.world.try_spawn_actor(bp, tf)
        if cart: cart.set_simulate_physics(False)
        return cart

    def _spawn_cattle(self, ahead_wp):
        return self._spawn_pedestrian_at_distance(ahead_wp, 40.0, 8.0)

    def _spawn_pedestrian_at_distance(self, ahead_wp, distance, offset):
        bps = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        if not bps: return None
        wp = ahead_wp; travelled = 0.0
        while travelled < distance:
            nxt = wp.next(2.0)
            if not nxt: break
            wp = nxt[0]; travelled += 2.0
        tf = wp.transform; right = tf.get_right_vector(); tf.location.x += right.x * offset; tf.location.y += right.y * offset; tf.location.z += 0.5
        return self.world.try_spawn_actor(random.choice(bps), tf)

    @staticmethod
    def _copy_location(loc): return carla.Location(float(loc.x), float(loc.y), float(loc.z))

    def _trigger_pedestrian(self):
        loc = self._copy_location(self.pedestrian.get_location()); wp = self.map.get_waypoint(loc); axis = wp.transform.get_right_vector(); center = wp.transform.location; sign = 1.0 if (center.x-loc.x)*axis.x + (center.y-loc.y)*axis.y >= 0 else -1.0
        self.pedestrian_start_loc = loc; self.pedestrian_target_loc = carla.Location(loc.x + sign*axis.x*10, loc.y + sign*axis.y*10, loc.z); self.pedestrian_crossing_progress = 0.0; self.pedestrian_triggered = True

    def _update_pedestrian_kinematic(self):
        if not self.pedestrian or not self.pedestrian_start_loc or not self.pedestrian_target_loc: return
        self.pedestrian_crossing_progress = min(1.0, self.pedestrian_crossing_progress + FIXED_DELTA_SECONDS / PEDESTRIAN_CROSSING_TIME_S); p = self.pedestrian_crossing_progress; a, b = self.pedestrian_start_loc, self.pedestrian_target_loc
        self.pedestrian.set_transform(carla.Transform(carla.Location(a.x+(b.x-a.x)*p, a.y+(b.y-a.y)*p, a.z), self.pedestrian_rotation))

    def _trigger_cattle(self):
        loc = self._copy_location(self.cattle.get_location()); wp = self.map.get_waypoint(loc); axis = wp.transform.get_right_vector(); center = wp.transform.location; sign = 1.0 if (center.x-loc.x)*axis.x + (center.y-loc.y)*axis.y >= 0 else -1.0
        self.cattle_start_loc = loc; self.cattle_target_loc = carla.Location(loc.x + sign*axis.x*10, loc.y + sign*axis.y*10, loc.z); self.cattle_crossing_progress = 0.0; self.cattle_triggered = True

    def _update_cattle_kinematic(self):
        if not self.cattle or not self.cattle_start_loc or not self.cattle_target_loc: return
        self.cattle_crossing_progress = min(1.0, self.cattle_crossing_progress + FIXED_DELTA_SECONDS / CATTLE_CROSSING_TIME_S); p = self.cattle_crossing_progress; a, b = self.cattle_start_loc, self.cattle_target_loc
        self.cattle.set_transform(carla.Transform(carla.Location(a.x+(b.x-a.x)*p, a.y+(b.y-a.y)*p, a.z), self.cattle_rotation))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--ego-spawn-index", type=int, default=35)
    parser.add_argument("--duration", type=float, default=90.0)
    parser.add_argument("--m2-url", default=M2_URL)
    parser.add_argument("--tm-port", type=int, default=TRAFFIC_MANAGER_PORT)
    args = parser.parse_args()
    scenario = Scenario1VillageRoad(args.host, args.port, args.ego_spawn_index, args.m2_url, args.tm_port)
    try:
        scenario.setup()
        scenario.ego.set_autopilot(True, scenario.tm.get_port())
        scenario.tm.vehicle_percentage_speed_difference(scenario.ego, int(EGO_SPEED_MULTIPLIER))
        for _ in range(int(args.duration / FIXED_DELTA_SECONDS)):
            scenario.step()
    except KeyboardInterrupt:
        print("[MAIN] Interrupted")
    finally:
        scenario.finish()


if __name__ == "__main__":
    main()
