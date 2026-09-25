"""Shared CARLA scenario infrastructure for M1's scenario scripts.

Extracted from scenario3.py's proven patterns (FrameBuffer/Uploader sensor
pipeline, walker/controller usage, sensor mounting, ground-truth printing)
so every scenario in this folder uses IDENTICAL sensor transforms and
upload contract. This matters beyond style: M3_Pipeline/config.py hardcodes
these same CAMERA_TRANSFORM/LIDAR_TRANSFORM/RADAR_TRANSFORM values and the
1280x720/90-FOV camera attributes to compute 3D positions -- any scenario
that changes them without updating M3's config.py would silently break its
math. scenario3.py itself is left untouched (already validated across
several live runs); it does not import from here, and new scenario_*.py
files should import from here instead of re-deriving this boilerplate.
"""
from __future__ import annotations

import json
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

DT = 0.05
M2_URL = "http://127.0.0.1:8000/api/v1/perception"
TM_PORT = 8001
LOG_DIR = Path("logs")
SENSOR_WAIT_S = 0.5
UPLOAD_QUEUE_SIZE = 4
WALKER_TARGET_RETRIES = 20
WALKER_MAX_SPEED_MPS = 1.35

# Must match M3_Pipeline/config.py's Transform constants exactly -- see the
# module docstring above.
CAMERA_TRANSFORM = carla.Transform(carla.Location(x=1.5, z=2.2), carla.Rotation(pitch=-5.0))
LIDAR_TRANSFORM = carla.Transform(carla.Location(z=2.3))
RADAR_TRANSFORM = carla.Transform(carla.Location(x=2.0, z=1.2))

CARS = [
    "vehicle.carlamotors.carlacola", "vehicle.volkswagen.t2",
    "vehicle.mini.cooper_s", "vehicle.nissan.micra",
    "vehicle.toyota.prius", "vehicle.audi.a2",
]
BIKES = [
    "vehicle.vespa.zx125", "vehicle.yamaha.yzf", "vehicle.kawasaki.ninja",
    "vehicle.harley-davidson.low_rider", "vehicle.diamondback.century",
    "vehicle.bh.crossbike", "vehicle.gazelle.omafiets",
]
# Pure human-pedalled bicycles only (subset of BIKES) -- for scenarios that
# want to spawn something M2/classes.yaml's "bicycle" class specifically,
# as opposed to a motorized two-wheeler ("motorcycle").
BICYCLES = ["vehicle.diamondback.century", "vehicle.bh.crossbike", "vehicle.gazelle.omafiets"]
# Motorized two-wheelers only (subset of BIKES) -- the complementary half of
# BICYCLES above, for scenarios that want classes.yaml's "motorcycle"
# specifically rather than a human-pedalled bike.
MOTORCYCLES = ["vehicle.vespa.zx125", "vehicle.yamaha.yzf", "vehicle.kawasaki.ninja", "vehicle.harley-davidson.low_rider"]
TRUCKS = [
    "vehicle.carlamotors.carlacola", "vehicle.volkswagen.t2",
    "vehicle.carlamotors.european_hgv",  # only genuine heavy-truck mesh in the 0.9.15 catalog
]
# classes.yaml's "bus" and "tempo" (light commercial van/load-carrier) had no
# spawns anywhere in this folder before -- confirmed against a live query of
# world.get_blueprint_library().filter("vehicle.*") that these are the only
# matching meshes in the installed 0.9.15 catalog.
BUSES = ["vehicle.mitsubishi.fusorosa"]
TEMPOS = ["vehicle.mercedes.sprinter"]
# CARLA 0.9.15 ships no auto-rickshaw asset. This is the closest available
# stand-in (small, boxy, low top speed) -- flagged here rather than silently
# labelling a car/bike as a rickshaw in scenario output.
RICKSHAW_STANDIN = "vehicle.micro.microlino"

# classes.yaml's "traffic_cones" and "road_sign" -- confirmed against a live
# world.get_blueprint_library().filter("static.prop.*<pattern>*") query.
TRAFFIC_CONE_PROPS = ["static.prop.trafficcone01", "static.prop.trafficcone02", "static.prop.constructioncone"]
ROAD_SIGN_PROPS = ["static.prop.streetsign", "static.prop.streetsign01", "static.prop.streetsign04"]
# classes.yaml's "speed_bumps" has NO matching asset anywhere in the 0.9.15
# static.prop.* catalog (checked live: no "bump"/"hump"/"speed" match at
# all) -- unlike RICKSHAW_STANDIN there is no visually honest stand-in for a
# low road hump, so it's left unexercised here rather than mislabelling
# something else as a speed bump.


class FrameBuffer:
    def __init__(self, name: str, max_frames: int = 32):
        self.name = name
        self.frames: dict[int, Any] = {}
        self.cv = threading.Condition()
        self.max_frames = max_frames

    def put(self, frame: int, data: Any) -> None:
        with self.cv:
            self.frames[int(frame)] = data
            while len(self.frames) > self.max_frames:
                self.frames.pop(min(self.frames))
            self.cv.notify_all()

    def get(self, frame: int, timeout: float) -> Any:
        deadline = time.monotonic() + timeout
        with self.cv:
            while int(frame) not in self.frames:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"{self.name} frame {frame} not received")
                self.cv.wait(remaining)
            return self.frames.pop(int(frame))


class Uploader:
    def __init__(self, url: str, sensor_id: str):
        self.url = url
        self.sensor_id = sensor_id
        self.q: queue.Queue[dict[str, Any] | None] = queue.Queue(UPLOAD_QUEUE_SIZE)
        self.stop_event = threading.Event()
        self.sent = self.failed = self.dropped = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, packet: dict[str, Any]) -> None:
        try:
            self.q.put_nowait(packet)
        except queue.Full:
            try:
                self.q.get_nowait()
                self.dropped += 1
                self.q.put_nowait(packet)
            except queue.Empty:
                self.dropped += 1

    def _run(self) -> None:
        with httpx.Client(timeout=5.0) as client:
            while not self.stop_event.is_set():
                try:
                    packet = self.q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if packet is None:
                    return
                frame = packet["frame"]
                files = {
                    "image": (f"rgb_{frame}.png", packet["image"], "image/png"),
                    "lidar_file": (f"lidar_{frame}.bin", packet["lidar"], "application/octet-stream"),
                    "radar_file": (f"radar_{frame}.bin", packet["radar"], "application/octet-stream"),
                }
                data = {
                    "sensor_id": self.sensor_id,
                    "frame_id": str(frame),
                    "timestamp": repr(packet["timestamp"]),
                    "ego_speed_mps": repr(packet["speed"]),
                    "ego_position": json.dumps(packet["ego_position"]),
                    "ego_velocity": json.dumps(packet["ego_velocity"]),
                    "ego_yaw_deg": repr(packet["ego_yaw_deg"]),
                }
                try:
                    client.post(self.url, data=data, files=files).raise_for_status()
                    self.sent += 1
                except (httpx.HTTPError, OSError) as exc:
                    self.failed += 1
                    print(f"[M2][WARN] frame={frame}: {exc}", flush=True)

    def close(self) -> None:
        self.stop_event.set()
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=6.0)


def png_bytes(image) -> bytes:
    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    ok, enc = cv2.imencode(".png", bgra[:, :, :3])
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return enc.tobytes()


def actor_speed(actor) -> float:
    v = actor.get_velocity()
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def connect(host: str, port: int, town: str, tm_port: int):
    """Load `town` in synchronous mode and start a synchronous Traffic
    Manager. Uses a longer client timeout than scenario3.py's original 10s
    -- live testing found a cold Town05 load can take ~30s on a modest GPU
    (RTX 3050 6GB) and 10s timed out client-side even though the server was
    still loading successfully."""
    client = carla.Client(host, port)
    client.set_timeout(30.0)
    world = client.load_world(town)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = DT
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = 5
    world.apply_settings(settings)
    tm = client.get_trafficmanager(tm_port)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(8.0)
    return client, world, tm


def teardown(world, tm, actors: list, controllers: list | None = None) -> None:
    """Stop+destroy actors and hand the world/TM back to asynchronous mode.
    Sensors are stopped (not just destroyed) first -- destroying a listening
    sensor outright is a known source of the client hanging on shutdown that
    scenario3.py's own runs have hit."""
    for controller in controllers or []:
        try:
            if controller.is_alive:
                controller.stop()
                controller.destroy()
        except RuntimeError:
            pass
    for actor in actors:
        try:
            if actor and actor.is_alive:
                if hasattr(actor, "stop") and actor.type_id.startswith("sensor."):
                    actor.stop()
                actor.destroy()
        except RuntimeError:
            pass
    if world:
        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
            tm.set_synchronous_mode(False)
        except RuntimeError:
            pass


def setup_ego_sensors(world, ego, rgb_buf: FrameBuffer, lidar_buf: FrameBuffer, radar_buf: FrameBuffer):
    """Attach camera/LiDAR/radar/collision sensors to `ego` with the exact
    mounts and attributes M3_Pipeline/config.py assumes. Returns
    (rgb, lidar, radar, collision_sensor)."""
    bp = world.get_blueprint_library()

    cam = bp.find("sensor.camera.rgb")
    cam.set_attribute("image_size_x", "1280")
    cam.set_attribute("image_size_y", "720")
    # Tried widening to 150 deg to cover more of LiDAR's 360 deg field (70%
    # of LiDAR-tracked objects sat outside the 90 deg cone). Controlled A/B
    # on village_road (same seed, same measurement) showed no real benefit:
    # unknown-classification fraction barely moved (97.4% at 90 vs 96.3% at
    # 150) and raw YOLO detections/frame actually got WORSE (mean 2.44->1.41,
    # 0%->20% zero-detection frames) -- objects appear smaller on the same
    # 1280px sensor at wider FOV, and that pixel-density loss outweighs the
    # coverage gain. Reverted to 90 deg. The real bottleneck is a sensor
    # sampling-density mismatch (M3 tracks 65-85 objects/frame from
    # LiDAR/radar vs M2 classifying only 1-2/frame from one camera request),
    # not camera coverage. M3_Pipeline/config.py's CAMERA_FOV_DEG MUST
    # mirror this.
    cam.set_attribute("fov", "90")
    cam.set_attribute("sensor_tick", str(DT))
    rgb = world.spawn_actor(cam, CAMERA_TRANSFORM, attach_to=ego)
    rgb.listen(lambda x: rgb_buf.put(x.frame, x))

    lidar = bp.find("sensor.lidar.ray_cast")
    lidar.set_attribute("channels", "32")
    lidar.set_attribute("range", "80")
    lidar.set_attribute("points_per_second", "640000")
    lidar.set_attribute("rotation_frequency", str(1.0 / DT))
    lidar.set_attribute("upper_fov", "10")
    lidar.set_attribute("lower_fov", "-30")
    lidar.set_attribute("sensor_tick", str(DT))
    lidar_actor = world.spawn_actor(lidar, LIDAR_TRANSFORM, attach_to=ego)
    lidar_actor.listen(lambda x: lidar_buf.put(x.frame, bytes(x.raw_data)))

    radar = bp.find("sensor.other.radar")
    radar.set_attribute("horizontal_fov", "35")
    radar.set_attribute("vertical_fov", "20")
    radar.set_attribute("range", "80")
    radar.set_attribute("points_per_second", "1500")
    radar.set_attribute("sensor_tick", str(DT))
    radar_actor = world.spawn_actor(radar, RADAR_TRANSFORM, attach_to=ego)
    radar_actor.listen(lambda x: radar_buf.put(x.frame, bytes(x.raw_data)))

    cbp = bp.find("sensor.other.collision")
    collision = world.spawn_actor(cbp, carla.Transform(), attach_to=ego)

    return rgb, lidar_actor, radar_actor, collision


def send_sensors(uploader: Uploader, rgb_buf: FrameBuffer, lidar_buf: FrameBuffer,
                  radar_buf: FrameBuffer, frame: int, ego, snap) -> None:
    try:
        image = rgb_buf.get(frame, SENSOR_WAIT_S)
        lidar = lidar_buf.get(frame, SENSOR_WAIT_S)
        radar = radar_buf.get(frame, SENSOR_WAIT_S)
        ego_tf = ego.get_transform()
        ego_velocity = ego.get_velocity()
        uploader.submit({
            "frame": frame,
            "timestamp": snap.timestamp.elapsed_seconds,
            "speed": actor_speed(ego),
            "ego_position": {"x": ego_tf.location.x, "y": ego_tf.location.y, "z": ego_tf.location.z},
            "ego_velocity": {"x": ego_velocity.x, "y": ego_velocity.y, "z": ego_velocity.z},
            "ego_yaw_deg": ego_tf.rotation.yaw,
            "image": png_bytes(image),
            "lidar": lidar,
            "radar": radar,
        })
    except TimeoutError as exc:
        print(f"[SENSOR][WARN] {exc}", flush=True)


def spawn_roadside_props(world, bp, ego_wp, blueprint_names: list[str], ahead_list: list[float],
                          side_m: float = 2.5, jitter_m: float = 1.5) -> list[Any]:
    """Spawn one static prop per distance in `ahead_list`, alternating sides
    of the road, picking a random blueprint from `blueprint_names` each time.
    Used for traffic cones / road signs -- purely cosmetic roadside clutter,
    static (physics disabled) and never updated after spawn, same pattern as
    each scenario's existing `_spawn_potholes`/`_spawn_handcarts`. Returns
    the list of successfully spawned actors (silently skips any distance
    that has no waypoint or fails to spawn, same as the rest of this file)."""
    props: list[Any] = []
    for i, ahead in enumerate(ahead_list):
        route = ego_wp.next(ahead)
        if not route:
            continue
        wp = route[0]
        right = wp.transform.get_right_vector()
        side = (-1 if i % 2 == 0 else 1) * (side_m + random.uniform(-jitter_m, jitter_m))
        tf = carla.Transform(wp.transform.location, wp.transform.rotation)
        tf.location.x += right.x * side
        tf.location.y += right.y * side
        tf.location.z += 0.1
        tf.rotation.yaw += random.uniform(-10.0, 10.0)
        name = random.choice(blueprint_names)
        try:
            prop_bp = bp.find(name)
        except IndexError:
            continue
        prop = world.try_spawn_actor(prop_bp, tf)
        if prop:
            prop.set_simulate_physics(False)
            props.append(prop)
    return props


def write_obstacle_manifest(path: "Path", entries: list[dict[str, Any]], note: str = "") -> None:
    """Write a one-shot JSON list of every deterministically-placed obstacle
    a scenario spawns (potholes, cones, signs, handcarts, scripted walkers,
    the merger/stalled vehicle, etc.) -- a human reference for visually
    checking "did M5 correctly navigate around obstacle X at (x, y)"
    against the CARLA window, since `print_ground_truth`'s per-tick stream
    never covers static props and is too noisy to scan by eye. Called once
    at the end of `setup()`, not per-tick, unlike print_ground_truth.
    Deliberately excludes continuous/ambient spawners (jaywalkers, racers)
    since their count/position isn't deterministic run-to-run even under a
    fixed RANDOM_SEED -- `note` should say so explicitly for the scenarios
    that have any, so a reader doesn't expect the manifest to be exhaustive."""
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"note": note, "obstacles": entries}, indent=2), encoding="utf-8")


def walker_nav_target(world, origin: "carla.Location", preferred: "carla.Location",
                       min_dist: float = 5.0, max_dist: float = 35.0) -> "carla.Location":
    """Return a reachable controller.ai.walker navigation-mesh target as
    close to `preferred` as retries allow.

    Live testing (Sep 2026) found Town07 has no baked pedestrian navigation
    mesh in this CARLA 0.9.15 install: world.get_random_location_from_
    navigation() silently serves a DIFFERENT map's cached nav-mesh instead
    of erroring (confirmed: CARLA's own log prints "Found the required file
    in cache! Carla/Maps/Nav/Town10HD_Opt.bin" while Town07 is the actually
    loaded world), so every candidate lands wildly outside [min_dist,
    max_dist] of `origin` and this used to return None every time --
    every walker/cattle actor in Town07 was destroyed immediately after
    spawning (every caller's `if target is None: walker.destroy()`), so
    none ever actually appeared. Falls back to `preferred` directly (the
    caller's own waypoint-relative offset, already a reasonable target) if
    no in-range navmesh candidate is found after all retries, rather than
    returning None -- worse path-following fidelity than a real validated
    navmesh point on a town where the navmesh DOES work, but the walker
    exists and moves, instead of not existing at all."""
    best, best_error = None, float("inf")
    for _ in range(WALKER_TARGET_RETRIES):
        candidate = world.get_random_location_from_navigation()
        if candidate is None:
            continue
        distance = origin.distance(candidate)
        error = candidate.distance(preferred)
        if min_dist <= distance <= max_dist and error < best_error:
            best, best_error = candidate, error
    return best if best is not None else preferred


def is_behind(ego, loc: "carla.Location") -> bool:
    if not ego or not ego.is_alive:
        return False
    tf = ego.get_transform()
    forward = tf.get_forward_vector()
    to_actor = loc - tf.location
    return (forward.x * to_actor.x + forward.y * to_actor.y) < 0.0


def print_ground_truth(frame: int, ego, groups: dict[str, list]) -> None:
    """Print CARLA ground truth in the exact format M3_Pipeline/evaluation.py's
    regex expects (`KIND actor=ID ... pos=(x,y) ... vel=(vx,vy) ...`), so
    every scenario's stdout log can be fed through the same evaluation
    script. `groups` maps a label (e.g. "WALKER", "CATTLE", "TRUCK") to its
    list of actors; "EGO" is printed separately and skipped by evaluation.py
    on purpose (it only scores non-ego objects)."""
    print(f"\n[GROUND TRUTH][frame {frame}]", flush=True)
    if ego and ego.is_alive:
        loc = ego.get_location()
        vel = ego.get_velocity()
        print(
            f"  EGO actor={ego.id} pos=({loc.x:.2f},{loc.y:.2f}) "
            f"vel=({vel.x:.2f},{vel.y:.2f}) speed={actor_speed(ego):.2f} m/s",
            flush=True,
        )
    for label, actors in groups.items():
        for actor in actors:
            if actor and actor.is_alive:
                loc = actor.get_location()
                vel = actor.get_velocity()
                print(
                    f"  {label} actor={actor.id} type={actor.type_id} "
                    f"pos=({loc.x:.2f},{loc.y:.2f}) vel=({vel.x:.2f},{vel.y:.2f}) "
                    f"speed={actor_speed(actor):.2f} m/s",
                    flush=True,
                )
