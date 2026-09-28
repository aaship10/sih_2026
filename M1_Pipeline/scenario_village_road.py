#!/usr/bin/env python3
"""SIH Scenario: unmarked village road for CARLA 0.9.15 (Town07).

Low traffic volume, high per-encounter unpredictability: pedestrians walk
*along* the road itself (no footpath), sometimes crossing at will; bikes and
motorcycles overtake and occasionally ride straight toward the ego since
there's no marked lane to keep them off it; cattle graze onto the road
unhurried; potholes/debris are the default road surface, not an occasional
hazard; and a few static handcarts sit at the shoulder. Town07 is CARLA's
own rural map -- narrow, largely unmarked roads, almost no traffic lights --
so no custom map/road painting is needed to get an "unmarked road" scene.

Uses the same CARLA-physics-plus-Traffic-Manager approach as scenario3.py
(no set_transform() on moving actors) and the same sensor/upload pipeline,
factored into common.py so every scenario in this folder produces data M3
can consume identically.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from typing import Any

import carla

import common

TOWN = "Town07"
SENSOR_ID = "carla-village-road-ego"
RANDOM_SEED = 3001
SCENARIO_WARMUP_S = 3.0

WALKER_COUNT = 10
WALKER_AHEAD_MIN_M, WALKER_AHEAD_MAX_M = 10.0, 80.0
WALKER_SIDE_MIN_M, WALKER_SIDE_MAX_M = 0.8, 2.2
WALKER_ALONG_MIN_M, WALKER_ALONG_MAX_M = 10.0, 25.0
WALKER_CROSSING_CHANCE = 0.3  # rest walk along the shoulder instead
WALKER_TRIGGER_DISTANCE_M = 14.0
WALKER_MAX_SPEED_MPS = common.WALKER_MAX_SPEED_MPS

CATTLE_GROUP_COUNT = 2
CATTLE_GROUP_SIZE = 3
CATTLE_AHEAD_M = [35.0, 100.0]
CATTLE_TRIGGER_DISTANCE_M = 10.0
CATTLE_MAX_SPEED_MPS = 0.3  # grazing pace, slower than scenario3.py's cattle

ONCOMING_COUNT = 5
ONCOMING_AHEAD_MIN_M, ONCOMING_AHEAD_MAX_M = 45.0, 130.0
ONCOMING_SPEED_DIFFERENCE = 10.0  # % slower than the (low) speed limit -- villages aren't fast

# Live testing (Sep 2026) found this scenario's M5-driven ego crawls at a
# ~1.4 m/s mean speed (YIELD/EMERGENCY_BRAKE dominant) -- over a 30s default
# --duration that's only ~40m of real travel. The original 160m search
# radius / 110m-150m "ahead" distances below placed most obstacles well
# past where a typical M5-driven run ever reaches, making them invisible to
# someone watching the run for visual verification. Tightened to keep every
# obstacle within realistic reach (same COUNTs -- not an intensity change).
POTHOLE_COUNT = 40
POTHOLE_SEARCH_RADIUS_M = 90.0

HANDCART_COUNT = 3
HANDCART_AHEAD_M = [15.0, 40.0, 70.0]

# Live CARLA testing (Sep 2026) found this scenario's obstacle count/variety
# was the lowest of the four new scenarios relative to classes.yaml's 14-class
# taxonomy -- no car/bus/tempo ever appeared, and no traffic_cones/road_sign
# at all. These additions round that out: a few genuinely parked/slow local
# vehicles (not just oncoming bikes) plus roadside cones/signs, matching the
# scenario's own "potholes/debris are the default road surface" density intent.
LOCAL_VEHICLE_AHEAD_M = [15.0, 45.0, 75.0]  # car, bus, tempo -- one of each
TRAFFIC_CONE_AHEAD_M = [6.0, 15.0, 24.0, 33.0, 42.0, 51.0, 60.0, 69.0, 78.0, 87.0]
ROAD_SIGN_AHEAD_M = [8.0, 22.0, 38.0, 54.0, 70.0, 86.0]

DT = common.DT


class VillageRoadScenario:
    def __init__(self, host: str, port: int, ego_index: int, m2_url: str, tm_port: int):
        self.host, self.port, self.ego_index = host, port, ego_index
        self.m2_url, self.tm_port = m2_url, tm_port
        self.client = self.world = self.map = self.tm = self.bp = None
        self.ego = self.collision_sensor = None
        self.rgb = self.lidar = self.radar = None
        self.rgb_buf = common.FrameBuffer("RGB")
        self.lidar_buf = common.FrameBuffer("LiDAR")
        self.radar_buf = common.FrameBuffer("Radar")
        self.uploader: common.Uploader | None = None
        self.ego_wp = None
        self.walkers: list[dict[str, Any]] = []
        self.cattle: list[dict[str, Any]] = []
        self.oncoming: list[Any] = []
        self.local_vehicles: list[Any] = []
        self.static_props: list[Any] = []
        self.traffic_cones: list[Any] = []
        self.road_signs: list[Any] = []
        self.controllers: list[Any] = []
        self.sim_time = 0.0
        self.frame_count = 0
        self.collisions: list[dict[str, Any]] = []
        self.trajectory: list[dict[str, Any]] = []

    def setup(self) -> None:
        random.seed(RANDOM_SEED)
        self.client, self.world, self.tm = common.connect(self.host, self.port, TOWN, self.tm_port)
        self.map = self.world.get_map()
        self.bp = self.world.get_blueprint_library()
        spawns = self.map.get_spawn_points()
        if not 0 <= self.ego_index < len(spawns):
            raise ValueError(f"Invalid ego spawn index {self.ego_index}; available={len(spawns)}")

        ego_bp = self.bp.find("vehicle.tesla.model3")
        ego_bp.set_attribute("role_name", "ego_vehicle")
        self.ego = self.world.spawn_actor(ego_bp, spawns[self.ego_index])
        self.rgb, self.lidar, self.radar, self.collision_sensor = common.setup_ego_sensors(
            self.world, self.ego, self.rgb_buf, self.lidar_buf, self.radar_buf
        )
        self.collision_sensor.listen(self._on_collision)

        self.ego_wp = self.map.get_waypoint(
            spawns[self.ego_index].location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if self.ego_wp is None:
            raise RuntimeError("Ego spawn point is not on a drivable lane; pick another --ego-spawn-index")

        self._spawn_walkers()
        self._spawn_cattle()
        self._spawn_oncoming()
        self._spawn_local_vehicles()
        self._spawn_potholes()
        self._spawn_handcarts()
        self._spawn_traffic_cones()
        self._spawn_road_signs()
        # CARLA synchronous mode: a freshly try_spawn_actor'd actor's
        # get_location() reads back a default (0,0,0) transform until the
        # world has ticked at least once -- this tick lets every spawn above
        # sync before _write_obstacle_manifest() below queries positions.
        self.world.tick()
        self.uploader = common.Uploader(self.m2_url, SENSOR_ID)
        print(
            f"[SETUP] village road: walkers={len(self.walkers)} cattle={len(self.cattle)} "
            f"oncoming={len(self.oncoming)} local_vehicles={len(self.local_vehicles)} "
            f"potholes={len(self.static_props)} cones={len(self.traffic_cones)} signs={len(self.road_signs)}",
            flush=True,
        )
        self._write_obstacle_manifest()

    def _write_obstacle_manifest(self) -> None:
        entries: list[dict[str, Any]] = []
        for prop in self.static_props:
            kind = "handcart" if prop.type_id == common.RICKSHAW_STANDIN else "pothole_or_debris"
            loc = prop.get_location()
            entries.append({"kind": kind, "blueprint": prop.type_id, "x": loc.x, "y": loc.y, "z": loc.z})
        for w in self.walkers:
            loc = w["actor"].get_location()
            entries.append({"kind": "walker", "blueprint": w["actor"].type_id, "x": loc.x, "y": loc.y, "z": loc.z,
                             "note": "crossing" if w["crossing"] else "along shoulder"})
        for c in self.cattle:
            loc = c["actor"].get_location()
            entries.append({"kind": "cattle", "blueprint": c["actor"].type_id, "x": loc.x, "y": loc.y, "z": loc.z})
        for actor in self.oncoming:
            loc = actor.get_location()
            entries.append({"kind": "oncoming_vehicle", "blueprint": actor.type_id, "x": loc.x, "y": loc.y, "z": loc.z,
                             "note": "spawn position -- moving toward the ego"})
        for actor in self.local_vehicles:
            loc = actor.get_location()
            entries.append({"kind": "local_vehicle", "blueprint": actor.type_id, "x": loc.x, "y": loc.y, "z": loc.z})
        common.write_obstacle_manifest(
            common.LOG_DIR / "obstacle_manifest_village_road.json", entries,
            note="All obstacles here are deterministically placed under RANDOM_SEED -- this scenario has no "
                 "continuous/ambient spawner. oncoming_vehicle/local_vehicle positions are spawn points only, "
                 "they keep moving afterward via the Traffic Manager.",
        )

    def _spawn_local_vehicles(self) -> None:
        """A parked/slow car, bus and tempo along the route -- previously
        this scenario had zero car/bus/tempo presence at all (only oncoming
        bikes and a handcart stand-in), leaving three of classes.yaml's 14
        classes completely unexercised here."""
        choices = [common.CARS, common.BUSES, common.TEMPOS]
        for ahead, names in zip(LOCAL_VEHICLE_AHEAD_M, choices):
            route = self.ego_wp.next(ahead)
            if not route:
                continue
            wp = route[0]
            tf = carla.Transform(wp.transform.location, wp.transform.rotation)
            right = tf.get_right_vector()
            side = random.choice([-1, 1]) * random.uniform(2.0, 3.0)
            tf.location.x += right.x * side
            tf.location.y += right.y * side
            tf.location.z += 0.3
            actor = self.world.try_spawn_actor(self.bp.find(random.choice(names)), tf)
            if actor is None:
                continue
            actor.set_simulate_physics(True)
            actor.set_enable_gravity(True)
            actor.set_autopilot(True, self.tm_port)
            self.tm.ignore_lights_percentage(actor, 100.0)
            self.tm.ignore_signs_percentage(actor, 100.0)
            self.tm.vehicle_percentage_speed_difference(actor, 30.0)
            self.local_vehicles.append(actor)

    def _spawn_traffic_cones(self) -> None:
        # Live testing found side_m=1.6 with spawn_roadside_props' default
        # jitter_m=1.5 could offset a cone as little as 0.1m from centerline
        # (side_m - jitter_m) -- combined with the tighter TRAFFIC_CONE_AHEAD_M
        # spacing above (for visibility within a 30s run), this produced 33
        # collision-sensor contacts against one cone the M5-driven ego got
        # wedged against (same fixed-2.5m-offset/no-lane-awareness limitation
        # already documented in M5_Pipeline/README.md). side_m=2.2/jitter_m=0.8
        # keeps a 1.4-3.0m clearance band instead.
        self.traffic_cones = common.spawn_roadside_props(
            self.world, self.bp, self.ego_wp, common.TRAFFIC_CONE_PROPS, TRAFFIC_CONE_AHEAD_M, side_m=2.2, jitter_m=0.8,
        )
        self.static_props += self.traffic_cones

    def _spawn_road_signs(self) -> None:
        self.road_signs = common.spawn_roadside_props(
            self.world, self.bp, self.ego_wp, common.ROAD_SIGN_PROPS, ROAD_SIGN_AHEAD_M, side_m=2.8,
        )
        self.static_props += self.road_signs

    def _spawn_walkers(self) -> None:
        walker_bps = self.bp.filter("walker.pedestrian.*")
        controller_bp = self.bp.find("controller.ai.walker")
        if not walker_bps or controller_bp is None:
            return
        for _ in range(WALKER_COUNT):
            ahead = random.uniform(WALKER_AHEAD_MIN_M, WALKER_AHEAD_MAX_M)
            route = self.ego_wp.next(ahead)
            if not route:
                continue
            wp = route[0]
            right = wp.transform.get_right_vector()
            forward = wp.transform.get_forward_vector()
            side = random.choice([-1, 1]) * random.uniform(WALKER_SIDE_MIN_M, WALKER_SIDE_MAX_M)
            tf = carla.Transform(
                carla.Location(
                    wp.transform.location.x + right.x * side,
                    wp.transform.location.y + right.y * side,
                    wp.transform.location.z + 0.5,
                ),
                wp.transform.rotation,
            )
            walker = self.world.try_spawn_actor(random.choice(walker_bps), tf)
            if walker is None:
                continue

            crossing = random.random() < WALKER_CROSSING_CHANCE
            if crossing:
                preferred = carla.Location(
                    tf.location.x - right.x * side * 2.0, tf.location.y - right.y * side * 2.0, tf.location.z
                )
            else:
                along = random.uniform(WALKER_ALONG_MIN_M, WALKER_ALONG_MAX_M)
                preferred = carla.Location(
                    tf.location.x + forward.x * along + right.x * side,
                    tf.location.y + forward.y * along + right.y * side,
                    tf.location.z,
                )
            target = common.walker_nav_target(self.world, tf.location, preferred)
            if target is None:
                walker.destroy()
                continue
            controller = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
            controller.start()
            controller.go_to_location(target)
            controller.set_max_speed(0.0)
            self.walkers.append({"actor": walker, "controller": controller, "started": False, "crossing": crossing})
            self.controllers.append(controller)

    def _spawn_cattle(self) -> None:
        walker_bps = self.bp.filter("walker.pedestrian.*")
        controller_bp = self.bp.find("controller.ai.walker")
        if not walker_bps or controller_bp is None:
            return
        for ahead in CATTLE_AHEAD_M[:CATTLE_GROUP_COUNT]:
            route = self.ego_wp.next(ahead)
            if not route:
                continue
            wp = route[0]
            right = wp.transform.get_right_vector()
            for _ in range(CATTLE_GROUP_SIZE):
                side = random.choice([-1, 1]) * (2.0 + random.uniform(-0.5, 0.5))
                tf = carla.Transform(
                    carla.Location(
                        wp.transform.location.x + right.x * side + random.uniform(-1, 1),
                        wp.transform.location.y + right.y * side + random.uniform(-1, 1),
                        wp.transform.location.z + 0.5,
                    ),
                    wp.transform.rotation,
                )
                walker = self.world.try_spawn_actor(random.choice(walker_bps), tf)
                if walker is None:
                    continue
                preferred = carla.Location(
                    tf.location.x - right.x * side * 2.5, tf.location.y - right.y * side * 2.5, tf.location.z
                )
                target = common.walker_nav_target(self.world, tf.location, preferred)
                if target is None:
                    walker.destroy()
                    continue
                controller = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
                controller.start()
                controller.go_to_location(target)
                controller.set_max_speed(0.0)
                self.cattle.append({"actor": walker, "controller": controller, "started": False})
                self.controllers.append(controller)

    def _spawn_oncoming(self) -> None:
        for _ in range(ONCOMING_COUNT):
            ahead = random.uniform(ONCOMING_AHEAD_MIN_M, ONCOMING_AHEAD_MAX_M)
            route = self.ego_wp.next(ahead)
            if not route:
                continue
            wp = route[0]
            tf = carla.Transform(wp.transform.location, wp.transform.rotation)
            tf.location.z += 0.3
            tf.rotation.yaw += 180.0  # riding back toward the ego -- no separate lane to hold it to one side
            right = tf.get_right_vector()
            side = random.uniform(-0.8, 0.8)
            tf.location.x += right.x * side
            tf.location.y += right.y * side
            name = random.choice(common.BIKES)
            actor = self.world.try_spawn_actor(self.bp.find(name), tf)
            if actor is None:
                continue
            actor.set_simulate_physics(True)
            actor.set_enable_gravity(True)
            actor.set_autopilot(True, self.tm_port)
            self.tm.ignore_lights_percentage(actor, 100.0)
            self.tm.ignore_signs_percentage(actor, 100.0)
            self.tm.vehicle_percentage_speed_difference(actor, ONCOMING_SPEED_DIFFERENCE)
            self.oncoming.append(actor)
            print(f"[SETUP] ONCOMING actor={actor.id} type={name} ahead={ahead:.1f}m", flush=True)

    def _spawn_potholes(self) -> None:
        pothole_bps = []
        for name in ["static.prop.dirtdebris01", "static.prop.dirtdebris02", "static.prop.dirtdebris03", "static.prop.tire"]:
            try:
                pothole_bps.append(self.bp.find(name))
            except IndexError:
                pass
        if not pothole_bps:
            pothole_bps = list(self.bp.filter("static.prop.dirtdebris*")) + list(self.bp.filter("static.prop.tire*"))
        if not pothole_bps:
            return
        waypoints = self.map.generate_waypoints(6.0)
        origin = self.ego_wp.transform.location
        close_wps = [wp for wp in waypoints if wp.transform.location.distance(origin) < POTHOLE_SEARCH_RADIUS_M]
        sample_wps = random.sample(close_wps, min(POTHOLE_COUNT, len(close_wps)))
        for wp in sample_wps:
            right = wp.transform.get_right_vector()
            side = random.uniform(-1.4, 1.4)  # narrow unmarked road -- debris can sit closer to the travel path
            tf = carla.Transform(wp.transform.location, wp.transform.rotation)
            tf.location.x += right.x * side
            tf.location.y += right.y * side
            tf.location.z += 0.05
            prop = self.world.try_spawn_actor(random.choice(pothole_bps), tf)
            if prop:
                prop.set_simulate_physics(False)
                self.static_props.append(prop)

    def _spawn_handcarts(self) -> None:
        cart_bp = self.bp.find(common.RICKSHAW_STANDIN)
        for ahead in HANDCART_AHEAD_M[:HANDCART_COUNT]:
            route = self.ego_wp.next(ahead)
            if not route:
                continue
            wp = route[0]
            tf = carla.Transform(wp.transform.location, wp.transform.rotation)
            right = tf.get_right_vector()
            side = random.choice([-1, 1]) * random.uniform(1.8, 2.6)
            tf.location.x += right.x * side
            tf.location.y += right.y * side
            tf.location.z += 0.2
            tf.rotation.yaw += random.uniform(-25.0, 25.0)
            cart = self.world.try_spawn_actor(cart_bp, tf)
            if cart:
                cart.set_simulate_physics(False)
                self.static_props.append(cart)

    def _update_walkers(self, ego_loc: carla.Location) -> None:
        survivors = []
        for item in self.walkers:
            actor, controller = item["actor"], item["controller"]
            if not actor.is_alive or not controller.is_alive:
                continue
            if not item["started"] and ego_loc.distance(actor.get_location()) < WALKER_TRIGGER_DISTANCE_M:
                controller.set_max_speed(WALKER_MAX_SPEED_MPS * random.uniform(0.6, 1.2))
                item["started"] = True
            if common.is_behind(self.ego, actor.get_location()) and ego_loc.distance(actor.get_location()) > 40.0:
                try:
                    controller.stop()
                    controller.destroy()
                    actor.destroy()
                except RuntimeError:
                    pass
                continue
            survivors.append(item)
        self.walkers = survivors

    def _update_cattle(self, ego_loc: carla.Location) -> None:
        for item in self.cattle:
            actor, controller = item["actor"], item["controller"]
            if not actor.is_alive or not controller.is_alive:
                continue
            if not item["started"] and ego_loc.distance(actor.get_location()) < CATTLE_TRIGGER_DISTANCE_M:
                controller.set_max_speed(CATTLE_MAX_SPEED_MPS)
                item["started"] = True

    def update_spectator(self) -> None:
        if not self.ego or not self.ego.is_alive:
            return
        tf = self.ego.get_transform()
        forward = tf.get_forward_vector()
        up = tf.get_up_vector()
        location = tf.location - forward * 10.0 + up * 5.0
        rotation = carla.Rotation(pitch=-20.0, yaw=tf.rotation.yaw, roll=0.0)
        self.world.get_spectator().set_transform(carla.Transform(location, rotation))

    def _print_ground_truth(self, frame: int) -> None:
        common.print_ground_truth(frame, self.ego, {
            "WALKER": [w["actor"] for w in self.walkers],
            "CATTLE": [c["actor"] for c in self.cattle],
            "ONCOMING": self.oncoming,
            "LOCAL_VEHICLE": self.local_vehicles,
        })

    def step(self) -> None:
        frame = self.world.tick()
        self._print_ground_truth(frame)
        self.update_spectator()
        self.frame_count += 1
        self.sim_time += DT
        snap = self.world.get_snapshot()
        ego_loc = self.ego.get_location()
        self.trajectory.append({"frame": frame, "t": snap.timestamp.elapsed_seconds, "x": ego_loc.x, "y": ego_loc.y, "z": ego_loc.z})
        common.send_sensors(self.uploader, self.rgb_buf, self.lidar_buf, self.radar_buf, frame, self.ego, snap)
        self._update_walkers(ego_loc)
        self._update_cattle(ego_loc)

    def _on_collision(self, event) -> None:
        self.collisions.append({"frame": int(event.frame), "other": event.other_actor.type_id})
        print(f"[COLLISION] frame={event.frame} other={event.other_actor.type_id}", flush=True)

    def finish(self) -> None:
        common.LOG_DIR.mkdir(exist_ok=True)
        with (common.LOG_DIR / "village_road_trajectory.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "t", "x", "y", "z"])
            writer.writeheader()
            writer.writerows(self.trajectory)
        actors = [self.rgb, self.lidar, self.radar, self.collision_sensor, self.ego]
        actors += [w["actor"] for w in self.walkers] + [c["actor"] for c in self.cattle]
        actors += self.oncoming + self.local_vehicles + self.static_props
        common.teardown(self.world, self.tm, actors, self.controllers)
        if self.uploader:
            self.uploader.close()
        print(f"[DONE] collisions={len(self.collisions)} frames={self.frame_count}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--ego-spawn-index", type=int, default=0)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--m2-url", default=common.M2_URL)
    p.add_argument("--tm-port", type=int, default=common.TM_PORT)
    p.add_argument("--drive-mode", choices=["autopilot", "m5"], default="autopilot",
                   help="'autopilot': CARLA's own Traffic Manager drives the ego (default). 'm5': "
                        "M5_Pipeline's UDP decisions drive the ego via m5_control.py -- see scenario3.py's "
                        "own --drive-mode for the full rationale (single writer to the ego's control).")
    args = p.parse_args()

    scenario = VillageRoadScenario(args.host, args.port, args.ego_spawn_index, args.m2_url, args.tm_port)
    control_log_file = None
    try:
        scenario.setup()
        if args.drive_mode == "autopilot":
            scenario.ego.set_autopilot(True, scenario.tm_port)
            scenario.tm.vehicle_percentage_speed_difference(scenario.ego, 40.0)  # village roads are slow
            scenario.tm.ignore_lights_percentage(scenario.ego, 100.0)
            for _ in range(int(args.duration / DT)):
                scenario.step()
        else:
            from m5_control import M5ControlExecutor
            executor = M5ControlExecutor()
            common.LOG_DIR.mkdir(exist_ok=True)
            control_log_file = (common.LOG_DIR / "m5_control_log_village_road.jsonl").open("w", encoding="utf-8")
            for _ in range(int(args.duration / DT)):
                scenario.step()
                record = executor.apply(scenario.ego, scenario.world)
                control_log_file.write(json.dumps(record) + "\n")
    except KeyboardInterrupt:
        print("[MAIN] interrupted", flush=True)
    finally:
        if control_log_file is not None:
            control_log_file.close()
        scenario.finish()


if __name__ == "__main__":
    main()

# CARLA 0.9.15 notes:
# - Start the server with a matching 0.9.15 client/server pair.
# - Start a dedicated Traffic Manager on --tm-port (default 8001; use a
#   different port than any other scenario running concurrently).
# - First test with WALKER_COUNT/CATTLE_GROUP_COUNT/ONCOMING_COUNT halved,
#   then increase once stable (same guidance as scenario3.py).
# - --ego-spawn-index defaults to 0; Town07 has few spawn points, so check
#   `len(world.get_map().get_spawn_points())` if this index is out of range.
