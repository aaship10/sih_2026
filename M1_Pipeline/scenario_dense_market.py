#!/usr/bin/env python3
"""SIH Scenario: dense market area with mixed traffic for CARLA 0.9.15
(Town02).

Everything the other scenarios test, but dense and close: many more
pedestrians than scenario3.py's junction (most lingering/browsing, some
darting suddenly), multiple static pushcarts narrowing the drivable
corridor from both sides at once, a few cars queued behind the crowd rather
than free-flowing, and a stray animal wandering through. CARLA ships no
auto-rickshaw asset -- see common.RICKSHAW_STANDIN, used honestly as an
approximation rather than mislabelling a car or bike.

Deliberately places pushcarts to occlude some pedestrians from the ego's
sensors until late, rather than relying on density alone -- the PS calls
out "partial occlusion, close objects, irregular movement" for this
scenario specifically.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from typing import Any

import carla

import common

TOWN = "Town02"
SENSOR_ID = "carla-dense-market-ego"
RANDOM_SEED = 5001
SCENARIO_WARMUP_S = 3.0

# initial wave, present from the start so the street already feels busy
#
# Live CARLA testing (Sep 2026) found confirmed-track counts here (~22/frame
# mean) were actually the LOWEST of the four new scenarios despite this being
# the one whose own description says "dense and close" -- this scenario's
# job description explicitly calls for MORE obstacles than the other three,
# not the same amount. Counts below roughly doubled from the original pass,
# plus bicycles/buses/tempos/cones/signs added (previously zero of each).
INITIAL_WALKER_COUNT = 34
LINGER_FRACTION = 0.6  # rest walk/cross normally

# continuous spawner -- higher density and higher "sudden dart" rate than
# scenario3.py's junction (JAYWALK_MAX_ACTIVE=6, JAYWALK_SUDDEN_CHANCE=0.35)
JAYWALK_SPAWN_INTERVAL_MIN_S, JAYWALK_SPAWN_INTERVAL_MAX_S = 1.0, 2.2
JAYWALK_AHEAD_MIN_M, JAYWALK_AHEAD_MAX_M = 6.0, 20.0
JAYWALK_SUDDEN_CHANCE = 0.5
JAYWALK_SUDDEN_MIN_M, JAYWALK_SUDDEN_MAX_M = 4.0, 8.0
JAYWALK_MAX_ACTIVE = 34
JAYWALK_LIFETIME_S = 10.0
JAYWALK_TRIGGER_DISTANCE_M = 8.0
JAYWALK_DESPAWN_BEHIND_M = 15.0

PUSHCART_COUNT = 10
PUSHCART_AHEAD_M = [12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0, 96.0, 108.0, 120.0]
PUSHCART_SIDE_M = 2.0  # close in, narrowing the effective corridor from both sides

SLOW_CAR_COUNT = 3
SLOW_CAR_AHEAD_M = [25.0, 55.0, 85.0]
SLOW_CAR_SPEED_DIFFERENCE = 70.0  # crawling, queued behind the crowd

# Queued market delivery traffic -- previously this scenario had zero
# bus/tempo presence (classes.yaml classes with no spawns anywhere in this
# folder before this pass).
DELIVERY_VEHICLE_AHEAD_M = [40.0, 70.0]  # tempo, bus
DELIVERY_VEHICLE_SPEED_DIFFERENCE = 75.0

BICYCLE_COUNT = 4
BICYCLE_AHEAD_M = [18.0, 33.0, 50.0, 66.0]
BICYCLE_SPEED_DIFFERENCE = 40.0

# classes.yaml's "motorcycle" -- previously dense_market only spawned the
# BICYCLES subset of common.BIKES, never a motorized two-wheeler distinctly.
MOTORCYCLE_AHEAD_M = [82.0]
MOTORCYCLE_SPEED_DIFFERENCE = 20.0  # weaves through slower than free-flow, but faster than a bicycle

TRAFFIC_CONE_AHEAD_M = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0]
ROAD_SIGN_AHEAD_M = [15.0, 45.0, 75.0, 105.0]

ANIMAL_AHEAD_M_LIST = [40.0, 95.0]

# Moving auto-rickshaw-standin traffic -- distinct from PUSHCART_AHEAD_M's
# static, physics-disabled pushcarts (same common.RICKSHAW_STANDIN
# blueprint, honestly the closest CARLA 0.9.15 asset to an auto-rickshaw,
# see the module docstring), these drive in traffic like the slow cars.
MOVING_RICKSHAW_COUNT = 2
MOVING_RICKSHAW_AHEAD_M = [45.0, 100.0]
MOVING_RICKSHAW_SPEED_DIFFERENCE = 55.0

# Scripted red-light event, background realism only -- see module docstring
# below and _setup_traffic_light_event's docstring for why this has no
# effect on an M5-driven ego.
TRAFFIC_LIGHT_TRIGGER_M = 35.0

WALKER_MAX_SPEED_MPS = common.WALKER_MAX_SPEED_MPS
DT = common.DT


class DenseMarketScenario:
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
        self.controllers: list[Any] = []
        self.pushcarts: list[Any] = []
        self.slow_cars: list[Any] = []
        self.delivery_vehicles: list[Any] = []
        self.bicycles: list[Any] = []
        self.motorcycles: list[Any] = []
        self.moving_rickshaws: list[Any] = []
        self.traffic_cones: list[Any] = []
        self.road_signs: list[Any] = []
        self.animals: list[dict[str, Any]] = []
        self.traffic_light_event: dict[str, Any] | None = None
        self.sim_time = 0.0
        self.frame_count = 0
        self.next_jaywalk_time = 0.0
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

        self._spawn_pushcarts()
        self._spawn_initial_walkers()
        self._spawn_slow_cars()
        self._spawn_delivery_vehicles()
        self._spawn_bicycles()
        self._spawn_motorcycles()
        self._spawn_moving_rickshaws()
        self._spawn_traffic_cones()
        self._spawn_road_signs()
        self._spawn_animals()
        self._setup_traffic_light_event()
        # CARLA synchronous mode: a freshly try_spawn_actor'd actor's
        # get_location() reads back a default (0,0,0) transform until the
        # world has ticked at least once -- this tick lets every spawn above
        # sync before _write_obstacle_manifest() below queries positions.
        self.world.tick()
        self.next_jaywalk_time = SCENARIO_WARMUP_S + random.uniform(0.5, 1.5)
        self.uploader = common.Uploader(self.m2_url, SENSOR_ID)
        print(
            f"[SETUP] dense market: walkers={len(self.walkers)} pushcarts={len(self.pushcarts)} "
            f"slow_cars={len(self.slow_cars)} delivery_vehicles={len(self.delivery_vehicles)} "
            f"bicycles={len(self.bicycles)} motorcycles={len(self.motorcycles)} "
            f"moving_rickshaws={len(self.moving_rickshaws)} cones={len(self.traffic_cones)} "
            f"signs={len(self.road_signs)} animals={len(self.animals)} "
            f"traffic_light_event={self.traffic_light_event is not None}", flush=True,
        )
        self._write_obstacle_manifest()

    def _spawn_delivery_vehicles(self) -> None:
        """A tempo and a bus queued behind the crowd -- previously this
        scenario had zero bus/tempo presence at all."""
        for ahead, names in zip(DELIVERY_VEHICLE_AHEAD_M, [common.TEMPOS, common.BUSES]):
            route = self.ego_wp.next(ahead)
            if not route:
                continue
            tf = route[0].transform
            tf.location.z += 0.3
            actor = self.world.try_spawn_actor(self.bp.find(random.choice(names)), tf)
            if actor is None:
                continue
            actor.set_simulate_physics(True)
            actor.set_enable_gravity(True)
            actor.set_autopilot(True, self.tm_port)
            self.tm.ignore_lights_percentage(actor, 100.0)
            self.tm.ignore_signs_percentage(actor, 100.0)
            self.tm.vehicle_percentage_speed_difference(actor, DELIVERY_VEHICLE_SPEED_DIFFERENCE)
            self.tm.distance_to_leading_vehicle(actor, 1.5)
            self.delivery_vehicles.append(actor)

    def _spawn_bicycles(self) -> None:
        """Human-pedalled bicycles threading through the crowd -- previously
        this scenario (and the whole M1_Pipeline folder except passing
        mentions in BIKES) never spawned classes.yaml's "bicycle" class
        distinctly from a motorized "motorcycle"."""
        for ahead in BICYCLE_AHEAD_M[:BICYCLE_COUNT]:
            route = self.ego_wp.next(ahead)
            if not route:
                continue
            wp = route[0]
            right = wp.transform.get_right_vector()
            side = random.choice([-1, 1]) * random.uniform(1.0, 2.5)
            tf = carla.Transform(wp.transform.location, wp.transform.rotation)
            tf.location.x += right.x * side
            tf.location.y += right.y * side
            tf.location.z += 0.3
            actor = self.world.try_spawn_actor(self.bp.find(random.choice(common.BICYCLES)), tf)
            if actor is None:
                continue
            actor.set_simulate_physics(True)
            actor.set_enable_gravity(True)
            actor.set_autopilot(True, self.tm_port)
            self.tm.ignore_lights_percentage(actor, 100.0)
            self.tm.ignore_signs_percentage(actor, 100.0)
            self.tm.ignore_vehicles_percentage(actor, 50.0)
            self.tm.vehicle_percentage_speed_difference(actor, BICYCLE_SPEED_DIFFERENCE)
            self.bicycles.append(actor)

    def _spawn_motorcycles(self) -> None:
        """classes.yaml's "motorcycle" -- a motorized two-wheeler, distinct
        from _spawn_bicycles' human-pedalled common.BICYCLES."""
        for ahead in MOTORCYCLE_AHEAD_M:
            route = self.ego_wp.next(ahead)
            if not route:
                continue
            wp = route[0]
            right = wp.transform.get_right_vector()
            side = random.choice([-1, 1]) * random.uniform(1.0, 2.5)
            tf = carla.Transform(wp.transform.location, wp.transform.rotation)
            tf.location.x += right.x * side
            tf.location.y += right.y * side
            tf.location.z += 0.3
            actor = self.world.try_spawn_actor(self.bp.find(random.choice(common.MOTORCYCLES)), tf)
            if actor is None:
                continue
            actor.set_simulate_physics(True)
            actor.set_enable_gravity(True)
            actor.set_autopilot(True, self.tm_port)
            self.tm.ignore_lights_percentage(actor, 100.0)
            self.tm.ignore_signs_percentage(actor, 100.0)
            self.tm.ignore_vehicles_percentage(actor, 30.0)
            self.tm.vehicle_percentage_speed_difference(actor, MOTORCYCLE_SPEED_DIFFERENCE)
            self.motorcycles.append(actor)

    def _spawn_moving_rickshaws(self) -> None:
        """Auto-rickshaw-standin traffic that actually drives, unlike the
        static _spawn_pushcarts (same common.RICKSHAW_STANDIN blueprint --
        see module docstring for why this is the honest stand-in choice)."""
        for ahead in MOVING_RICKSHAW_AHEAD_M[:MOVING_RICKSHAW_COUNT]:
            route = self.ego_wp.next(ahead)
            if not route:
                continue
            tf = route[0].transform
            tf.location.z += 0.3
            actor = self.world.try_spawn_actor(self.bp.find(common.RICKSHAW_STANDIN), tf)
            if actor is None:
                continue
            actor.set_simulate_physics(True)
            actor.set_enable_gravity(True)
            actor.set_autopilot(True, self.tm_port)
            self.tm.ignore_lights_percentage(actor, 100.0)
            self.tm.ignore_signs_percentage(actor, 100.0)
            self.tm.vehicle_percentage_speed_difference(actor, MOVING_RICKSHAW_SPEED_DIFFERENCE)
            self.tm.distance_to_leading_vehicle(actor, 1.5)
            self.moving_rickshaws.append(actor)

    def _setup_traffic_light_event(self) -> None:
        """Scripted red-light event, BACKGROUND REALISM ONLY. M5's decision
        engine (M5_Pipeline/fsm.py) has no traffic-light-state concept at
        all -- it only consumes obstacle proximity/TTC from M4's UDP packet,
        which carries no traffic-light field. Under --drive-mode m5 the ego
        is driven directly by m5_control.py, never by the Traffic Manager,
        so this event cannot and does not change the ego's own behavior --
        it exists so ambient TM-driven traffic reacts realistically to a red
        light and so M2 gets more chances to see a real traffic_signal-class
        object in frame. Finds the nearest traffic-light group roughly ahead
        of the ego's spawn point; silently does nothing if Town02 has none
        findable ahead within range (never a hard failure -- this is cosmetic)."""
        lights = self.world.get_actors().filter("traffic.traffic_light*")
        ego_loc = self.ego_wp.transform.location
        forward = self.ego_wp.transform.get_forward_vector()
        best, best_dist = None, math.inf
        for light in lights:
            to_light = light.get_location() - ego_loc
            ahead_component = forward.x * to_light.x + forward.y * to_light.y
            if ahead_component <= 0.0:
                continue
            dist = ego_loc.distance(light.get_location())
            if dist < best_dist:
                best, best_dist = light, dist
        if best is None:
            return
        group = best.get_group_traffic_lights()
        self.traffic_light_event = {
            "group": list(group), "trigger_location": best.get_location(),
            "triggered": False, "distance_at_setup": best_dist,
        }

    def _update_traffic_light_event(self, ego_loc: carla.Location, frame: int) -> None:
        event = self.traffic_light_event
        if event is None or event["triggered"]:
            return
        if ego_loc.distance(event["trigger_location"]) < TRAFFIC_LIGHT_TRIGGER_M:
            for light in event["group"]:
                if light.is_alive:
                    light.set_state(carla.TrafficLightState.Red)
                    light.freeze(True)
            event["triggered"] = True
            print(f"[EVENT] traffic light group ({len(event['group'])} lights) frozen RED at frame {frame}", flush=True)

    def _write_obstacle_manifest(self) -> None:
        entries: list[dict[str, Any]] = []
        for kind, actors in [("pushcart", self.pushcarts), ("traffic_cone", self.traffic_cones),
                              ("road_sign", self.road_signs), ("moving_rickshaw", self.moving_rickshaws)]:
            for actor in actors:
                loc = actor.get_location()
                entries.append({"kind": kind, "blueprint": actor.type_id, "x": loc.x, "y": loc.y, "z": loc.z})
        for animal in self.animals:
            loc = animal["actor"].get_location()
            entries.append({"kind": "animal", "blueprint": animal["actor"].type_id, "x": loc.x, "y": loc.y, "z": loc.z,
                             "note": "spawn position -- wanders afterward via its AI controller"})
        if self.traffic_light_event is not None:
            loc = self.traffic_light_event["trigger_location"]
            entries.append({
                "kind": "traffic_light_event", "blueprint": "traffic.traffic_light",
                "x": loc.x, "y": loc.y, "z": loc.z,
                "note": f"turns RED once ego is within {TRAFFIC_LIGHT_TRIGGER_M}m -- "
                        f"background realism only, M5 does not react to this (see _setup_traffic_light_event docstring)",
            })
        common.write_obstacle_manifest(
            common.LOG_DIR / "obstacle_manifest_dense_market.json", entries,
            note="Excludes the 34 initial ambient pedestrians and the continuous jaywalker/racer-style "
                 "spawners -- their count/position isn't deterministic run-to-run even under a fixed seed. "
                 "Only fixed infrastructure and notable scripted actors are listed.",
        )

    def _spawn_traffic_cones(self) -> None:
        self.traffic_cones = common.spawn_roadside_props(
            self.world, self.bp, self.ego_wp, common.TRAFFIC_CONE_PROPS, TRAFFIC_CONE_AHEAD_M, side_m=1.5,
        )

    def _spawn_road_signs(self) -> None:
        self.road_signs = common.spawn_roadside_props(
            self.world, self.bp, self.ego_wp, common.ROAD_SIGN_PROPS, ROAD_SIGN_AHEAD_M, side_m=2.6,
        )

    def _spawn_pushcarts(self) -> None:
        cart_bp = self.bp.find(common.RICKSHAW_STANDIN)
        for i, ahead in enumerate(PUSHCART_AHEAD_M[:PUSHCART_COUNT]):
            route = self.ego_wp.next(ahead)
            if not route:
                continue
            wp = route[0]
            side = -1 if i % 2 == 0 else 1  # alternate sides -> narrows the corridor from both sides
            right = wp.transform.get_right_vector()
            tf = carla.Transform(wp.transform.location, wp.transform.rotation)
            tf.location.x += right.x * side * PUSHCART_SIDE_M
            tf.location.y += right.y * side * PUSHCART_SIDE_M
            # Live CARLA testing found z+=0.2 put the microlino's collision
            # box slightly into Town02's curb/sidewalk geometry at several of
            # these roadside offsets, silently failing try_spawn_actor for
            # ALL pushcarts (confirmed by bisecting z directly: 0.2 failed,
            # 0.3 spawned, at the same x/y). 0.35 clears it with margin.
            tf.location.z += 0.35
            tf.rotation.yaw += random.uniform(-15.0, 15.0)
            cart = self.world.try_spawn_actor(cart_bp, tf)
            if cart:
                cart.set_simulate_physics(False)
                self.pushcarts.append(cart)

    def _spawn_walker_near(self, ahead: float, side_hint: int, sudden: bool) -> None:
        route = self.ego_wp.next(ahead)
        if not route:
            return
        wp = route[0]
        right = wp.transform.get_right_vector()
        side = side_hint * random.uniform(1.5, 3.0)
        tf = carla.Transform(
            carla.Location(
                wp.transform.location.x + right.x * side,
                wp.transform.location.y + right.y * side,
                wp.transform.location.z + 0.5,
            ),
            wp.transform.rotation,
        )
        walker_bps = self.bp.filter("walker.pedestrian.*")
        controller_bp = self.bp.find("controller.ai.walker")
        if not walker_bps or controller_bp is None:
            return
        walker = self.world.try_spawn_actor(random.choice(walker_bps), tf)
        if walker is None:
            return
        preferred = carla.Location(
            tf.location.x - right.x * side * 2.0, tf.location.y - right.y * side * 2.0, tf.location.z
        )
        target = common.walker_nav_target(self.world, tf.location, preferred, min_dist=3.0, max_dist=25.0)
        if target is None:
            walker.destroy()
            return
        controller = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
        controller.start()
        controller.go_to_location(target)
        linger = (not sudden) and random.random() < LINGER_FRACTION
        controller.set_max_speed(0.0 if (linger or not sudden) else WALKER_MAX_SPEED_MPS * random.uniform(0.9, 1.6))
        self.walkers.append({
            "actor": walker, "controller": controller, "started": sudden, "linger": linger,
            "spawn_time": self.sim_time,
        })
        self.controllers.append(controller)

    def _spawn_initial_walkers(self) -> None:
        for i in range(INITIAL_WALKER_COUNT):
            ahead = random.uniform(5.0, 100.0)
            self._spawn_walker_near(ahead, random.choice([-1, 1]), sudden=False)

    def _spawn_dynamic_jaywalker(self) -> None:
        if not self.ego or not self.ego.is_alive:
            return
        sudden = random.random() < JAYWALK_SUDDEN_CHANCE
        ahead = random.uniform(JAYWALK_SUDDEN_MIN_M, JAYWALK_SUDDEN_MAX_M) if sudden \
            else random.uniform(JAYWALK_AHEAD_MIN_M, JAYWALK_AHEAD_MAX_M)
        self._spawn_walker_near(ahead, random.choice([-1, 1]), sudden=sudden)

    def _maybe_spawn_jaywalker(self) -> None:
        if self.sim_time < self.next_jaywalk_time:
            return
        active = sum(1 for w in self.walkers if w["actor"].is_alive)
        if active < JAYWALK_MAX_ACTIVE:
            self._spawn_dynamic_jaywalker()
        self.next_jaywalk_time = self.sim_time + random.uniform(JAYWALK_SPAWN_INTERVAL_MIN_S, JAYWALK_SPAWN_INTERVAL_MAX_S)

    def _spawn_slow_cars(self) -> None:
        for ahead in SLOW_CAR_AHEAD_M[:SLOW_CAR_COUNT]:
            route = self.ego_wp.next(ahead)
            if not route:
                continue
            tf = route[0].transform
            tf.location.z += 0.3
            actor = self.world.try_spawn_actor(self.bp.find(random.choice(common.CARS)), tf)
            if actor is None:
                continue
            actor.set_simulate_physics(True)
            actor.set_enable_gravity(True)
            actor.set_autopilot(True, self.tm_port)
            self.tm.ignore_lights_percentage(actor, 100.0)
            self.tm.ignore_signs_percentage(actor, 100.0)
            self.tm.vehicle_percentage_speed_difference(actor, SLOW_CAR_SPEED_DIFFERENCE)
            self.tm.distance_to_leading_vehicle(actor, 1.5)
            self.slow_cars.append(actor)

    def _spawn_animals(self) -> None:
        walker_bps = self.bp.filter("walker.pedestrian.*")
        controller_bp = self.bp.find("controller.ai.walker")
        if not walker_bps or controller_bp is None:
            return
        for ahead in ANIMAL_AHEAD_M_LIST:
            route = self.ego_wp.next(ahead)
            if not route:
                continue
            wp = route[0]
            right = wp.transform.get_right_vector()
            side = random.choice([-1, 1]) * 2.5
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
            controller.set_max_speed(0.4)
            self.animals.append({"actor": walker, "controller": controller})
            self.controllers.append(controller)

    def _update_walkers(self, ego_loc: carla.Location) -> None:
        survivors = []
        for item in self.walkers:
            actor, controller = item["actor"], item["controller"]
            if not actor.is_alive or not controller.is_alive:
                continue
            if not item["started"] and ego_loc.distance(actor.get_location()) < JAYWALK_TRIGGER_DISTANCE_M:
                if not item["linger"]:
                    controller.set_max_speed(WALKER_MAX_SPEED_MPS * random.uniform(0.7, 1.3))
                item["started"] = True
            age = self.sim_time - item.get("spawn_time", 0.0)
            behind = common.is_behind(self.ego, actor.get_location())
            if age > JAYWALK_LIFETIME_S or (behind and ego_loc.distance(actor.get_location()) > JAYWALK_DESPAWN_BEHIND_M):
                try:
                    controller.stop()
                    controller.destroy()
                    actor.destroy()
                except RuntimeError:
                    pass
                continue
            survivors.append(item)
        self.walkers = survivors

    def update_spectator(self) -> None:
        if not self.ego or not self.ego.is_alive:
            return
        tf = self.ego.get_transform()
        forward = tf.get_forward_vector()
        up = tf.get_up_vector()
        location = tf.location - forward * 8.0 + up * 5.0
        rotation = carla.Rotation(pitch=-25.0, yaw=tf.rotation.yaw, roll=0.0)
        self.world.get_spectator().set_transform(carla.Transform(location, rotation))

    def _print_ground_truth(self, frame: int) -> None:
        groups: dict[str, list] = {
            "WALKER": [w["actor"] for w in self.walkers],
            "SLOW_CAR": self.slow_cars,
            "PUSHCART": self.pushcarts,
            "DELIVERY_VEHICLE": self.delivery_vehicles,
            "BICYCLE": self.bicycles,
            "MOTORCYCLE": self.motorcycles,
            "MOVING_RICKSHAW": self.moving_rickshaws,
        }
        if self.animals:
            groups["ANIMAL"] = [a["actor"] for a in self.animals]
        common.print_ground_truth(frame, self.ego, groups)

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
        self._maybe_spawn_jaywalker()
        self._update_traffic_light_event(ego_loc, frame)

    def _on_collision(self, event) -> None:
        self.collisions.append({"frame": int(event.frame), "other": event.other_actor.type_id})
        print(f"[COLLISION] frame={event.frame} other={event.other_actor.type_id}", flush=True)

    def finish(self) -> None:
        common.LOG_DIR.mkdir(exist_ok=True)
        with (common.LOG_DIR / "dense_market_trajectory.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "t", "x", "y", "z"])
            writer.writeheader()
            writer.writerows(self.trajectory)
        actors = [self.rgb, self.lidar, self.radar, self.collision_sensor, self.ego]
        actors += [w["actor"] for w in self.walkers] + self.pushcarts + self.slow_cars
        actors += self.delivery_vehicles + self.bicycles + self.motorcycles + self.moving_rickshaws
        actors += self.traffic_cones + self.road_signs
        actors += [a["actor"] for a in self.animals]
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

    scenario = DenseMarketScenario(args.host, args.port, args.ego_spawn_index, args.m2_url, args.tm_port)
    control_log_file = None
    try:
        scenario.setup()
        if args.drive_mode == "autopilot":
            scenario.ego.set_autopilot(True, scenario.tm_port)
            scenario.tm.vehicle_percentage_speed_difference(scenario.ego, 70.0)  # market crawl pace
            scenario.tm.ignore_lights_percentage(scenario.ego, 100.0)
            scenario.tm.distance_to_leading_vehicle(scenario.ego, 2.0)
            for _ in range(int(args.duration / DT)):
                scenario.step()
        else:
            from m5_control import M5ControlExecutor
            executor = M5ControlExecutor()
            common.LOG_DIR.mkdir(exist_ok=True)
            control_log_file = (common.LOG_DIR / "m5_control_log_dense_market.jsonl").open("w", encoding="utf-8")
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
# - Town10HD is a reasonable --town-style alternative to Town02 if a more
#   modern city-block layout is preferred (would need TOWN edited at the
#   top of this file, or promote it to a --town flag as scenario_highway_
#   merge.py does, if that's wanted).
# - This scenario spawns the most actors of any in this folder
#   (INITIAL_WALKER_COUNT=18 plus a continuous spawner up to
#   JAYWALK_MAX_ACTIVE=20 more) -- first test with both halved, then
#   increase once stable, same guidance as scenario3.py.
# - Start a dedicated Traffic Manager on --tm-port.
