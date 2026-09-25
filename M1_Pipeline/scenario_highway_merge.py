#!/usr/bin/env python3
"""SIH Scenario: highway merge with slow-moving vehicles for CARLA 0.9.15
(Town06).

Ego travels at highway speed in a fast stream; the perception/planning
challenge is a heavily loaded truck moving well below stream speed, either
already occupying the ego's lane or merging into it from an adjacent lane,
plus a motorcycle lane-splitting past that same truck at a much higher
closing speed. Town06 is CARLA's own highway-focused map (long multi-lane
highways, multiple on/off-ramps); Town04's highway loop is a reasonable
fallback if Town06 isn't present in the installed CARLA build.

This is the scenario the problem statement explicitly calls out radar for
("Radar velocity should be especially useful here") -- large, fast-changing
closing speeds between the ego, the slow truck, and the overtaking bike are
exactly what M3_Pipeline's radar_los_innovation gate is meant to handle.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from typing import Any

import carla

import common

TOWN = "Town06"
TOWN_FALLBACK = "Town04"
SENSOR_ID = "carla-highway-merge-ego"
RANDOM_SEED = 4001
SCENARIO_WARMUP_S = 4.0

SLOW_TRUCK_AHEAD_M = 60.0
SLOW_TRUCK_SPEED_DIFFERENCE = 65.0  # % SLOWER than the speed limit (positive = slower)

MERGER_BEHIND_M = 25.0
MERGER_SPEED_DIFFERENCE = 55.0  # also slow -- merging in from an on/off-ramp-adjacent lane
MERGE_TRIGGER_GAP_M = 35.0
MERGE_RETRIGGER_S = 8.0

FAST_LANE_PLACEMENTS = [
    # (direction, distance_m, blueprint, speed_difference_pct, side)
    ("ahead", 20.0, "vehicle.toyota.prius", -10.0, None),
    ("ahead", 90.0, "vehicle.nissan.micra", -5.0, None),
    ("ahead", 150.0, "vehicle.mitsubishi.fusorosa", 20.0, None),  # a bus holding the lane below stream speed
    ("behind", 20.0, "vehicle.audi.a2", -15.0, None),
    ("behind", 45.0, "vehicle.mini.cooper_s", -10.0, None),
    ("behind", 70.0, "vehicle.mercedes.sprinter", -5.0, None),  # a tempo/van in the mix
]

RACER_SPAWN_COOLDOWN_MIN_S, RACER_SPAWN_COOLDOWN_MAX_S = 5.0, 8.0
RACER_BEHIND_MIN_M, RACER_BEHIND_MAX_M = 30.0, 55.0
RACER_SPEED_DIFFERENCE = -70.0  # negative => faster than the speed limit
RACER_LIFETIME_S = 15.0

# Live testing (Sep 2026, --drive-mode m5) found the M5-driven ego runs far
# below highway speed here too (YIELD/EMERGENCY_BRAKE dominant, ~1.6 m/s
# mean) -- the slow_truck/merger/fast-lane mechanics still fire reliably
# (they close the gap themselves, or are already close at spawn), but the
# purely cosmetic stalled vehicle/cones/signs below were placed assuming
# highway-speed ego travel and are tightened here so they stay reachable
# within a 30s run (same counts -- not an intensity change).
STALLED_VEHICLE_AHEAD_M = 90.0

# Live CARLA testing (Sep 2026) found this scenario under-exercised
# classes.yaml's taxonomy: no bus/tempo ever appeared (fixed above via
# FAST_LANE_PLACEMENTS), and no traffic_cones/road_sign at all. A lane-
# closure cone line approaching the merge point is a natural fit thematically
# (real highway merges are very often cone-marked construction/lane-closure
# zones) rather than an arbitrary addition.
LANE_CLOSURE_CONE_AHEAD_M = [40.0, 48.0, 56.0, 64.0, 72.0, 80.0, 88.0]
HIGHWAY_SIGN_AHEAD_M = [10.0, 50.0, 90.0]

DT = common.DT


class HighwayMergeScenario:
    def __init__(self, host: str, port: int, ego_index: int, m2_url: str, tm_port: int, town: str):
        self.host, self.port, self.ego_index, self.town = host, port, ego_index, town
        self.m2_url, self.tm_port = m2_url, tm_port
        self.client = self.world = self.map = self.tm = self.bp = None
        self.ego = self.collision_sensor = None
        self.rgb = self.lidar = self.radar = None
        self.rgb_buf = common.FrameBuffer("RGB")
        self.lidar_buf = common.FrameBuffer("LiDAR")
        self.radar_buf = common.FrameBuffer("Radar")
        self.uploader: common.Uploader | None = None
        self.ego_wp = None
        self.slow_truck = None
        self.merger = None
        self.merger_triggered = False
        self.merger_next_trigger = 0.0
        self.lane_traffic: list[Any] = []
        self.racers: list[dict[str, Any]] = []
        self.stalled_vehicle = None
        self.traffic_cones: list[Any] = []
        self.road_signs: list[Any] = []
        self.sim_time = 0.0
        self.frame_count = 0
        self.next_racer_time = 0.0
        self.collisions: list[dict[str, Any]] = []
        self.trajectory: list[dict[str, Any]] = []

    def setup(self) -> None:
        random.seed(RANDOM_SEED)
        try:
            self.client, self.world, self.tm = common.connect(self.host, self.port, self.town, self.tm_port)
        except RuntimeError as exc:
            if self.town != TOWN_FALLBACK:
                print(f"[SETUP][WARN] could not load {self.town} ({exc}); falling back to {TOWN_FALLBACK}", flush=True)
                self.town = TOWN_FALLBACK
                self.client, self.world, self.tm = common.connect(self.host, self.port, self.town, self.tm_port)
            else:
                raise

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

        self._spawn_slow_truck()
        self._spawn_merger()
        self._spawn_fast_lane_traffic()
        self._spawn_stalled_vehicle()
        self._spawn_lane_closure_cones()
        self._spawn_highway_signs()
        # CARLA synchronous mode: a freshly try_spawn_actor'd actor's
        # get_location() reads back a default (0,0,0) transform until the
        # world has ticked at least once -- this tick lets every spawn above
        # sync before _write_obstacle_manifest() below queries positions.
        self.world.tick()
        self.next_racer_time = SCENARIO_WARMUP_S + random.uniform(2.0, 4.0)
        self.merger_next_trigger = SCENARIO_WARMUP_S + random.uniform(1.0, 3.0)
        self.uploader = common.Uploader(self.m2_url, SENSOR_ID)
        print(
            f"[SETUP] highway merge ({self.town}): slow_truck={'yes' if self.slow_truck else 'no'} "
            f"merger={'yes' if self.merger else 'no'} lane_traffic={len(self.lane_traffic)} "
            f"cones={len(self.traffic_cones)} signs={len(self.road_signs)}", flush=True,
        )
        self._write_obstacle_manifest()

    def _write_obstacle_manifest(self) -> None:
        entries: list[dict[str, Any]] = []
        for kind, actor in [("slow_truck", self.slow_truck), ("merger", self.merger), ("stalled_vehicle", self.stalled_vehicle)]:
            if actor is None:
                continue
            loc = actor.get_location()
            note = "spawn position -- moving afterward" if kind != "stalled_vehicle" else None
            entry = {"kind": kind, "blueprint": actor.type_id, "x": loc.x, "y": loc.y, "z": loc.z}
            if note:
                entry["note"] = note
            entries.append(entry)
        for actor in self.lane_traffic:
            loc = actor.get_location()
            entries.append({"kind": "fast_lane_traffic", "blueprint": actor.type_id, "x": loc.x, "y": loc.y, "z": loc.z,
                             "note": "spawn position -- moving afterward"})
        for kind, actors in [("traffic_cone", self.traffic_cones), ("road_sign", self.road_signs)]:
            for actor in actors:
                loc = actor.get_location()
                entries.append({"kind": kind, "blueprint": actor.type_id, "x": loc.x, "y": loc.y, "z": loc.z})
        common.write_obstacle_manifest(
            common.LOG_DIR / "obstacle_manifest_highway_merge.json", entries,
            note="Excludes the continuous 'racer' motorcycle spawner -- its count/timing/position isn't "
                 "deterministic run-to-run even under a fixed seed. The merger's scripted lane-change trigger "
                 "(force_lane_change once within MERGE_TRIGGER_GAP_M) fires at runtime, not at this spawn position.",
        )

    def _spawn_lane_closure_cones(self) -> None:
        self.traffic_cones = common.spawn_roadside_props(
            self.world, self.bp, self.ego_wp, common.TRAFFIC_CONE_PROPS, LANE_CLOSURE_CONE_AHEAD_M, side_m=1.8,
        )

    def _spawn_highway_signs(self) -> None:
        self.road_signs = common.spawn_roadside_props(
            self.world, self.bp, self.ego_wp, common.ROAD_SIGN_PROPS, HIGHWAY_SIGN_AHEAD_M, side_m=3.2,
        )

    def _spawn_slow_truck(self) -> None:
        route = self.ego_wp.next(SLOW_TRUCK_AHEAD_M)
        if not route:
            return
        tf = route[0].transform
        tf.location.z += 0.3
        name = random.choice(common.TRUCKS)
        actor = self.world.try_spawn_actor(self.bp.find(name), tf)
        if actor is None:
            return
        actor.set_simulate_physics(True)
        actor.set_enable_gravity(True)
        actor.set_autopilot(True, self.tm_port)
        self.tm.ignore_lights_percentage(actor, 100.0)
        self.tm.ignore_signs_percentage(actor, 100.0)
        self.tm.vehicle_percentage_speed_difference(actor, SLOW_TRUCK_SPEED_DIFFERENCE)
        self.tm.distance_to_leading_vehicle(actor, 3.0)
        self.slow_truck = actor
        print(f"[SETUP] SLOW_TRUCK actor={actor.id} type={name} ahead={SLOW_TRUCK_AHEAD_M}m", flush=True)

    def _spawn_merger(self) -> None:
        """A second slow vehicle in an adjacent lane, released to cut into
        the ego's lane once the ego closes to MERGE_TRIGGER_GAP_M -- the
        actual "merge" event, distinct from the truck already in-lane."""
        route = self.ego_wp.previous(MERGER_BEHIND_M)
        if not route:
            return
        wp = route[0]
        left_wp = wp.get_left_lane()
        target_wp = left_wp if (left_wp and left_wp.lane_type == carla.LaneType.Driving) else wp
        tf = target_wp.transform
        tf.location.z += 0.3
        name = random.choice(common.TRUCKS + common.TEMPOS + [common.RICKSHAW_STANDIN])
        actor = self.world.try_spawn_actor(self.bp.find(name), tf)
        if actor is None:
            return
        actor.set_simulate_physics(True)
        actor.set_enable_gravity(True)
        actor.set_autopilot(True, self.tm_port)
        self.tm.ignore_lights_percentage(actor, 100.0)
        self.tm.ignore_signs_percentage(actor, 100.0)
        self.tm.vehicle_percentage_speed_difference(actor, MERGER_SPEED_DIFFERENCE)
        self.tm.auto_lane_change(actor, False)  # only changes lane when WE force it (the merge event)
        self.merger = actor
        print(f"[SETUP] MERGER actor={actor.id} type={name} behind={MERGER_BEHIND_M}m", flush=True)

    def _spawn_fast_lane_traffic(self) -> None:
        for direction, distance, name, speed_difference, _side in FAST_LANE_PLACEMENTS:
            route = self.ego_wp.next(distance) if direction == "ahead" else self.ego_wp.previous(distance)
            if not route:
                continue
            tf = route[0].transform
            tf.location.z += 0.3
            actor = self.world.try_spawn_actor(self.bp.find(name), tf)
            if actor is None:
                continue
            actor.set_simulate_physics(True)
            actor.set_enable_gravity(True)
            actor.set_autopilot(True, self.tm_port)
            self.tm.ignore_lights_percentage(actor, 100.0)
            self.tm.ignore_signs_percentage(actor, 100.0)
            self.tm.auto_lane_change(actor, True)
            self.tm.vehicle_percentage_speed_difference(actor, speed_difference)
            self.lane_traffic.append(actor)

    def _spawn_stalled_vehicle(self) -> None:
        route = self.ego_wp.next(STALLED_VEHICLE_AHEAD_M)
        if not route:
            return
        wp = route[0]
        tf = carla.Transform(wp.transform.location, wp.transform.rotation)
        right = tf.get_right_vector()
        tf.location.x += right.x * 3.0
        tf.location.y += right.y * 3.0
        tf.location.z += 0.2
        actor = self.world.try_spawn_actor(self.bp.find(random.choice(common.CARS)), tf)
        if actor:
            actor.set_simulate_physics(False)
            self.stalled_vehicle = actor
            print(f"[SETUP] STALLED_VEHICLE actor={actor.id} ahead={STALLED_VEHICLE_AHEAD_M}m", flush=True)

    def _spawn_and_launch_racer(self) -> None:
        if not self.ego or not self.ego.is_alive:
            return
        ego_wp = self.map.get_waypoint(self.ego.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
        if ego_wp is None:
            return
        behind_dist = random.uniform(RACER_BEHIND_MIN_M, RACER_BEHIND_MAX_M)
        route = ego_wp.previous(behind_dist)
        if not route:
            return
        tf = route[0].transform
        tf.location.z += 0.3
        name = random.choice(common.BIKES)
        actor = self.world.try_spawn_actor(self.bp.find(name), tf)
        if actor is None:
            return
        actor.set_simulate_physics(True)
        actor.set_enable_gravity(True)
        actor.set_autopilot(True, self.tm_port)
        self.tm.ignore_lights_percentage(actor, 100.0)
        self.tm.ignore_signs_percentage(actor, 100.0)
        self.tm.ignore_vehicles_percentage(actor, 70.0)
        self.tm.auto_lane_change(actor, True)
        self.tm.vehicle_percentage_speed_difference(actor, RACER_SPEED_DIFFERENCE)
        self.racers.append({"actor": actor, "spawn_time": self.sim_time})
        print(f"[RUNTIME] RACER_SPAWN actor={actor.id} type={name} behind={behind_dist:.1f}m", flush=True)

    def _maybe_spawn_racer(self) -> None:
        if self.sim_time < self.next_racer_time:
            return
        self._spawn_and_launch_racer()
        self.next_racer_time = self.sim_time + random.uniform(RACER_SPAWN_COOLDOWN_MIN_S, RACER_SPAWN_COOLDOWN_MAX_S)

    def _update_racers(self) -> None:
        survivors = []
        for item in self.racers:
            actor = item["actor"]
            if not actor.is_alive:
                continue
            age = self.sim_time - item["spawn_time"]
            far_ahead = (
                not common.is_behind(self.ego, actor.get_location())
                and actor.get_location().distance(self.ego.get_location()) > 60.0
            )
            if age > RACER_LIFETIME_S or far_ahead:
                try:
                    actor.destroy()
                except RuntimeError:
                    pass
                continue
            survivors.append(item)
        self.racers = survivors

    def _maybe_trigger_merge(self, ego_loc: carla.Location) -> None:
        if self.merger is None or not self.merger.is_alive or self.sim_time < self.merger_next_trigger:
            return
        gap = self.merger.get_location().distance(ego_loc)
        if gap < MERGE_TRIGGER_GAP_M:
            try:
                self.tm.force_lane_change(self.merger, random.choice([True, False]))
                self.merger_triggered = True
                self.merger_next_trigger = self.sim_time + MERGE_RETRIGGER_S
                print(f"[RUNTIME] MERGE_CUT actor={self.merger.id} gap={gap:.1f}m", flush=True)
            except RuntimeError as exc:
                print(f"[RUNTIME][WARN] merge lane change failed: {exc}", flush=True)

    def update_spectator(self) -> None:
        if not self.ego or not self.ego.is_alive:
            return
        tf = self.ego.get_transform()
        forward = tf.get_forward_vector()
        up = tf.get_up_vector()
        location = tf.location - forward * 12.0 + up * 6.0
        rotation = carla.Rotation(pitch=-18.0, yaw=tf.rotation.yaw, roll=0.0)
        self.world.get_spectator().set_transform(carla.Transform(location, rotation))

    def _print_ground_truth(self, frame: int) -> None:
        groups: dict[str, list] = {"LANE": self.lane_traffic, "RACER": [r["actor"] for r in self.racers]}
        if self.slow_truck:
            groups["SLOW_TRUCK"] = [self.slow_truck]
        if self.merger:
            groups["MERGER"] = [self.merger]
        if self.stalled_vehicle:
            groups["STALLED"] = [self.stalled_vehicle]
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
        self._maybe_trigger_merge(ego_loc)
        self._maybe_spawn_racer()
        self._update_racers()

    def _on_collision(self, event) -> None:
        self.collisions.append({"frame": int(event.frame), "other": event.other_actor.type_id})
        print(f"[COLLISION] frame={event.frame} other={event.other_actor.type_id}", flush=True)

    def finish(self) -> None:
        common.LOG_DIR.mkdir(exist_ok=True)
        with (common.LOG_DIR / "highway_merge_trajectory.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "t", "x", "y", "z"])
            writer.writeheader()
            writer.writerows(self.trajectory)
        actors = [self.rgb, self.lidar, self.radar, self.collision_sensor, self.ego, self.slow_truck, self.merger, self.stalled_vehicle]
        actors += self.lane_traffic + [r["actor"] for r in self.racers] + self.traffic_cones + self.road_signs
        common.teardown(self.world, self.tm, actors)
        if self.uploader:
            self.uploader.close()
        print(f"[DONE] collisions={len(self.collisions)} frames={self.frame_count}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--town", default=TOWN)
    p.add_argument("--ego-spawn-index", type=int, default=0)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--m2-url", default=common.M2_URL)
    p.add_argument("--tm-port", type=int, default=common.TM_PORT)
    p.add_argument("--drive-mode", choices=["autopilot", "m5"], default="autopilot",
                   help="'autopilot': CARLA's own Traffic Manager drives the ego (default). 'm5': "
                        "M5_Pipeline's UDP decisions drive the ego via m5_control.py -- see scenario3.py's "
                        "own --drive-mode for the full rationale (single writer to the ego's control).")
    args = p.parse_args()

    scenario = HighwayMergeScenario(args.host, args.port, args.ego_spawn_index, args.m2_url, args.tm_port, args.town)
    control_log_file = None
    try:
        scenario.setup()
        if args.drive_mode == "autopilot":
            scenario.ego.set_autopilot(True, scenario.tm_port)
            scenario.tm.vehicle_percentage_speed_difference(scenario.ego, -5.0)  # ego runs close to the highway limit
            scenario.tm.ignore_lights_percentage(scenario.ego, 100.0)
            for _ in range(int(args.duration / DT)):
                scenario.step()
        else:
            from m5_control import M5ControlExecutor
            executor = M5ControlExecutor()
            common.LOG_DIR.mkdir(exist_ok=True)
            control_log_file = (common.LOG_DIR / "m5_control_log_highway_merge.jsonl").open("w", encoding="utf-8")
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
# - Town06 ships with the standard CARLA asset package; if it's missing from
#   the installed build, pass --town Town04 explicitly (its highway loop is
#   the next-closest fit) -- the script also auto-falls-back if load_world()
#   raises for the requested town.
# - Start a dedicated Traffic Manager on --tm-port, distinct from any other
#   scenario running concurrently.
# - --ego-spawn-index defaults to 0; pick one on the highway proper (not a
#   ramp) by checking the map in the CARLA UE4 editor/spectator first.
