#!/usr/bin/env python3
"""SIH Scenario: sudden cattle-crossing events for CARLA 0.9.15 (Town05).

Unlike scenario3.py's cattle (visible from CATTLE_TRIGGER_DISTANCE_M=12.0
while already in view), this scenario takes the problem statement's "sudden
appearance" language literally: each crossing point has an occluding
static prop (a parked truck, a container standing in for a wall, or dense
foliage) placed so the cattle are out of the ego camera/LiDAR's line of
sight until the ego is already close, and the group only starts moving once
the ego crosses a short trigger distance. Runs on a straight Town05 segment
away from any junction and deliberately skips scenario3.py's other chaos
(no jaywalkers/overtakers/racers) so the metric this scenario exists to
produce -- detection-to-track latency on a zero-warning event -- isn't
confounded by unrelated traffic. The crossing repeats several times across
the run (varying occluder type and group size) for a statistically
meaningful sample instead of one anecdote.
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

TOWN = "Town05"
SENSOR_ID = "carla-cattle-crossing-ego"
RANDOM_SEED = 6001
SCENARIO_WARMUP_S = 3.0

# Each event is `ahead` meters from the ego's start along its own lane.
# Spaced far enough apart that one event's cattle are gone before the next
# triggers (see CROSSING_LIFETIME_S).
#
# Live CARLA testing (Sep 2026) found this scenario's obstacle count was the
# lowest of the four new scenarios (by design -- see the module docstring's
# "isn't confounded by unrelated traffic"). Group sizes below were increased
# (13 -> 20 cattle total) since more animals per event is still squarely
# within the scenario's own purpose; static roadside clutter (potholes,
# traffic cones, road signs -- see _spawn_static_clutter) was added too,
# since those don't move and can't confound the detection-latency metric the
# way jaywalkers/other traffic would. No moving, unrelated traffic was added,
# on purpose, per the module docstring above.
#
CROSSING_EVENTS_AHEAD_M = [40.0, 90.0, 140.0, 190.0, 240.0]
CROSSING_GROUP_SIZES = [3, 4, 5, 3, 5]
OCCLUDER_TYPES = ["truck", "container", "truck", "container", "truck"]

POTHOLE_AHEAD_M = [15.0, 65.0, 115.0, 165.0, 215.0, 260.0]
TRAFFIC_CONE_AHEAD_M = [25.0, 75.0, 125.0, 175.0, 225.0]
ROAD_SIGN_AHEAD_M = [30.0, 110.0, 200.0]

# Visibility fix (live-run finding): with the crossing events starting at
# 40m, the pothole (15m) and cone (25m) clutter ahead of it meant a run that
# hit any AVOID/recovery trouble on that early clutter -- or simply ran out
# of --duration -- could end without a single cattle crossing ever becoming
# visible, defeating the scenario's whole purpose. A short 2-cow event at
# 25m guarantees a crossing is visible almost immediately regardless of what
# happens later in the run.
#
# 25m, not 15m: live-tested at 15m first and found the ego never moves at
# all -- its own occluder sits at ahead-CROSSING_OCCLUDER_OFFSET_M=12m, so
# little runway remains for the FSM to escalate gradually (CRUISE->FOLLOW->
# AVOID->YIELD) while still accelerating from a dead stop; TTC collapses
# straight through to EMERGENCY_BRAKE within the first ~12s and the vehicle
# never recovers (confirmed via m5_control_log_cattle_crossing.jsonl:
# applied_state=EMERGENCY_BRAKE for 827/900 ticks, net displacement <1.2m
# over the full 45s run). 25m gives enough room for that escalation to
# happen the way it does for every other event above.
#
# Isolated RNG, not just appended to CROSSING_EVENTS_AHEAD_M/inserted into
# the pothole/cone lists (both tried first): `setup()` seeds the single
# global `random` module once, and every spawn call after that -- cattle
# sides, clutter jitter, blueprint choices -- draws from that ONE sequence
# in order. Adding, removing, or reordering ANY entry anywhere upstream
# reshuffles every draw downstream of it. Confirmed live, three times in a
# row: each attempt to fix one incident this way (removing the 15m pothole,
# then the 25m cone, then tightening the cone's jitter) only relocated the
# problem to a DIFFERENT obstacle (a traffic cone, then a cone+occluder
# truck, then Town05's actual roadside fence geometry -- 146, then 346,
# then 360 repeated collisions), never fixed it, because each edit was
# itself just another reshuffle. `EARLY_CROSSING_RNG_SEED` below drives its
# own private random.Random() instance (see _setup_event's `rng` parameter)
# used ONLY for this one event's spawn calls, so it cannot perturb the
# already-validated global sequence the 5 events above (and all static
# clutter) still rely on -- whatever happens with this event's own random
# draws, everything else replays EXACTLY as it did before this was added.
EARLY_CROSSING_AHEAD_M = 25.0
EARLY_CROSSING_GROUP_SIZE = 5  # "a group of 4-5 people crossing together"
EARLY_CROSSING_OCCLUDER_TYPE = "container"
EARLY_CROSSING_RNG_SEED = 6101

# Second isolated event, same reasoning as EARLY_CROSSING_* above: a lone
# crossing later in the run, positioned between the (untouched) 40m and 90m
# events in the main list above so it can't collide spatially with either.
# Both this and EARLY_CROSSING_* are confirmed live-reachable inside the
# default 45s run (2026-09-24 verification: the 90m event -- farther out
# than this one -- triggered with zero collisions), so both the group and
# the lone crossing land within the scenario's own runtime.
LATE_SINGLE_CROSSING_AHEAD_M = 60.0
LATE_SINGLE_CROSSING_GROUP_SIZE = 1
LATE_SINGLE_CROSSING_OCCLUDER_TYPE = "truck"
LATE_SINGLE_CROSSING_RNG_SEED = 6102

CROSSING_TRIGGER_DISTANCE_M = 15.0  # EXPERIMENT (was 9.0): trigger earlier, giving cattle more head start before the ego arrives. 25.0 triggered instantly at spawn for the 25m-ahead event (froze the ego at t=0, not a real test) -- 15.0 still gives a real ~10m approach first.
CROSSING_STAGGER_S = 0.6  # leader-follower: each animal starts a beat after the last
CROSSING_MAX_SPEED_MPS = 1.6  # startled/urgent crossing, faster than a grazing amble
CROSSING_LIFETIME_S = 12.0
CROSSING_OCCLUDER_OFFSET_M = 3.0  # occluder sits this far before the cattle spawn point
CROSSING_SIDE_OFFSET_M = 3.0

DT = common.DT


class CattleCrossingScenario:
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
        self.events: list[dict[str, Any]] = []
        self.controllers: list[Any] = []
        self.static_clutter: list[Any] = []
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

        for ahead, group_size, occluder_type in zip(CROSSING_EVENTS_AHEAD_M, CROSSING_GROUP_SIZES, OCCLUDER_TYPES):
            self._setup_event(ahead, group_size, occluder_type)
        self._spawn_static_clutter()
        # Own private RNG (see EARLY_CROSSING_RNG_SEED's module-level
        # comment) -- must not consume from the shared `random` module the
        # loop above and _spawn_static_clutter() both rely on.
        self._setup_event(
            EARLY_CROSSING_AHEAD_M, EARLY_CROSSING_GROUP_SIZE, EARLY_CROSSING_OCCLUDER_TYPE,
            rng=random.Random(EARLY_CROSSING_RNG_SEED),
        )
        self._setup_event(
            LATE_SINGLE_CROSSING_AHEAD_M, LATE_SINGLE_CROSSING_GROUP_SIZE, LATE_SINGLE_CROSSING_OCCLUDER_TYPE,
            rng=random.Random(LATE_SINGLE_CROSSING_RNG_SEED),
        )

        # CARLA synchronous mode: a freshly try_spawn_actor'd actor's
        # get_location() reads back a default (0,0,0) transform until the
        # world has ticked at least once -- this tick lets every spawn above
        # sync before _write_obstacle_manifest() below queries positions.
        self.world.tick()
        self.uploader = common.Uploader(self.m2_url, SENSOR_ID)
        total_cattle = sum(len(e["cattle"]) for e in self.events)
        print(
            f"[SETUP] cattle crossing: events={len(self.events)} total_cattle={total_cattle} "
            f"static_clutter={len(self.static_clutter)}", flush=True,
        )
        self._write_obstacle_manifest()

    def _write_obstacle_manifest(self) -> None:
        entries: list[dict[str, Any]] = []
        for event in self.events:
            loc = event["crossing_location"]
            entries.append({"kind": "crossing_event", "blueprint": None, "x": loc.x, "y": loc.y, "z": loc.z,
                             "note": f"triggers at {CROSSING_TRIGGER_DISTANCE_M}m, group_size={len(event['cattle'])}"})
            if event["occluder"]:
                oloc = event["occluder"].get_location()
                entries.append({"kind": "occluder", "blueprint": event["occluder"].type_id,
                                 "x": oloc.x, "y": oloc.y, "z": oloc.z})
        for prop in self.static_clutter:
            loc = prop.get_location()
            entries.append({"kind": "static_clutter", "blueprint": prop.type_id, "x": loc.x, "y": loc.y, "z": loc.z})
        common.write_obstacle_manifest(
            common.LOG_DIR / "obstacle_manifest_cattle_crossing.json", entries,
            note="All obstacles here are deterministically placed under RANDOM_SEED -- no continuous/ambient "
                 "spawner in this scenario. Default --duration is 45s (not 30s) since 7 spaced-out events need "
                 "more road; a shorter run will only trigger the earliest ones.",
        )

    def _spawn_static_clutter(self) -> None:
        """Potholes/cones/signs along the route -- static, so they can't
        confound the detection-to-track-latency metric this scenario exists
        to measure the way moving unrelated traffic would (see this module's
        docstring and the constants above)."""
        pothole_bps = []
        for name in ["static.prop.dirtdebris01", "static.prop.dirtdebris02", "static.prop.dirtdebris03", "static.prop.tire"]:
            try:
                pothole_bps.append(self.bp.find(name))
            except IndexError:
                pass
        if pothole_bps:
            for i, ahead in enumerate(POTHOLE_AHEAD_M):
                route = self.ego_wp.next(ahead)
                if not route:
                    continue
                wp = route[0]
                right = wp.transform.get_right_vector()
                side = random.uniform(-1.4, 1.4)
                tf = carla.Transform(wp.transform.location, wp.transform.rotation)
                tf.location.x += right.x * side
                tf.location.y += right.y * side
                tf.location.z += 0.05
                prop = self.world.try_spawn_actor(random.choice(pothole_bps), tf)
                if prop:
                    prop.set_simulate_physics(False)
                    self.static_clutter.append(prop)
        # jitter_m tightened from spawn_roadside_props's 1.5m default: with
        # side_m=1.6 and that much jitter, a cone could land as little as
        # 0.1m off centerline -- well inside the ego's own ~0.94m half-width,
        # i.e. squarely in its straight-line path rather than beside it.
        # Live-confirmed as the cause of a 7-cycle stuck/reverse/re-collide
        # loop (112 collision-sensor contacts) after the early-crossing-event
        # reorder shifted this scenario's random draw sequence: the lane
        # itself was a normal 3.5m (verified via a direct CARLA map query at
        # the wedge position), but the cone sat close enough to center that
        # even AVOID's full achievable lateral room couldn't clear it. 0.1-
        # 0.3m jitter keeps every draw in the 1.3-1.9m band -- past the
        # ego's own path, comfortably within the lane, actually avoidable.
        self.static_clutter += common.spawn_roadside_props(
            self.world, self.bp, self.ego_wp, common.TRAFFIC_CONE_PROPS, TRAFFIC_CONE_AHEAD_M,
            side_m=1.6, jitter_m=0.3,
        )
        self.static_clutter += common.spawn_roadside_props(
            self.world, self.bp, self.ego_wp, common.ROAD_SIGN_PROPS, ROAD_SIGN_AHEAD_M, side_m=2.8,
        )

    def _occluder_blueprint(self, occluder_type: str):
        candidates = {
            "truck": common.TRUCKS,
            "container": ["static.prop.container", "static.prop.containertrash"],
        }.get(occluder_type, common.TRUCKS)
        for name in candidates:
            try:
                return self.bp.find(name), name.startswith("vehicle.")
            except IndexError:
                continue
        return self.bp.find(common.TRUCKS[0]), True

    def _setup_event(self, ahead: float, group_size: int, occluder_type: str, rng=random) -> None:
        """`rng`: defaults to the shared, seeded `random` module (the 5
        original events + all static clutter draw from that ONE sequence,
        in order -- see EARLY_CROSSING_RNG_SEED's module-level comment for
        why). Pass a private random.Random() instance instead for any event
        that must not perturb that shared sequence."""
        route = self.ego_wp.next(ahead)
        if not route:
            return
        crossing_wp = route[0]
        right = crossing_wp.transform.get_right_vector()
        side = rng.choice([-1, 1])

        # Occluder: sits closer to the ego's approach than the cattle, on the
        # crossing side, so it blocks the direct line of sight to the cattle
        # spawn point until the ego is nearly level with it.
        occluder_route = self.ego_wp.next(max(ahead - CROSSING_OCCLUDER_OFFSET_M, 1.0))
        occ_wp = occluder_route[0] if occluder_route else crossing_wp
        occ_bp, occ_is_vehicle = self._occluder_blueprint(occluder_type)
        occ_tf = carla.Transform(occ_wp.transform.location, occ_wp.transform.rotation)
        occ_right = occ_tf.get_right_vector()
        occ_tf.location.x += occ_right.x * side * (CROSSING_SIDE_OFFSET_M - 0.5)
        occ_tf.location.y += occ_right.y * side * (CROSSING_SIDE_OFFSET_M - 0.5)
        occ_tf.location.z += 0.3 if occ_is_vehicle else 0.1
        occluder = self.world.try_spawn_actor(occ_bp, occ_tf)
        if occluder:
            occluder.set_simulate_physics(False)

        # Cattle spawn just past the occluder (further from the road), so
        # the occluder sits between the ego's oncoming view and them.
        #
        # Walker-navmesh fix (live-run finding, the actual root cause of "no
        # cattle ever visibly cross"): this used to drive each cow via
        # controller.ai.walker + go_to_location(), which needs CARLA's own
        # baked pedestrian navmesh -- and this installation serves Town10HD_
        # Opt's navmesh for EVERY town regardless of which one is actually
        # loaded (confirmed live: "Found the required file in cache! Carla/
        # Maps/Nav/Town10HD_Opt.bin" prints even right after an explicit
        # client.load_world("Town05"), and even when CARLA is launched with
        # Town05 as the boot map -- unconditional, not fixable by picking a
        # different spawn point). Every go_to_location() call failed outright
        # ("WARNING: NAV: Failed to set request..."), so no cattle actor in
        # a full run ever moved a single centimeter -- confirmed by ground-
        # truth position logging showing every cattle actor frozen exactly
        # at its spawn point for all 900 ticks. WalkerControl(direction,
        # speed) bypasses the AI controller and navmesh entirely -- verified
        # live to actually move a walker -- so cattle are driven directly
        # with it below instead. No controller.ai.walker involved at all.
        walker_bps = self.bp.filter("walker.pedestrian.*")
        cattle: list[dict[str, Any]] = []
        if walker_bps:
            for i in range(group_size):
                # Crossing-distance fix (live-run finding): the walk target
                # used to be `cow_side * 2.5` on the far side, not `cow_side`
                # -- combined with cow_side itself reaching 4.5-6.1m for a
                # 5-cow group (spawn offset grows 0.4m per animal), that made
                # the one-way walk 16-21m at CROSSING_MAX_SPEED_MPS=1.6, far
                # longer than the ego spends near a single ~25m crossing
                # zone even slowed by AVOID/YIELD. A true mirror (spawn at
                # +cow_side, walk to -cow_side) halves that to 2*cow_side.
                base_cow_side = CROSSING_SIDE_OFFSET_M + 1.5 + i * 0.4

                # Live CARLA testing found the intended `side` is sometimes
                # off-road/into scenery at a given `ahead` distance (Town05's
                # roadside geometry isn't uniform), silently failing EVERY
                # cattle spawn in the group even though the mirrored side is
                # clear -- confirmed by bisecting side=+1 vs side=-1 at the
                # same crossing_wp (one failed 100%, the other succeeded
                # 100%). Falling back to the mirrored side keeps the group
                # size intact instead of losing the whole event to bad luck
                # on one random side choice; this only changes which side of
                # the road a spawn LANDS on when the first attempt is
                # physically blocked, not the occluder's side (still `side`),
                # so occlusion is preserved whenever the primary side works.
                walker = None
                direction = None
                for s in (side * base_cow_side, -side * base_cow_side):
                    sign = 1.0 if s > 0 else -1.0
                    # Crossing direction is always the pure lateral mirror
                    # (spawn at +s, walk toward -s along the same `right`
                    # axis) -- spawning already facing it, rather than the
                    # walker's default forward-of-lane orientation, removes
                    # the ~2s in-place turn CARLA's own walker locomotion
                    # otherwise spends before it starts covering ground
                    # (measured live: full turn-to-target by ~2s, then a
                    # further ~5s ramp to reach CROSSING_MAX_SPEED_MPS).
                    d = carla.Vector3D(-right.x * sign, -right.y * sign, 0.0)
                    face_yaw = math.degrees(math.atan2(d.y, d.x))
                    tf = carla.Transform(
                        carla.Location(
                            crossing_wp.transform.location.x + right.x * s + rng.uniform(-0.5, 0.5),
                            crossing_wp.transform.location.y + right.y * s + rng.uniform(-0.5, 0.5),
                            crossing_wp.transform.location.z + 0.5,
                        ),
                        carla.Rotation(yaw=face_yaw),
                    )
                    walker = self.world.try_spawn_actor(rng.choice(walker_bps), tf)
                    if walker is not None:
                        direction = d
                        break
                if walker is None:
                    continue
                # Standing still until _update_events triggers this cow --
                # same pre-trigger state the old set_max_speed(0.0) gave.
                walker.apply_control(carla.WalkerControl(direction=direction, speed=0.0))
                cattle.append({
                    "actor": walker, "direction": direction, "started": False, "finished": False,
                    "offset": i * CROSSING_STAGGER_S, "walk_distance_m": 2.0 * base_cow_side,
                    "spawn_location": tf.location,
                })

        self.events.append({
            "crossing_location": crossing_wp.transform.location,
            "occluder": occluder,
            "cattle": cattle,
            "triggered": False,
            "trigger_time": None,
        })
        print(
            f"[SETUP] CROSSING_EVENT ahead={ahead:.0f}m occluder={occluder_type} "
            f"group_size={len(cattle)}/{group_size}", flush=True,
        )

    def _update_events(self, ego_loc: carla.Location) -> None:
        for event in self.events:
            if not event["triggered"]:
                if ego_loc.distance(event["crossing_location"]) < CROSSING_TRIGGER_DISTANCE_M:
                    event["triggered"] = True
                    event["trigger_time"] = self.sim_time
                    print(f"[RUNTIME] CROSSING_TRIGGERED at ({event['crossing_location'].x:.1f},"
                          f"{event['crossing_location'].y:.1f}) group_size={len(event['cattle'])}", flush=True)
            else:
                elapsed = self.sim_time - event["trigger_time"]
                for cow in event["cattle"]:
                    if not cow["actor"].is_alive:
                        continue
                    if cow["finished"]:
                        continue
                    if not cow["started"] and elapsed >= cow["offset"]:
                        cow["started"] = True
                        cow["speed"] = CROSSING_MAX_SPEED_MPS * random.uniform(0.85, 1.15)
                    if cow["started"]:
                        # Arrival check (live-run finding): WalkerControl has
                        # no destination of its own -- unlike the old
                        # controller.ai.walker/go_to_location approach, which
                        # stopped on arrival by itself, this direction+speed
                        # command just walks forever until told otherwise.
                        # Without this check a cow kept walking in a straight
                        # line for the rest of the run (confirmed live: one
                        # travelled 50m, straight through every lane, long
                        # after its intended ~10-12m crossing was complete).
                        #
                        # Real measured distance, not walking-time * speed
                        # (live-run finding): CARLA's walker locomotion takes
                        # ~2s to turn to face a new direction and a further
                        # ~5s to ramp up to the commanded speed (confirmed
                        # live via direct position tracking) -- during all of
                        # that, actual distance covered is far below what
                        # elapsed-time * nominal-speed assumes, so that
                        # estimate crossed the walk_distance_m threshold and
                        # stopped cattle well before they'd actually reached
                        # the far side, stranding them mid-road instead of on
                        # the opposite sidewalk (the live-run symptom this
                        # fixes). Querying the actor's own real position
                        # sidesteps the turn/ramp profile entirely.
                        traveled = cow["actor"].get_location().distance(cow["spawn_location"])
                        if traveled >= cow["walk_distance_m"]:
                            cow["actor"].apply_control(carla.WalkerControl(direction=cow["direction"], speed=0.0))
                            cow["finished"] = True
                        else:
                            # Reapplied every tick, not just once at "start"
                            # -- WalkerControl needs the same per-tick
                            # reapplication a vehicle's own control does (a
                            # one-time call still produced real motion in
                            # testing, but per-tick matches CARLA's standard
                            # actor-control usage).
                            cow["actor"].apply_control(carla.WalkerControl(direction=cow["direction"], speed=cow["speed"]))

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
        cattle_actors = [cow["actor"] for event in self.events for cow in event["cattle"]]
        occluders = [event["occluder"] for event in self.events if event["occluder"]]
        common.print_ground_truth(frame, self.ego, {"CATTLE": cattle_actors, "OCCLUDER": occluders})

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
        self._update_events(ego_loc)

    def _on_collision(self, event) -> None:
        self.collisions.append({"frame": int(event.frame), "other": event.other_actor.type_id})
        print(f"[COLLISION] frame={event.frame} other={event.other_actor.type_id}", flush=True)

    def finish(self) -> None:
        common.LOG_DIR.mkdir(exist_ok=True)
        with (common.LOG_DIR / "cattle_crossing_trajectory.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "t", "x", "y", "z"])
            writer.writeheader()
            writer.writerows(self.trajectory)
        actors = [self.rgb, self.lidar, self.radar, self.collision_sensor, self.ego]
        for event in self.events:
            actors += [cow["actor"] for cow in event["cattle"]]
            if event["occluder"]:
                actors.append(event["occluder"])
        actors += self.static_clutter
        common.teardown(self.world, self.tm, actors, self.controllers)
        if self.uploader:
            self.uploader.close()
        print(
            f"[DONE] collisions={len(self.collisions)} frames={self.frame_count} "
            f"events_triggered={sum(1 for e in self.events if e['triggered'])}/{len(self.events)}", flush=True,
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--ego-spawn-index", type=int, default=0)
    p.add_argument("--duration", type=float, default=45.0)  # longer run: 7 spaced-out events need more road
    p.add_argument("--m2-url", default=common.M2_URL)
    p.add_argument("--tm-port", type=int, default=common.TM_PORT)
    p.add_argument("--drive-mode", choices=["autopilot", "m5"], default="autopilot",
                   help="'autopilot': CARLA's own Traffic Manager drives the ego (default). 'm5': "
                        "M5_Pipeline's UDP decisions drive the ego via m5_control.py -- see scenario3.py's "
                        "own --drive-mode for the full rationale (single writer to the ego's control).")
    args = p.parse_args()

    scenario = CattleCrossingScenario(args.host, args.port, args.ego_spawn_index, args.m2_url, args.tm_port)
    control_log_file = None
    try:
        scenario.setup()
        if args.drive_mode == "autopilot":
            scenario.ego.set_autopilot(True, scenario.tm_port)
            scenario.tm.vehicle_percentage_speed_difference(scenario.ego, 20.0)
            scenario.tm.ignore_lights_percentage(scenario.ego, 100.0)
            for _ in range(int(args.duration / DT)):
                scenario.step()
        else:
            from m5_control import M5ControlExecutor
            executor = M5ControlExecutor()
            common.LOG_DIR.mkdir(exist_ok=True)
            control_log_file = (common.LOG_DIR / "m5_control_log_cattle_crossing.jsonl").open("w", encoding="utf-8")
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
# - --ego-spawn-index must be on a long-enough straight stretch to fit all 5
#   events (spaced up to 240m ahead) without hitting the road's end or a
#   junction; check the spawn point list in the CARLA UE4 editor/spectator
#   first, same as scenario3.py's own --ego-spawn-index guidance.
# - Start a dedicated Traffic Manager on --tm-port.
# - "container" occluders use static.prop.container/containertrash if the
#   installed CARLA build ships them; falls back to a parked truck otherwise
#   -- CARLA has no purpose-built "wall" or hedge prop, so this is the
#   closest available large, boxy occluder.
