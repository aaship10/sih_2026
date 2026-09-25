"""Scratch benchmark (not part of the pytest suite): real end-to-end
M5Engine.step() latency with StateflowFSM swapped in for BehaviorFSM.
Measures the FULL tick (TTC + now/future costmap risk + FSM + quintic
path planning), not just the isolated FSM call the earlier MATLAB-side
benchmarks measured."""
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from fsm_stateflow import StateflowFSM
from planner import M5Engine
from schema import parse_obstacle_packet


def _packet(obstacles=(), statics=(), ego_pos=(0.0, 0.0), ego_vel=(5.0, 0.0), ego_yaw_deg=0.0, frame_id=1):
    return parse_obstacle_packet({
        "timestamp": 0.0, "frame_id": frame_id,
        "ego_position": list(ego_pos), "ego_velocity": list(ego_vel), "ego_yaw_deg": ego_yaw_deg,
        "obstacles": list(obstacles), "statics": list(statics),
    })


def _obstacle(track_id, pos, vel, points, time_step_s=0.3):
    return {
        "track_id": track_id, "class": "car", "confidence": 0.9,
        "pos_x": pos[0], "pos_y": pos[1], "vel_x": vel[0], "vel_y": vel[1],
        "width": 2.0, "length": 4.5,
        "trajectories": [{
            "mode": "nominal", "probability": 1.0,
            "points": [{"x": p[0], "y": p[1], "t": i * time_step_s} for i, p in enumerate(points)],
            "sigma_m": [0.5] * len(points),
        }],
    }


def main():
    engine = M5Engine()
    t0 = time.time()
    engine.fsm = StateflowFSM()
    print("StateflowFSM init: %.2fs" % (time.time() - t0))

    n = 300
    times_ms = []
    for i in range(n):
        # Alternate obstacle distance/speed each tick so the FSM actually
        # transitions through multiple states, not a single frozen case --
        # representative of a real varying scenario, not a best-case no-op.
        ahead = 3.0 + (i % 12)
        obstacle = _obstacle(1, pos=(ahead, 0.0), vel=(-4.0, 0.0),
                              points=[(ahead - 4.0 * k * 0.3, 0.0) for k in range(16)])
        packet = _packet(obstacles=[obstacle], ego_vel=(5.0, 0.0), frame_id=i)
        t0 = time.perf_counter()
        decision = engine.step(packet)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms.sort()
    median = statistics.median(times_ms)
    p95 = times_ms[int(0.95 * n)]
    p99 = times_ms[int(0.99 * n)]
    print(f"n={n} median={median:.2f}ms p95={p95:.2f}ms p99={p99:.2f}ms max={max(times_ms):.2f}ms first={times_ms[0]:.2f}ms")
    engine.fsm.close()


if __name__ == "__main__":
    main()
