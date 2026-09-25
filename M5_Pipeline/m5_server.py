"""M5's live server: a UDP listener on M4_UDP_PORT (5004) feeding
M5Engine.step() at a fixed config.PLANNING_HZ (spec section 2: "running
continuously at 10Hz"), independent of how often M4 actually sends -- the
newest received ObstaclePacket is held and reused between arrivals, the
same "hold the last command" pattern spec section 1b asks M1's own
executor to use on ITS incoming side. Broadcasts each DecisionPacket to M1
over UDP:5006.

REASSEMBLY: M4's output/m5_udp_sender.py splits one frame's obstacle list
across multiple datagrams when it doesn't fit UDP's 65507-byte hard limit
(a dense scene, checked against real recorded data, easily exceeds it --
see that module's docstring). `_ChunkReassembler` below independently
implements the matching merge (not a shared import with M4 -- each
pipeline stays runnable without the others present, same as M4 keeping its
own copy of M3's ground-truth parser). Since this is UDP, a chunk CAN be
lost: an incomplete frame is simply dropped once a newer frame's chunks
start arriving, never blocked on forever.

Run from inside this directory: `python m5_server.py`.
"""
from __future__ import annotations

import json
import logging
import socket
import time

import config
from planner import M5Engine
from recorder import JsonlRecorder
from schema import parse_obstacle_packet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("m5_server")


class _ChunkReassembler:
    def __init__(self) -> None:
        self._pending: dict[int, list[dict | None]] = {}

    def add(self, chunk: dict) -> dict | None:
        """Returns the fully-merged packet once every chunk for its
        frame_id has arrived, else None. Drops any pending frame_id older
        than the one this chunk belongs to -- an M4 frame that never
        completes (lost chunk) must not block newer frames forever."""
        frame_id = chunk["frame_id"]
        chunk_count = chunk.get("chunk_count", 1)

        for old_frame_id in [f for f in self._pending if f < frame_id]:
            del self._pending[old_frame_id]

        if chunk_count == 1:
            return chunk

        buf = self._pending.setdefault(frame_id, [None] * chunk_count)
        buf[chunk["chunk_index"]] = chunk
        if any(c is None for c in buf):
            return None

        del self._pending[frame_id]
        merged = dict(buf[0])
        merged["obstacles"] = [o for c in buf for o in c["obstacles"]]
        merged["statics"] = [s for c in buf for s in c["statics"]]
        return merged


def _drain_latest(sock: socket.socket, reassembler: _ChunkReassembler) -> dict | None:
    """Non-blocking: returns the NEWEST fully-reassembled packet currently
    available (draining everything queued on the socket first), or None if
    nothing completed since the last check."""
    latest = None
    while True:
        try:
            data, _addr = sock.recvfrom(65536)
        except BlockingIOError:
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError as exc:
            LOGGER.warning("Could not decode packet from M4: %s", exc)
            continue
        merged = reassembler.add(chunk)
        if merged is not None:
            latest = merged
    return latest


def _drain_latest_lane_width(sock: socket.socket) -> float | None:
    """Non-blocking: returns the newest lane_width_m received since the
    last check (a single small JSON object per packet, no chunking needed
    at this size), or None if nothing arrived. Malformed packets are
    logged and skipped rather than killing the main loop -- this channel
    is a pure enhancement (see config.M1_LANE_INFO_UDP_PORT), M5 already
    runs correctly without it."""
    latest = None
    while True:
        try:
            data, _addr = sock.recvfrom(4096)
        except BlockingIOError:
            break
        try:
            latest = json.loads(data)["lane_width_m"]
        except (json.JSONDecodeError, KeyError) as exc:
            LOGGER.warning("Could not decode lane-info packet from M1: %s", exc)
    return latest


def main() -> None:
    in_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Default OS socket receive buffers are too small for several ~60KB
    # chunks (M4's output/m5_udp_sender.py) arriving back-to-back --
    # confirmed by testing: with the OS default, replaying 100 real
    # recorded frames over actual loopback UDP silently dropped chunks on
    # 28% of multi-chunk frames (never raising, never logging -- the
    # frame just never completed); 1MB eliminated every drop.
    in_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    in_sock.bind((config.M4_UDP_HOST, config.M4_UDP_PORT))
    in_sock.setblocking(False)
    out_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # SPEC.md section 4's M1->M5 map interface (see config.
    # M1_LANE_INFO_UDP_PORT) -- a separate socket/port from M4's obstacle
    # feed since it's a different sender (M1, not M4) with its own,
    # independent update cadence.
    lane_info_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    lane_info_sock.bind((config.M1_LANE_INFO_UDP_HOST, config.M1_LANE_INFO_UDP_PORT))
    lane_info_sock.setblocking(False)

    engine = M5Engine()
    recorder = JsonlRecorder(config.DECISIONS_LOG_FILE)
    reassembler = _ChunkReassembler()

    LOGGER.info("M5 listening for M4 on udp://%s:%d, broadcasting decisions to udp://%s:%d at %.0fHz",
                config.M4_UDP_HOST, config.M4_UDP_PORT, config.M1_UDP_HOST, config.M1_UDP_PORT, config.PLANNING_HZ)

    last_packet_raw: dict | None = None
    last_lane_width_m: float | None = None
    try:
        while True:
            tick_start = time.perf_counter()

            new_raw = _drain_latest(in_sock, reassembler)
            if new_raw is not None:
                last_packet_raw = new_raw

            new_lane_width = _drain_latest_lane_width(lane_info_sock)
            if new_lane_width is not None:
                last_lane_width_m = new_lane_width

            if last_packet_raw is not None:
                packet = parse_obstacle_packet(last_packet_raw)
                decision = engine.step(packet, lane_width_m=last_lane_width_m)
                body = json.dumps(decision.to_dict()).encode("utf-8")
                out_sock.sendto(body, (config.M1_UDP_HOST, config.M1_UDP_PORT))
                # diagnostics (raw TTC/risk inputs) recorded to the LOG
                # only -- never sent over UDP, keeps the M1-facing wire
                # format exactly per spec while still making "why this
                # state" inspectable after the fact.
                record = decision.to_dict()
                record["diagnostics"] = engine.last_diagnostics
                recorder.write(record)
                LOGGER.info(
                    "[frame %s] state=%s target_speed=%.2f replan=%s compute=%.2fms ttc=%s now_risk=%s future_risk=%s",
                    decision.frame_id, decision.behavior_state, decision.target_speed,
                    decision.replan_triggered, decision.computation_time_ms,
                    engine.last_diagnostics.get("min_ttc"), engine.last_diagnostics.get("now_risk"),
                    engine.last_diagnostics.get("future_risk"),
                )

            elapsed = time.perf_counter() - tick_start
            time.sleep(max(0.0, config.PLANNING_PERIOD_S - elapsed))
    except KeyboardInterrupt:
        LOGGER.info("Shutting down")
    finally:
        recorder.close()


if __name__ == "__main__":
    main()
