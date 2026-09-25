"""M4 -> M5 sender, UDP (see M5_Pipeline/README.md for why: M5's own design
fixes UDP:5004 as its input transport, replacing the earlier proposed
HTTP/port-10000 contract this pipeline had before M5 existed).

A UDP `sendto()` on a connectionless socket is a single non-blocking
syscall -- no connection, no response, no await -- so unlike the old HTTP
sender (and M2->M3, M3->M4's own HTTP fire-and-forget background tasks)
this needs no asyncio.create_task wrapping: calling it inline from the
request handler adds no measurable latency and there is nothing to await
or clean up at shutdown.

CHUNKING: a single UDP datagram cannot exceed 65507 bytes (65535 - 20-byte
IP header - 8-byte UDP header) -- a hard protocol limit, not a tunable
buffer size. Checked against real recorded data (M4_Pipeline/logs/
m4_predictions.jsonl, the dense "market" scenario this project already
uses for evaluation, ~29 objects/frame): a single frame's full obstacle
packet runs ~140KB, more than double that limit. Sending it as one
datagram would not raise here (each PIECE is under the limit) if it were
naively truncated, but sendto() itself raises OSError on an oversized
payload -- so without this, M4->M5 delivery would silently fail on every
single frame of exactly the scenario this pipeline is evaluated against.
Fixed by splitting `obstacles` across multiple datagrams sharing one
frame_id (`chunk_index`/`chunk_count` envelope fields); M5_Pipeline's own
m5_server.py implements the matching reassembly independently (not a
shared import -- same reason M4 keeps its own copy of M3's ground-truth
parser: each pipeline stays runnable without the others present)."""
from __future__ import annotations

import json
import logging
import socket

import numpy as np

import config

LOGGER = logging.getLogger("m5_udp_sender")

_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Safety margin under the 65507-byte hard limit for JSON overhead
# (quoting, the chunk_index/chunk_count fields themselves, one more
# obstacle pushing a chunk slightly over the running estimate).
CHUNK_SAFE_BYTES = 60000


def _json_default(obj):
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _pack_chunks(obstacle_packet: dict) -> list[dict]:
    header = {k: v for k, v in obstacle_packet.items() if k not in ("obstacles", "statics")}
    obstacles = obstacle_packet["obstacles"]
    statics = obstacle_packet["statics"]

    chunks: list[dict] = []
    current_obstacles: list[dict] = []
    # statics never move and are cheap (no trajectories) -- always sent
    # whole in the first chunk rather than split, simpler reassembly.
    current_size = len(json.dumps({**header, "obstacles": [], "statics": statics}, default=_json_default))

    for obs in obstacles:
        obs_size = len(json.dumps(obs, default=_json_default))
        if current_obstacles and current_size + obs_size > CHUNK_SAFE_BYTES:
            chunks.append({**header, "obstacles": current_obstacles, "statics": statics if not chunks else []})
            current_obstacles, current_size = [], len(json.dumps({**header, "obstacles": [], "statics": []}, default=_json_default))
        current_obstacles.append(obs)
        current_size += obs_size

    chunks.append({**header, "obstacles": current_obstacles, "statics": statics if not chunks else []})
    return chunks


def send_to_m5(obstacle_packet: dict) -> None:
    try:
        chunks = _pack_chunks(obstacle_packet)
        for i, chunk in enumerate(chunks):
            chunk["chunk_index"] = i
            chunk["chunk_count"] = len(chunks)
            body = json.dumps(chunk, default=_json_default).encode("utf-8")
            _sock.sendto(body, (config.M5_UDP_HOST, config.M5_UDP_PORT))
    except OSError as exc:
        LOGGER.warning("Could not send UDP packet to M5 (frame %s): %s", obstacle_packet.get("frame_id"), exc)
