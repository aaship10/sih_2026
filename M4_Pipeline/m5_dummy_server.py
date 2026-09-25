"""Placeholder M5 receiver so M4's UDP forwarding path can be exercised
without the real M5_Pipeline engine running -- e.g. to confirm M4 is
actually sending well-formed packets before debugging M5 itself. Mirrors
this file's own earlier HTTP-dummy role, updated for M5's actual UDP:5004
contract (see output/m5_udp_schema.py). Run from inside this directory:
`python m5_dummy_server.py` (listens on UDP port 5004).
"""
from __future__ import annotations

import json
import logging
import socket

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("m5-dummy")


def main() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # See M5_Pipeline/m5_server.py's matching comment: the OS default
    # receive buffer silently drops chunks under this pipeline's real
    # packet sizes/rate -- confirmed by testing, not theoretical.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    sock.bind((config.M5_UDP_HOST, config.M5_UDP_PORT))
    logger.info("Listening for M4 obstacle packets on udp://%s:%d", config.M5_UDP_HOST, config.M5_UDP_PORT)

    while True:
        data, _addr = sock.recvfrom(65536)
        try:
            packet = json.loads(data)
        except json.JSONDecodeError as exc:
            logger.warning("Could not decode packet: %s", exc)
            continue
        obstacles = packet.get("obstacles", [])
        statics = packet.get("statics", [])
        multi_mode = sum(1 for o in obstacles if len(o.get("trajectories", [])) > 1)
        logger.info(
            "[frame %s] %d obstacles (%d multimodal), %d statics",
            packet.get("frame_id"), len(obstacles), multi_mode, len(statics),
        )


if __name__ == "__main__":
    main()
