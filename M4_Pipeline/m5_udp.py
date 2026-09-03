"""M4 -> M5 UDP broadcaster.

Sends a compact, JSON-encoded UDP packet containing only the information
M5 needs for local prediction-aware planning:

{
    "timestamp": <float>,
    "obstacles": [
        {
            "track_id": <int>,
            "class": <str>,
            "confidence": <float>,
            "pos_x": <float>,
            "pos_y": <float>,
            "vel_x": <float>,
            "vel_y": <float>,
            "width": <float>,
            "length": <float>,
            "trajectories": [
                {
                    "mode": <str>,
                    "probability": <float>,
                    "points": [
                        {"x": <float>, "y": <float>, "t": <float>},
                        ...
                    ]
                }
            ]
        }
    ]
}

Risk/TTC/collision analysis is intentionally not included; it belongs
in M5.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from typing import Any

LOGGER = logging.getLogger("m4_to_m5")


class M5UDPBroadcaster:
    """Rate-limited, best-effort UDP publisher for M4 -> M5."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5004,
                 max_hz: float = 10.0) -> None:
        self.host = host
        self.port = int(port)
        self.max_hz = max(float(max_hz), 0.1)
        self._min_interval = 1.0 / self.max_hz
        self._last_send_monotonic = 0.0
        self._lock = threading.Lock()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sent = 0
        self.dropped_rate_limit = 0
        self.failed = 0

    @staticmethod
    def _round(value: Any, digits: int = 3) -> float:
        return round(float(value), digits)

    def _build_packet(
        self,
        timestamp: float,
        predictions: list[dict[str, Any]],
        ego_speed_mps: float,
        ego_position: list[float],
        ego_velocity: list[float],
        ego_yaw_deg: float,
    ) -> dict[str, Any]:
        obstacles = []
        for prediction in predictions:
            position = prediction.get("position") or [0.0, 0.0]
            velocity = prediction.get("velocity") or [0.0, 0.0]
            size = prediction.get("size") or [0.0, 0.0, 0.0]

            obstacles.append({
                "track_id": int(prediction["track_id"]),
                "class": prediction.get("class", "unknown"),
                "confidence": self._round(prediction.get("confidence", 0.0)),
                "pos_x": self._round(position[0]),
                "pos_y": self._round(position[1]),
                "vel_x": self._round(velocity[0]),
                "vel_y": self._round(velocity[1]),
                "width": self._round(size[1] if len(size) > 1 else 0.0),
                "length": self._round(size[0] if len(size) > 0 else 0.0),
                "trajectories": [
                    {
                        "mode": branch.get("mode", "unknown"),
                        "probability": self._round(branch.get("probability", 0.0)),
                        "points": [
                            {
                                "x": self._round(point.get("x", 0.0)),
                                "y": self._round(point.get("y", 0.0)),
                                "t": self._round(point.get("t", 0.0)),
                            }
                            for point in branch.get("points", [])
                        ],
                    }
                    for branch in prediction.get("trajectories", [])
                ],
            })

        return {
            "timestamp": self._round(timestamp, 3),
            "ego": {
                "speed_mps": self._round(ego_speed_mps),
                "position": [
                    self._round(ego_position[0]),
                    self._round(ego_position[1]),
                ],
                "velocity": [
                    self._round(ego_velocity[0]),
                    self._round(ego_velocity[1]),
                ],
                "yaw_deg": self._round(ego_yaw_deg),
            },
            "obstacles": obstacles,
        }

    def send(
        self,
        timestamp: float,
        predictions: list[dict[str, Any]],
        ego_speed_mps: float,
        ego_position: list[float],
        ego_velocity: list[float],
        ego_yaw_deg: float,
    ) -> bool:
        """Send the latest M4 frame unless rate-limited."""
        now = time.monotonic()
        with self._lock:
            if now - self._last_send_monotonic < self._min_interval:
                self.dropped_rate_limit += 1
                return False
            self._last_send_monotonic = now

        packet = self._build_packet(
            timestamp,
            predictions,
            ego_speed_mps,
            ego_position,
            ego_velocity,
            ego_yaw_deg,
        )
        payload = json.dumps(
            packet, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")

        try:
            self._socket.sendto(payload, (self.host, self.port))
            self.sent += 1
            LOGGER.info(
                "[M4->M5] timestamp=%.3f obstacles=%d bytes=%d -> %s:%d",
                float(timestamp), len(packet["obstacles"]), len(payload),
                self.host, self.port,
            )
            return True
        except OSError:
            self.failed += 1
            LOGGER.exception(
                "[M4->M5] UDP send failed for timestamp %.3f", float(timestamp)
            )
            return False

    def close(self) -> None:
        try:
            self._socket.close()
        except OSError:
            pass
