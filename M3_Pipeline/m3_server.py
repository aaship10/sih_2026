"""M3 server: sensor fusion + object tracking.

Receives the FramePacket M2 already forwards to `M3_DOWNSTREAM_URL`
(http://127.0.0.1:9000/api/v1/downstream by default -- see
M2_Pipeline/perception_server.py). This is a drop-in replacement for
M2_Pipeline/M3_dummy_server.py: same URL/port/endpoint contract, but it
actually runs the pipeline -- LiDAR/radar decoding -> sensor fusion ->
multi-object tracking -> tracked_objects -- instead of just logging byte
counts. Output is forwarded (best-effort, mirroring M2's own forwarding
pattern to M3) to M4 and logged to logs/m3_tracks.jsonl for evaluation and
offline visualization (see evaluation.py, visualize_tracks.py).

Run from inside this directory: `python m3_server.py` (listens on port 9000).
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import config
import sensor_fusion
from coordinate_transforms import ego_to_world, world_to_ego
from tracker import Tracker

LOGGER = logging.getLogger("m3_server")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

M4_DOWNSTREAM_URL = os.getenv("M4_DOWNSTREAM_URL", config.M4_DOWNSTREAM_URL)
M4_TIMEOUT_SECONDS = float(os.getenv("M4_TIMEOUT_SECONDS", str(config.M4_TIMEOUT_SECONDS)))

LOG_DIR = Path(config.LOG_DIR)
LOG_DIR.mkdir(exist_ok=True)
TRACKS_LOG_PATH = LOG_DIR / config.TRACKS_LOG_FILE


def _json_default(obj: Any):
    """Defensive safety net for json.dumps(...) below and for the M4 forward
    payload. Live testing found a raw numpy.bool_ (from
    `python_bool and numpy_bool` short-circuiting) leak through
    Track.is_static into this exact path -- json.dumps has no idea how to
    serialize it, which silently 500'd the tracks-log write for ~150
    consecutive frames and, since M4 forwarding is fire-and-forget, silently
    broke 100% of M3->M4 delivery with no visible error. The root cause is
    fixed at the source (track_types.Track.refresh_static_flag now wraps in
    bool(...)), but this stays as a second line of defense against the same
    class of bug from any future numpy leak."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class CameraDetection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    box: list[float]


class FramePacket(BaseModel):
    frame_id: int
    timestamp: float
    ego_speed_mps: float = 0.0
    ego_position: list[float]
    ego_velocity: list[float] = [0.0, 0.0, 0.0]
    ego_yaw_deg: float = 0.0
    camera_detections: list[CameraDetection] = []
    lidar_bytes_b64: str
    radar_bytes_b64: str
    lidar_size_bytes: int = 0
    radar_size_bytes: int = 0
    lidar_encoding: str = "base64"
    radar_encoding: str = "base64"


class M3Runtime:
    def __init__(self) -> None:
        self.tracker = Tracker()
        self.last_timestamp: float | None = None
        self.lock = asyncio.Lock()
        self.frames_processed = 0
        self.log_file = TRACKS_LOG_PATH.open("a", encoding="utf-8")
        # asyncio.create_task() doesn't hold a strong reference to the task
        # it returns -- without keeping one ourselves, a fire-and-forget M4
        # forward can be garbage-collected mid-flight. Tasks remove
        # themselves from this set via add_done_callback once complete.
        self.background_tasks: set[asyncio.Task] = set()

    def close(self) -> None:
        self.log_file.close()


runtime = M3Runtime()
http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(M4_TIMEOUT_SECONDS))
    yield
    if runtime.background_tasks:
        await asyncio.gather(*runtime.background_tasks, return_exceptions=True)
    await http_client.aclose()
    runtime.close()


app = FastAPI(title="M3 Sensor Fusion + Tracking Server", lifespan=lifespan)


@app.middleware("http")
async def log_request_time(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


def _to_track_dict(track, timestamp: float, ego_position: np.ndarray, ego_yaw_deg: float) -> dict[str, Any]:
    position_world = track.kf.position
    velocity_world = track.kf.velocity
    position_ego = world_to_ego(position_world.reshape(1, 3), ego_position, ego_yaw_deg)[0]
    return {
        "track_id": track.track_id,
        "class": track.class_name,
        "position": [float(v) for v in position_world],
        "position_ego_relative": [float(v) for v in position_ego],
        "velocity": [float(v) for v in velocity_world],
        "speed_mps": float(np.linalg.norm(velocity_world[:2])),
        "heading_deg": track.heading_deg,
        "size": [float(v) for v in track.size],
        "confidence": float(track.class_confidence),
        "age": track.age_frames,
        "hits": track.hits,
        "time_since_update": float(timestamp - track.last_update_time),
        "last_seen": track.last_update_time,
        "sensor_sources": sorted(track.sensor_sources),
        "timestamp": timestamp,
        "track_status": track.status,
        "is_static": track.is_static,
        # Diagnostic only -- not part of the frozen M3->M4 contract (see
        # README's "Run #4 (planned diagnostic)" note). Which association
        # pass last updated this track ("primary" | "cascade" | "spawn") and
        # that match's raw cost (squared Mahalanobis distance for "primary",
        # meters for "cascade"), so a residual rare velocity-jump case can be
        # attributed to a specific pass from the logged data alone.
        "last_match_pass": track.last_match_pass,
        "last_match_cost": track.last_match_cost,
    }


async def _forward_to_m4(payload: dict) -> None:
    """Fire-and-forget: awaited from a background task, never from the
    request handler. An earlier version awaited this inline before replying
    to M2, which chained an extra network round-trip onto the already
    latency-sensitive M1->M2->M3 critical path -- live testing flagged this
    as a contributor to the pipeline's effective throughput dropping well
    below CARLA's 20Hz tick rate. M4's outcome is logged here instead of
    being reported back through the HTTP response."""
    if http_client is None:
        return
    try:
        body = json.dumps(payload, default=_json_default).encode("utf-8")
        response = await http_client.post(
            M4_DOWNSTREAM_URL, content=body, headers={"Content-Type": "application/json"}
        )
        if response.status_code != 200:
            LOGGER.warning("M4 returned HTTP %s for frame %s", response.status_code, payload.get("frame_id"))
    except (httpx.HTTPError, OSError, TypeError) as exc:
        LOGGER.warning("Could not forward tracks to M4 (frame %s): %s", payload.get("frame_id"), exc)


@app.post("/api/v1/downstream")
async def downstream(packet: FramePacket) -> JSONResponse:
    t0 = time.perf_counter()
    async with runtime.lock:
        ego_position = np.array(packet.ego_position, dtype=float)
        ego_yaw_deg = packet.ego_yaw_deg

        try:
            lidar_bytes = base64.b64decode(packet.lidar_bytes_b64)
            radar_bytes = base64.b64decode(packet.radar_bytes_b64)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail="Invalid base64 sensor payload") from exc

        camera_detections = [d.model_dump() for d in packet.camera_detections]

        fused_ego = sensor_fusion.fuse_frame(camera_detections, lidar_bytes, radar_bytes)

        radar_mount = np.array([[config.RADAR_TRANSFORM.x, config.RADAR_TRANSFORM.y, config.RADAR_TRANSFORM.z]])
        radar_sensor_world = ego_to_world(radar_mount, ego_position, ego_yaw_deg)[0]

        detections_world = []
        for obj in fused_ego:
            position_world = ego_to_world(obj["position_ego"].reshape(1, 3), ego_position, ego_yaw_deg)[0]
            det = {**obj, "position_world": position_world}
            if "radar_target" in obj:
                det["radar_sensor_position_world"] = radar_sensor_world
            detections_world.append(det)

        dt = config.DEFAULT_DT
        if runtime.last_timestamp is not None:
            dt = max(1e-3, packet.timestamp - runtime.last_timestamp)
        runtime.last_timestamp = packet.timestamp

        runtime.tracker.predict(dt)
        runtime.tracker.update(detections_world, packet.timestamp, ego_position)
        runtime.frames_processed += 1

        tracked = [
            _to_track_dict(t, packet.timestamp, ego_position, ego_yaw_deg)
            for t in runtime.tracker.confirmed_tracks()
        ]

        log_record = {
            "frame_id": packet.frame_id,
            "timestamp": packet.timestamp,
            "ego_position": packet.ego_position,
            "ego_yaw_deg": ego_yaw_deg,
            "num_camera_detections": len(camera_detections),
            "num_fused_candidates": len(fused_ego),
            "num_tracks_total": len(runtime.tracker.tracks),
            "tracked_objects": tracked,
        }
        runtime.log_file.write(json.dumps(log_record, default=_json_default) + "\n")
        runtime.log_file.flush()

    m4_task = asyncio.create_task(_forward_to_m4({
        "frame_id": packet.frame_id,
        "timestamp": packet.timestamp,
        "ego_position": packet.ego_position,
        "ego_velocity": packet.ego_velocity,
        "ego_yaw_deg": ego_yaw_deg,
        "tracked_objects": tracked,
    }))
    runtime.background_tasks.add(m4_task)
    m4_task.add_done_callback(runtime.background_tasks.discard)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    LOGGER.info(
        "[M3] frame=%s fused=%d tracks_total=%d confirmed=%d latency=%.2fms",
        packet.frame_id, len(fused_ego), len(runtime.tracker.tracks), len(tracked), elapsed_ms,
    )

    return JSONResponse(content={
        "status": "ok",
        "frame_id": packet.frame_id,
        "processed_detections": len(camera_detections),
        "fused_candidates": len(fused_ego),
        "tracked_objects_count": len(tracked),
        "latency_ms": elapsed_ms,
    })


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(content={
        "status": "ok",
        "service": "m3-sensor-fusion-tracking",
        "frames_processed": runtime.frames_processed,
        "active_tracks": len(runtime.tracker.tracks),
        "confirmed_tracks": len(runtime.tracker.confirmed_tracks()),
        "m4_downstream_url": M4_DOWNSTREAM_URL,
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "m3_server:app",
        host=os.getenv("HOST", config.HOST),
        port=int(os.getenv("PORT", str(config.PORT))),
        workers=1,
    )
