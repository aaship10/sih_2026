"""
m3_server.py
=============
The REAL M3 downstream server -- replaces M3_dummy_server.py.

Receives the exact same FramePacket that perception_server.py (M2)
already POSTs to http://localhost:9000/api/v1/downstream, so
**M2's code needs ZERO changes** -- this is a drop-in replacement for
the dummy stub.

What this file actually does, per incoming frame:
    1. Decode base64 lidar/radar bytes -> numpy arrays
    2. Convert M2's camera_detections format -> M3's expected format
    3. Run the REAL M3 pipeline (unchanged files: lidar_processing.py,
       radar_processing.py, fusion.py, tracker.py) -> tracked_objects
    4. Feed tracked_objects into the REAL M4 pipeline (same logic as
       run_pipeline.py's M4 section) -> predictions
    5. Return a small JSON summary to M2 (mirrors M2's own response style)

IMPORTANT ASSUMPTIONS -- CONFIRM THESE WITH WHOEVER OWNS M1/CARLA:
--------------------------------------------------------------------
This file cannot see M1's actual sensor-capture code, so the exact
byte layout of lidar_bytes/radar_bytes is an assumption based on
CARLA's standard sensor output format. If frames get processed with
zero detections or obviously wrong positions, THIS is the first place
to check -- see LIDAR_POINT_STRIDE and RADAR_COLUMN_ORDER below.

  - LiDAR bytes: assumed to be CARLA's standard
    `sensor.lidar.ray_cast` raw_data -- a flat float32 buffer,
    4 values per point (x, y, z, intensity). This is CARLA's default
    LiDAR format and is very likely correct as-is.

  - Radar bytes: assumed to be CARLA's standard `sensor.other.radar`
    raw_data -- a flat float32 buffer, 4 values per detection, in the
    order (velocity, azimuth, altitude, depth) per CARLA's
    RadarDetection struct layout. Radar column order varies more
    across CARLA versions/community code than LiDAR does -- if radar
    positions/velocities look wrong, try reordering
    RADAR_COLUMN_ORDER below first.

  - Both are assumed to be in each SENSOR's own local frame (as CARLA
    hands them to a `sensor.listen()` callback), so this file applies
    transforms.sensor_to_ego() using the mounting offsets below.
    Update LIDAR_SENSOR_OFFSET / RADAR_SENSOR_OFFSET to match your
    actual sensor mounting position in your CARLA vehicle setup.

  - M2 currently does NOT forward the ego vehicle's real position or
    velocity (perception_server.py explicitly discards ego_speed_mps:
    `del sensor_id, ego_speed_mps`). Until that's fixed on the M2
    side, this file falls back to M4_Pipeline's generate_dummy_ego()
    for ego_state, same as run_pipeline.py does. See
    get_ego_state() below -- swapping in a real ego state later is a
    one-line change once M2 forwards it.

RUN THIS WITH:
    python m3_server.py
(listens on port 9000, same as M3_dummy_server.py, so M2's
M3_DOWNSTREAM_URL doesn't need to change)
"""

from __future__ import annotations

import base64
import logging
import os
import sys
import threading
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ============================================================
# 1. Make both M3 root and M4_Pipeline imports available
#    (same pattern as run_pipeline.py)
# ============================================================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
M4_DIR = os.path.join(ROOT_DIR, "M4_Pipeline")

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if M4_DIR not in sys.path:
    sys.path.insert(0, M4_DIR)

# M3 imports -- unchanged files, exactly as tested in main.py
from lidar_processing import process_lidar_frame, get_raw_roi_points
from radar_processing import process_radar_frame, from_carla_format
from fusion import fuse_frame
from tracker import MultiObjectTracker
from transforms import make_default_camera_intrinsics, sensor_to_ego

# M4 imports -- unchanged files, exactly as tested in run_pipeline.py
from buffer import TrackHistoryBuffer
from dummy_ego import generate_dummy_ego
from interface import predict


LOGGER = logging.getLogger("m3_server")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# ============================================================
# 2. Configuration -- CONFIRM/ADJUST these against your actual
#    CARLA sensor setup (see module docstring above)
# ============================================================

# LiDAR: CARLA's ray_cast LiDAR gives (x, y, z, intensity) per point
LIDAR_POINT_STRIDE = int(os.getenv("LIDAR_POINT_STRIDE", "4"))

# Radar: CARLA's RadarDetection struct order (velocity, azimuth, altitude, depth).
# If radar results look wrong, try changing this order first.
RADAR_COLUMN_ORDER = ("velocity", "altitude", "azimuth", "depth")

# Sensor mounting offsets relative to ego vehicle center (meters) --
# MUST match your actual CARLA sensor spawn transforms in M1's setup.
LIDAR_SENSOR_OFFSET = tuple(
    float(v) for v in os.getenv("LIDAR_SENSOR_OFFSET", "0.0,0.0,2.3").split(",")
)
RADAR_SENSOR_OFFSET = tuple(
    float(v) for v in os.getenv("RADAR_SENSOR_OFFSET", "2.0,0.0,1.2").split(",")
)

CAMERA_INTRINSICS = make_default_camera_intrinsics(
    image_width=int(os.getenv("CAMERA_IMAGE_WIDTH", "1280")),
    image_height=int(os.getenv("CAMERA_IMAGE_HEIGHT", "720")),
    fov_deg=float(os.getenv("CAMERA_FOV_DEG", "90")),
)
CAMERA_EXTRINSICS = {
    "offset": tuple(
        float(v) for v in os.getenv("CAMERA_SENSOR_OFFSET", "1.5,0.0,2.2").split(",")
    )
}

DT_FALLBACK = float(os.getenv("M3_DT_FALLBACK", "0.2"))  # used only if two frames share a timestamp
MIN_HISTORY_TO_PREDICT = int(os.getenv("MIN_HISTORY_TO_PREDICT", "2"))
HISTORY_MAX_LEN = int(os.getenv("HISTORY_MAX_LEN", "15"))


# ============================================================
# 3. Byte -> numpy decoding
# ============================================================

def decode_lidar_bytes(lidar_bytes: bytes) -> np.ndarray:
    """
    Raw LiDAR bytes -> (N, 3) numpy array of (x, y, z) in the EGO frame.

    Empty/undersized input returns an empty (0, 3) array rather than
    raising, so a frame with a temporarily-empty LiDAR buffer doesn't
    crash the whole pipeline -- lidar_processing.py's functions already
    handle empty arrays gracefully.
    """
    if not lidar_bytes:
        return np.empty((0, 3), dtype=np.float32)

    flat = np.frombuffer(lidar_bytes, dtype=np.float32)
    if flat.size % LIDAR_POINT_STRIDE != 0:
        LOGGER.warning(
            "LiDAR byte buffer size (%d floats) isn't divisible by "
            "LIDAR_POINT_STRIDE=%d -- check LIDAR_POINT_STRIDE against "
            "your actual M1 sensor format.",
            flat.size, LIDAR_POINT_STRIDE,
        )
        return np.empty((0, 3), dtype=np.float32)

    points = flat.reshape(-1, LIDAR_POINT_STRIDE)
    xyz_sensor_frame = points[:, :3]  # drop intensity (4th column)

    return sensor_to_ego(xyz_sensor_frame, LIDAR_SENSOR_OFFSET)


def decode_radar_bytes(radar_bytes: bytes) -> list[dict[str, Any]]:
    """
    Raw radar bytes -> M3's expected radar_dets format:
        [{"position": [x, y], "relative_velocity": [vx, vy]}, ...]

    Converts each raw (velocity, azimuth, altitude, depth) detection
    via radar_processing.from_carla_format(), which already handles
    the range/azimuth -> (x,y) trigonometry and applies the sensor
    mounting offset.
    """
    if not radar_bytes:
        return []

    flat = np.frombuffer(radar_bytes, dtype=np.float32)
    if flat.size % 4 != 0:
        LOGGER.warning(
            "Radar byte buffer size (%d floats) isn't divisible by 4 -- "
            "check the raw radar format against your M1 setup.",
            flat.size,
        )
        return []

    detections = flat.reshape(-1, 4)
    col = {name: idx for idx, name in enumerate(RADAR_COLUMN_ORDER)}

    radar_dets = []
    for row in detections:
        converted = from_carla_format(
            range_m=float(row[col["depth"]]),
            azimuth_rad=float(row[col["azimuth"]]),
            altitude_rad=float(row[col["altitude"]]),
            radial_velocity=float(row[col["velocity"]]),
            sensor_offset_xyz=RADAR_SENSOR_OFFSET,
        )
        radar_dets.append(converted)
    return radar_dets


def convert_camera_detections(m2_detections: list[dict[str, Any]],
                                timestamp: float, frame_id: int) -> list[dict[str, Any]]:
    """
    Converts M2's detection format:
        {"class_id": int, "class_name": str, "confidence": float, "box": [x1,y1,x2,y2]}
    into M3's fusion.py expected format:
        {"class": str, "confidence": float, "bbox": [x1,y1,x2,y2], "timestamp": float, "frame_id": int}
    """
    converted = []
    for det in m2_detections:
        converted.append({
            "class": det["class_name"],
            "confidence": det["confidence"],
            "bbox": det["box"],
            "timestamp": timestamp,
            "frame_id": frame_id,
        })
    return converted


# ============================================================
# 4. Persistent M3 + M4 runtime state
#    (tracking and history MUST persist across HTTP requests --
#    each POST is one frame, not a full run)
# ============================================================

class M3Runtime:
    """Owns the tracker, history buffer, and per-run state across frames."""

    def __init__(self) -> None:
        self.tracker = MultiObjectTracker(
            max_match_distance_m=2.0,
            min_hits_to_confirm=2,
            max_age_without_update=5,
        )
        self.history = TrackHistoryBuffer(max_len=HISTORY_MAX_LEN)
        self.last_predictions: dict[int, dict] = {}
        self.t0: float | None = None
        self.last_timestamp: float | None = None
        self.lock = threading.Lock()  # frames must be processed strictly in order

    def get_ego_state(self, timestamp: float) -> dict[str, Any]:
        """
        Falls back to M4_Pipeline's dummy ego generator, since M2 does
        not currently forward real ego position/velocity (see module
        docstring). Swap this for a real ego_state once M2's
        FramePacket includes one.
        """
        if self.t0 is None:
            self.t0 = timestamp
        ego_state = generate_dummy_ego(timestamp - self.t0)
        ego_state["timestamp"] = round(timestamp, 2)
        return ego_state

    def process_frame(self, frame_id: int, timestamp: float,
                       camera_dets: list[dict], lidar_points: np.ndarray,
                       radar_dets: list[dict]) -> dict[str, Any]:
        with self.lock:
            dt = DT_FALLBACK
            if self.last_timestamp is not None:
                dt = max(timestamp - self.last_timestamp, 1e-3)
            self.last_timestamp = timestamp

            # ---- M3: LiDAR processing ----
            lidar_clusters = process_lidar_frame(lidar_points)
            raw_roi_points = get_raw_roi_points(lidar_points)

            # ---- M3: Radar processing ----
            radar_clean = process_radar_frame(radar_dets)

            # ---- M3: Fusion ----
            fused = fuse_frame(
                camera_dets, lidar_clusters, radar_clean,
                CAMERA_INTRINSICS, CAMERA_EXTRINSICS, raw_roi_points,
            )

            # ---- M3: Tracking -> tracked_objects (UNCHANGED output format) ----
            tracked_objects = self.tracker.step(fused, dt=dt, timestamp=timestamp)

            # ---- M4: history + predictions (same logic as run_pipeline.py) ----
            self.history.update(tracked_objects)
            ego_state = self.get_ego_state(timestamp)

            frame_predictions = []
            for obj in tracked_objects:
                track_id = obj["track_id"]
                track_history = self.history.get(track_id)
                if len(track_history) < MIN_HISTORY_TO_PREDICT:
                    continue
                prediction = predict(
                    track_id=track_id,
                    cls=obj["class"],
                    track_history=track_history,
                    ego_state=ego_state,
                    confidence=obj["confidence"],
                )
                frame_predictions.append(prediction)
                self.last_predictions[track_id] = prediction

            return {
                "tracked_objects": tracked_objects,
                "predictions": frame_predictions,
            }


runtime = M3Runtime()


# ============================================================
# 5. FastAPI app -- same FramePacket contract as M3_dummy_server.py,
#    so M2's forward_to_m3() needs ZERO changes.
# ============================================================

class FramePacket(BaseModel):
    frame_id: int
    timestamp: float
    camera_detections: list = []
    lidar_bytes_b64: str
    radar_bytes_b64: str
    lidar_size_bytes: int = 0
    radar_size_bytes: int = 0
    lidar_encoding: str = "base64"
    radar_encoding: str = "base64"


@asynccontextmanager
async def lifespan(_: FastAPI):
    LOGGER.info("M3 server ready. LIDAR_SENSOR_OFFSET=%s RADAR_SENSOR_OFFSET=%s",
                LIDAR_SENSOR_OFFSET, RADAR_SENSOR_OFFSET)
    yield


app = FastAPI(title="M3 Fusion + Tracking Server", version="1.0.0", lifespan=lifespan)


@app.post("/api/v1/downstream")
async def receive_downstream_packet(packet: FramePacket) -> JSONResponse:
    LOGGER.info("[frame %s] received: detections=%d lidar_bytes=%d radar_bytes=%d",
                packet.frame_id, len(packet.camera_detections),
                packet.lidar_size_bytes, packet.radar_size_bytes)

    try:
        lidar_bytes = base64.b64decode(packet.lidar_bytes_b64)
        radar_bytes = base64.b64decode(packet.radar_bytes_b64)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("[frame %s] base64 decoding failed: %s", packet.frame_id, exc)
        raise HTTPException(status_code=400, detail="Invalid Base64 payload encoding") from exc

    try:
        lidar_points = decode_lidar_bytes(lidar_bytes)
        radar_dets = decode_radar_bytes(radar_bytes)
        camera_dets = convert_camera_detections(
            packet.camera_detections, packet.timestamp, packet.frame_id
        )
        result = runtime.process_frame(
            packet.frame_id, packet.timestamp, camera_dets, lidar_points, radar_dets
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("[frame %s] M3/M4 processing failed", packet.frame_id)
        raise HTTPException(status_code=500, detail=f"M3 processing failed: {exc}") from exc

    LOGGER.info(
        "[frame %s] tracked_objects=%d predictions=%d",
        packet.frame_id, len(result["tracked_objects"]), len(result["predictions"]),
    )

    return JSONResponse(content={
        "status": "received",
        "frame_id": packet.frame_id,
        "processed_detections": len(packet.camera_detections),
        "tracked_objects_count": len(result["tracked_objects"]),
        "predictions_count": len(result["predictions"]),
    })


@app.get("/health")
async def health_check() -> JSONResponse:
    return JSONResponse(content={
        "status": "ok",
        "service": "m3-fusion-tracking",
        "active_tracks": len(runtime.tracker.tracks),
        "lidar_point_stride": LIDAR_POINT_STRIDE,
        "lidar_sensor_offset": LIDAR_SENSOR_OFFSET,
        "radar_sensor_offset": RADAR_SENSOR_OFFSET,
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "m3_server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "9000")),
        workers=1,
    )


__all__ = ["app", "runtime", "decode_lidar_bytes", "decode_radar_bytes", "convert_camera_detections"]
