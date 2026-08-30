"""M2 Perception Server with asynchronous M3 downstream forwarding."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
import torch
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO

LOGGER = logging.getLogger("perception_server")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

MODEL_PATH = Path(os.getenv("YOLO_MODEL_PATH", "best.pt"))
CLASSES_PATH = Path(os.getenv("CLASSES_YAML_PATH", "classes.yaml"))
WARMUP_IMAGE_PATH = Path(os.getenv("WARMUP_IMAGE_PATH", "test.jpg"))
M3_DOWNSTREAM_URL = os.getenv(
    "M3_DOWNSTREAM_URL", "http://localhost:9000/api/v1/downstream"
)
M3_TIMEOUT_SECONDS = float(os.getenv("M3_TIMEOUT_SECONDS", "10"))
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.05"))
YOLO_IOU = float(os.getenv("YOLO_IOU", "0.45"))
YOLO_IMAGE_SIZE = int(os.getenv("YOLO_IMAGE_SIZE", "1280"))
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "")
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(25 * 1024 * 1024)))


class PerceptionRuntime:
    """Owns the detector, class mapping, warm-up state, and inference lock."""

    def __init__(self) -> None:
        self.model: YOLO | None = None
        self.class_names: dict[int, str] = {}
        self.model_error: str | None = None
        self.warmed_up = False
        self.inference_lock = threading.Lock()

    @property
    def device(self) -> str:
        return YOLO_DEVICE or ("cuda:0" if torch.cuda.is_available() else "cpu")

    def load(self) -> None:
        """Load YOLO, safely load classes, and warm up using test.jpg."""
        try:
            self.model = YOLO(str(MODEL_PATH))
            self.class_names = load_class_names(CLASSES_PATH, self.model)
            self._warm_up()
            self.model_error = None
            LOGGER.info(
                "Loaded YOLO model from %s on %s; warm-up=%s",
                MODEL_PATH,
                self.device,
                self.warmed_up,
            )
        except Exception as exc:  # noqa: BLE001
            self.model = None
            self.model_error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("Unable to initialize YOLO model")

    def _warm_up(self) -> None:
        if self.model is None or not WARMUP_IMAGE_PATH.is_file():
            LOGGER.warning("Warm-up image not found: %s", WARMUP_IMAGE_PATH)
            return

        image = cv2.imdecode(
            np.frombuffer(WARMUP_IMAGE_PATH.read_bytes(), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            LOGGER.warning("Warm-up image could not be decoded: %s", WARMUP_IMAGE_PATH)
            return

        with self.inference_lock:
            for _ in range(max(1, int(os.getenv("WARMUP_PASSES", "2")))):
                self.model.predict(
                    source=image,
                    imgsz=YOLO_IMAGE_SIZE,
                    conf=YOLO_CONFIDENCE,
                    iou=YOLO_IOU,
                    device=self.device,
                    verbose=False,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        self.warmed_up = True

    def predict_detections(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Run one inference and return JSON-serializable detection dictionaries."""
        if self.model is None:
            raise RuntimeError(self.model_error or "YOLO model is unavailable")

        with self.inference_lock:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            results = self.model.predict(
                source=image,
                imgsz=YOLO_IMAGE_SIZE,
                conf=YOLO_CONFIDENCE,
                iou=YOLO_IOU,
                device=self.device,
                verbose=False,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        if not results or results[0].boxes is None:
            return []

        boxes = results[0].boxes
        xyxy = boxes.xyxy.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist() if boxes.conf is not None else []
        class_ids = boxes.cls.detach().cpu().tolist() if boxes.cls is not None else []

        detections: list[dict[str, Any]] = []
        for index, coordinates in enumerate(xyxy):
            class_id = int(class_ids[index]) if index < len(class_ids) else -1
            confidence = (
                float(confidences[index]) if index < len(confidences) else 0.0
            )
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": self.class_names.get(class_id, f"class_{class_id}"),
                    "confidence": confidence,
                    "box": [float(value) for value in coordinates],
                }
            )
        return detections


runtime = PerceptionRuntime()


def load_class_names(classes_path: Path, model: YOLO) -> dict[int, str]:
    """Safely load YAML names, falling back to model names or generated names."""
    names: Any = None
    if classes_path.is_file():
        try:
            with classes_path.open("r", encoding="utf-8") as stream:
                payload = yaml.safe_load(stream)
            if isinstance(payload, dict):
                names = payload.get("names", payload.get("classes"))
            elif isinstance(payload, (list, tuple)):
                names = payload
        except (OSError, yaml.YAMLError) as exc:
            LOGGER.warning("Could not load class mapping %s: %s", classes_path, exc)

    if names is None:
        names = getattr(model, "names", None)
    if isinstance(names, dict):
        normalized: dict[int, str] = {}
        for key, value in names.items():
            try:
                normalized[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        if normalized:
            return normalized
    elif isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def decode_image(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The image file is empty")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="The image file is too large")
    image = cv2.imdecode(
        np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if image is None:
        raise HTTPException(status_code=400, detail="Unable to decode the camera image")
    return image


async def forward_to_m3(
    *,
    frame_id: int,
    timestamp: float,
    camera_detections: list[dict[str, Any]],
    lidar_bytes: bytes,
    radar_bytes: bytes,
) -> bool:
    """POST a FramePacket to M3; return false for any transport or HTTP failure."""
    payload = {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "camera_detections": camera_detections,
        "lidar_bytes_b64": base64.b64encode(lidar_bytes).decode("ascii"),
        "radar_bytes_b64": base64.b64encode(radar_bytes).decode("ascii"),
        "lidar_size_bytes": len(lidar_bytes),
        "radar_size_bytes": len(radar_bytes),
        "lidar_encoding": "base64",
        "radar_encoding": "base64",
    }
    try:
        LOGGER.info("Forwarding frame %s to M3: %s", frame_id, M3_DOWNSTREAM_URL)
        timeout = httpx.Timeout(M3_TIMEOUT_SECONDS)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(M3_DOWNSTREAM_URL, json=payload)
        if response.status_code != 200:
            LOGGER.warning("M3 returned HTTP %s: %s", response.status_code, response.text[:500])
            return False
        LOGGER.info("M3 accepted frame %s with HTTP 200", frame_id)
        return True
    except (httpx.HTTPError, OSError) as exc:
        LOGGER.warning("Could not forward frame %s to M3: %s", frame_id, exc)
        return False


@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.to_thread(runtime.load)
    yield


app = FastAPI(
    title="M2 Perception Server",
    version=os.getenv("APP_VERSION", "1.0.0"),
    lifespan=lifespan,
)


@app.post("/api/v1/perception")
async def perception(
    sensor_id: str = Form(...),
    frame_id: int = Form(...),
    timestamp: float = Form(...),
    ego_speed_mps: float = Form(...),
    image: UploadFile = File(...),
    lidar_file: UploadFile = File(...),
    radar_file: UploadFile = File(...),
) -> JSONResponse:
    """Receive M1 data, detect objects, forward FramePacket to M3, and acknowledge M1."""
    del sensor_id, ego_speed_mps  # Accepted by the M1 contract; not part of FramePacket.

    image_bytes, lidar_bytes, radar_bytes = await asyncio.gather(
        image.read(), lidar_file.read(), radar_file.read()
    )
    decoded_image = decode_image(image_bytes)

    started = time.perf_counter()
    try:
        camera_detections = await asyncio.to_thread(
            runtime.predict_detections, decoded_image
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Camera inference failed") from exc

    forwarded_to_m3 = await forward_to_m3(
        frame_id=frame_id,
        timestamp=timestamp,
        camera_detections=camera_detections,
        lidar_bytes=lidar_bytes,
        radar_bytes=radar_bytes,
    )
    LOGGER.debug(
        "Frame %s processed in %.2f ms; detections=%s; forwarded_to_m3=%s",
        frame_id,
        (time.perf_counter() - started) * 1000.0,
        len(camera_detections),
        forwarded_to_m3,
    )

    return JSONResponse(
        content={
            "status": "ok",
            "frame_id": frame_id,
            "camera_detections_count": len(camera_detections),
            "lidar_bytes_received": len(lidar_bytes),
            "radar_bytes_received": len(radar_bytes),
            "forwarded_to_m3": forwarded_to_m3,
        }
    )


@app.get("/health")
async def health() -> JSONResponse:
    """Report model readiness and GPU availability."""
    model_loaded = runtime.model is not None
    return JSONResponse(
        status_code=200 if model_loaded else 503,
        content={
            "status": "ok" if model_loaded else "degraded",
            "model_loaded": model_loaded,
            "model_path": str(MODEL_PATH),
            "model_status": "ready" if model_loaded else (runtime.model_error or "unavailable"),
            "gpu_available": bool(torch.cuda.is_available()),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "warmed_up": runtime.warmed_up,
            "m3_downstream_url": M3_DOWNSTREAM_URL,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "perception_server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        workers=1,
    )


__all__ = ["app", "forward_to_m3", "load_class_names", "runtime"]
