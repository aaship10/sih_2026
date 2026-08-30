"""
M2 Perception Server
=====================
Receives camera frames + telemetry from CARLA (M1), runs YOLO object
detection using best.pt, and returns detections in the standardized
JSON schema expected downstream (M3).

Run with:
    uvicorn perception_server:app --host 0.0.0.0 --port 8000
"""

import cv2
import numpy as np
import yaml
import time
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from ultralytics import YOLO

# ------------------------------------------------------------------------------
# Configuration — adjust paths/thresholds as needed
# ------------------------------------------------------------------------------
MODEL_PATH = "best.pt"
CLASSES_PATH = "classes.yaml"
CONFIDENCE_THRESHOLD = 0.01
IOU_THRESHOLD = 0.45
IMG_SIZE = 1280  # lower (e.g. 480) = faster but less accurate; must match training size ideally

# Must match the actual resolution CARLA (M1) sends. A mismatch here means the
# warm-up dummy gets letterboxed to a different internal shape than real
# frames, so cuDNN's shape-specific algorithm cache misses on the first real
# frame anyway. Update these if your camera resolution changes.
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("m2-perception")

# ------------------------------------------------------------------------------
# GPU setup
# ------------------------------------------------------------------------------
GPU_AVAILABLE = torch.cuda.is_available()
DEVICE = 0 if GPU_AVAILABLE else "cpu"      # ultralytics wants an int index or "cpu"

if GPU_AVAILABLE:
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"CUDA available — using GPU: {gpu_name}")
    torch.backends.cudnn.benchmark = True    # speeds up repeated same-size inference
else:
    logger.warning("CUDA not available — falling back to CPU (will be much slower)")

# ------------------------------------------------------------------------------
# Load model + move to GPU
# ------------------------------------------------------------------------------
logger.info(f"Loading model from {MODEL_PATH} ...")
model = YOLO(MODEL_PATH)
model.to(DEVICE)

# ------------------------------------------------------------------------------
# Dedicated single-thread executor for all inference.
# ------------------------------------------------------------------------------
inference_executor = ThreadPoolExecutor(max_workers=1)

def _run_inference(frame: np.ndarray):
    """Runs on inference_executor's single dedicated thread — never the main thread."""
    return model.predict(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE,
        device=DEVICE,
        verbose=False
        # Removed deprecated 'half' parameter
    )

# ------------------------------------------------------------------------------
# Modern FastAPI Lifespan (replaces deprecated @app.on_event)
# ------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    logger.info("Preloading + warming up model on GPU...")
    # Same resolution as real CARLA frames — matches the exact shape cuDNN
    # will see in production, so the algorithm search happens now, not on
    # your first real frame.
    dummy = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
    loop = asyncio.get_running_loop()
    for _ in range(2):  # run twice — first call finds the algorithm, second confirms it's cached
        await loop.run_in_executor(inference_executor, _run_inference, dummy)
    logger.info("Warm-up complete — every real frame from here on is fast")
    
    yield  # Application is now running and accepting requests
    
    # --- Shutdown ---
    inference_executor.shutdown(wait=True)

# Initialize FastAPI with the lifespan context manager
app = FastAPI(title="M2 Perception Server", lifespan=lifespan)

# ------------------------------------------------------------------------------
# Load Classes
# ------------------------------------------------------------------------------
class_names = None
classes_file = Path(CLASSES_PATH)
if classes_file.exists():
    with open(classes_file, "r") as f:
        classes_data = yaml.safe_load(f)

    if isinstance(classes_data, dict) and "names" in classes_data:
        class_names = classes_data["names"]
    elif isinstance(classes_data, list):
        class_names = {i: name for i, name in enumerate(classes_data)}
    elif isinstance(classes_data, dict):
        class_names = classes_data
    logger.info(f"Loaded {len(class_names)} classes from {CLASSES_PATH}")
else:
    logger.warning(f"{CLASSES_PATH} not found — falling back to model's built-in names")
    class_names = model.names  # Ultralytics models carry names internally


def get_class_name(class_id: int) -> str:
    """Resolve a class index to its human-readable name."""
    try:
        return str(class_names[class_id])
    except (KeyError, IndexError):
        return f"unknown_{class_id}"


# ------------------------------------------------------------------------------
# Main endpoint
# ------------------------------------------------------------------------------
@app.post("/api/v1/perception")
async def process_sensor_stream(
    sensor_id: str = Form(...),
    frame_id: int = Form(...),
    timestamp: float = Form(...),
    ego_speed_mps: float = Form(...),
    image: UploadFile = File(...)
):
    request_start = time.perf_counter()

    # 1. Decode incoming image
    image_bytes = await image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    height, width = frame.shape[:2]
    decode_time_ms = (time.perf_counter() - request_start) * 1000

    # 2. Run inference
    inference_start = time.perf_counter()
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(inference_executor, _run_inference, frame)
    inference_time_ms = (time.perf_counter() - inference_start) * 1000

    # 3. Parse detections
    detections = []
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                detections.append({
                    "class": get_class_name(cls_id),
                    "confidence": round(conf, 4),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
                })

    # 4. Build response
    response = {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "camera_id": sensor_id,
        "image_width": width,
        "image_height": height,
        "inference_time_ms": round(inference_time_ms, 2),
        "detections": detections
    }

    total_time_ms = (time.perf_counter() - request_start) * 1000

    logger.info(
        f"[frame {frame_id}] decode={decode_time_ms:.3f}ms  "
        f"inference={inference_time_ms:.3f}ms  "
        f"total={total_time_ms:.3f}ms  "
        f"detections={len(detections)}"
    )

    return response


@app.get("/health")
async def health_check():
    """Simple health check."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "gpu": torch.cuda.get_device_name(0) if GPU_AVAILABLE else "cpu (no GPU detected)"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)