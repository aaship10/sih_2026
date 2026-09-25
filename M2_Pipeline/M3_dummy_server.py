import time
import base64
import logging
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("m3-downstream")

app = FastAPI(title="M3 Dummy Downstream Server")


@app.middleware("http")
async def log_request_time(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    logger.info(
        f"Request {request.method} {request.url.path} processed in {process_time * 1000:.2f} ms"
    )
    response.headers["X-Process-Time"] = str(process_time)
    return response


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


@app.post("/api/v1/downstream")
async def receive_downstream_packet(packet: FramePacket):
    logger.info(f"[frame {packet.frame_id}] Raw packet data received from upstream server")

    try:
        lidar_data = base64.b64decode(packet.lidar_bytes_b64)
        radar_data = base64.b64decode(packet.radar_bytes_b64)
    except Exception as e:
        logger.error(f"[frame {packet.frame_id}] Base64 decoding failed: {e}")
        raise HTTPException(
            status_code=400, detail="Invalid Base64 payload encoding"
        )

    logger.info(
        f"[frame {packet.frame_id}] "
        f"timestamp={packet.timestamp:.3f} | "
        f"detections={len(packet.camera_detections)} | "
        f"lidar_bytes={len(lidar_data)} | "
        f"radar_bytes={len(radar_data)}"
    )

    return {
        "status": "received",
        "frame_id": packet.frame_id,
        "processed_detections": len(packet.camera_detections),
    }


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "m3-dummy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
