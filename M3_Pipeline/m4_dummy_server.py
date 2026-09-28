"""Placeholder M4 receiver so M3's forwarding path can be exercised end to
end before M4 (motion prediction) exists. Mirrors the pattern of
M2_Pipeline/M3_dummy_server.py. Run from inside this directory:
`python m4_dummy_server.py` (listens on port 9500).
"""
import logging
import time

from fastapi import FastAPI, Request
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("m4-dummy")

app = FastAPI(title="M4 Dummy Downstream Server")


@app.middleware("http")
async def log_request_time(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    logger.info(
        "Request %s %s processed in %.2f ms",
        request.method, request.url.path, (time.perf_counter() - start_time) * 1000.0,
    )
    return response


class TrackedObjectsPacket(BaseModel):
    frame_id: int
    timestamp: float
    ego_position: list[float]
    ego_velocity: list[float] = [0.0, 0.0, 0.0]
    ego_yaw_deg: float = 0.0
    tracked_objects: list = []


@app.post("/api/v1/tracks")
async def receive_tracks(packet: TrackedObjectsPacket):
    logger.info("[frame %s] received %d tracked objects", packet.frame_id, len(packet.tracked_objects))
    return {"status": "received", "frame_id": packet.frame_id, "count": len(packet.tracked_objects)}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "m4-dummy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9500)
