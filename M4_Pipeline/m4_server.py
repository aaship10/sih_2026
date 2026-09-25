"""M4 receiving server: a drop-in replacement for
M3_Pipeline/m4_dummy_server.py, same URL/port/endpoint contract
(POST /api/v1/tracks, port 9500), but it actually runs motion prediction
(predictor.py) instead of just logging a count, and forwards the result on
to M5 (output/m5_udp_sender.py) instead of doing nothing with it.

Every request: record the raw packet verbatim (recorder.py, BEFORE any
parsing, so a schema bug here can never corrupt the recording M4 relies on
for replay) -> parse -> run the predictor core -> record the prediction ->
forward to M5 over UDP (output/m5_udp_sender.py -- a single non-blocking
`sendto()`, no background task needed, see that module's docstring).

Run from inside this directory: `python m4_server.py` (listens on port 9500).
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

import config
from output.m5_udp_schema import build_obstacle_packet
from output.m5_udp_sender import send_to_m5
from predictor import Predictor
from recorder import JsonlRecorder
from tracking_input.schema import FramePacket

LOGGER = logging.getLogger("m4_server")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


class M4Runtime:
    def __init__(self) -> None:
        self.predictor = Predictor()
        self.input_recorder = JsonlRecorder(config.INPUT_PACKETS_LOG_FILE)
        self.prediction_recorder = JsonlRecorder(config.PREDICTIONS_LOG_FILE)
        self.lock = asyncio.Lock()

    def close(self) -> None:
        self.input_recorder.close()
        self.prediction_recorder.close()


runtime = M4Runtime()

app = FastAPI(title="M4 Motion Prediction Server", on_shutdown=[lambda: runtime.close()])


@app.middleware("http")
async def log_request_time(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = str(time.perf_counter() - start_time)
    return response


@app.post("/api/v1/tracks")
async def receive_tracks(request: Request) -> JSONResponse:
    t0 = time.perf_counter()
    raw = await request.json()
    runtime.input_recorder.write(raw)

    try:
        packet = FramePacket.model_validate(raw)
    except ValidationError as exc:
        LOGGER.warning("Rejecting malformed FramePacket: %s", exc)
        return JSONResponse(status_code=400, content={"status": "invalid", "detail": str(exc)})

    async with runtime.lock:
        prediction_packet = runtime.predictor.process_frame_packet(packet)

    if prediction_packet is None:
        LOGGER.info("[M4] frame=%s dropped (out of order / duplicate)", packet.frame_id)
        return JSONResponse(content={"status": "dropped_out_of_order", "frame_id": packet.frame_id})

    prediction_packet_dict = prediction_packet.model_dump()
    runtime.prediction_recorder.write(prediction_packet_dict)

    send_to_m5(build_obstacle_packet(raw, prediction_packet_dict))

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    LOGGER.info(
        "[M4] frame=%s objects=%d predictions=%d latency=%.2fms",
        packet.frame_id, len(packet.tracked_objects), len(prediction_packet.predictions), elapsed_ms,
    )

    return JSONResponse(content={
        "status": "received",
        "frame_id": packet.frame_id,
        "count": len(packet.tracked_objects),
        "predictions": len(prediction_packet.predictions),
        "latency_ms": elapsed_ms,
    })


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(content={
        "status": "ok",
        "service": "m4-motion-prediction",
        "frames_processed": runtime.predictor.frames_processed,
        "tracked_histories": len(runtime.predictor.history),
        "packets_dropped_out_of_order": runtime.predictor.orderer.dropped_count,
        "m5_udp_target": f"{config.M5_UDP_HOST}:{config.M5_UDP_PORT}",
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "m4_server:app",
        host=os.getenv("HOST", config.HOST),
        port=int(os.getenv("PORT", str(config.PORT))),
        workers=1,
    )
