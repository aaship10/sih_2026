"""Replay recorded M3 packets (logs/m4_input_packets.jsonl, written by
m4_server.py -- NOT M3's own m3_tracks.jsonl, which omits ego_velocity, see
recorder.py's docstring) through the exact same Predictor core the live
server uses. This is the offline development/test loop (section 38): no
CARLA, no M3, no M2 needs to be running.

Usage:
    python replay.py [path/to/m4_input_packets.jsonl] [--out path/to/predictions.jsonl]

The only difference from m4_server.py's request handler is where the raw
dict comes from (a file line here, an HTTP request there) -- everything
from `FramePacket.model_validate(raw)` onward is identical code, per the
"prediction/risk core is independent of the input source" requirement.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pydantic import ValidationError

import config
from predictor import Predictor
from recorder import read_jsonl
from tracking_input.schema import FramePacket


def replay(input_path: str, output_path: str | None) -> None:
    records = read_jsonl(input_path)
    print(f"[replay] loaded {len(records)} packets from {input_path}")

    predictor = Predictor()
    out_file = open(output_path, "w", encoding="utf-8") if output_path else None

    dropped_invalid = 0
    t0 = time.perf_counter()
    for raw in records:
        try:
            packet = FramePacket.model_validate(raw)
        except ValidationError as exc:
            dropped_invalid += 1
            print(f"[replay][WARN] skipping malformed packet frame_id={raw.get('frame_id')}: {exc}", file=sys.stderr)
            continue
        prediction_packet = predictor.process_frame_packet(packet)
        if prediction_packet is None:
            continue
        if out_file:
            out_file.write(json.dumps(prediction_packet.model_dump()) + "\n")
    elapsed = time.perf_counter() - t0

    if out_file:
        out_file.close()

    print(
        f"[replay] processed {predictor.frames_processed} frames "
        f"({dropped_invalid} invalid, {predictor.orderer.dropped_count} out-of-order dropped) "
        f"in {elapsed:.2f}s ({predictor.frames_processed / elapsed:.1f} frames/s) "
        f"-- {len(predictor.history)} tracks with live history at end, "
        f"{predictor.objects_dropped_stale_tracks} track histories pruned as stale"
    )
    if output_path:
        print(f"[replay] predictions written to {output_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", nargs="?", default=str(Path(config.LOG_DIR) / config.INPUT_PACKETS_LOG_FILE))
    p.add_argument("--out", default=None, help="write predictions JSONL here (default: don't write)")
    args = p.parse_args()
    replay(args.input, args.out)


if __name__ == "__main__":
    main()
