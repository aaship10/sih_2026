"""M4's own packet recorder (section 38: M3's own log, logs/m3_tracks.jsonl,
omits ego_velocity and cannot be relied on as a replay source -- M4 must
record its own input). Two JSONL files:

  logs/m4_input_packets.jsonl  -- the RAW JSON M3 sent, byte-for-byte as
                                   received (before any pydantic parsing),
                                   so a schema bug in M4 can never corrupt
                                   the recording. This is the replay source
                                   (replay.py reads it back).
  logs/m4_predictions.jsonl    -- M4's own output, one record per frame, for
                                   offline evaluation (evaluation/metrics.py)
                                   and visualization (visualization/) without
                                   needing a live M5 or a live CARLA run.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import config


class JsonlRecorder:
    def __init__(self, filename: str) -> None:
        log_dir = Path(config.LOG_DIR)
        log_dir.mkdir(exist_ok=True)
        self._path = log_dir / filename
        self._file = self._path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
