"""M5's own decision recorder, mirroring M4_Pipeline/recorder.py's
JsonlRecorder exactly -- same rationale: an offline replay/eval loop needs
a durable, append-only log independent of whether M1 was even listening."""
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
