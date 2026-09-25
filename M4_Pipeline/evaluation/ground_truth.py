"""Parses scenario3.py's (and the other M1_Pipeline scenario_*.py's, which
share common.print_ground_truth -- same line format) `[GROUND TRUTH][frame
N]` stdout blocks.

This is M4's OWN copy of the same regex M3_Pipeline/evaluation.py uses
(read directly from that file to confirm the exact pattern -- see this
module's tests), not an import of it: M4 must stay runnable without the
M3_Pipeline folder being present at some assumed relative path, and M3 is
frozen so there's no risk of the two copies drifting apart from a change on
the M3 side.

Unlike M3's own evaluation.py (which explicitly skips the EGO line: it only
scores non-ego tracked objects), M4's version KEEPS the EGO row -- M4's
metrics need the ego's own ground-truth position/velocity too, to compute
the actual future ego-object distance/TTC that predicted risk is checked
against (evaluation/metrics.py).
"""
from __future__ import annotations

import re
from collections import defaultdict

GT_FRAME_RE = re.compile(r"\[GROUND TRUTH\]\[frame (\d+)\]")
GT_ACTOR_RE = re.compile(
    r"^\s*(\w+) actor=(\d+).*?pos=\(([-\d.]+),([-\d.]+)\).*?vel=\(([-\d.]+),([-\d.]+)\)"
)


def parse_ground_truth(path: str) -> dict[int, dict]:
    """Returns {frame_id: {"ego": {...} | None, "actors": [{...}, ...]}}.
    Each actor dict: {"actor_id", "kind", "x", "y", "vx", "vy"}."""
    frames: dict[int, dict] = defaultdict(lambda: {"ego": None, "actors": []})
    current_frame = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = GT_FRAME_RE.search(line)
            if m:
                current_frame = int(m.group(1))
                continue
            m = GT_ACTOR_RE.match(line)
            if m and current_frame is not None:
                kind, actor_id, x, y, vx, vy = m.groups()
                entry = {"actor_id": int(actor_id), "kind": kind,
                          "x": float(x), "y": float(y), "vx": float(vx), "vy": float(vy)}
                if kind == "EGO":
                    frames[current_frame]["ego"] = entry
                else:
                    frames[current_frame]["actors"].append(entry)
    return dict(frames)
