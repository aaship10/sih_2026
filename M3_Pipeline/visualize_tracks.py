"""Offline bird's-eye-view animation of M3's tracked objects, read from
logs/m3_tracks.jsonl after (or during) a run.

Usage: python visualize_tracks.py [path/to/m3_tracks.jsonl]
"""
from __future__ import annotations

import json
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt

CLASS_COLORS = {
    "pedestrian": "red", "animal": "brown", "car": "blue", "truck": "navy",
    "bus": "purple", "motorcycle": "orange", "bicycle": "gold",
    "rickshaw": "green", "tempo": "teal", "unknown": "gray",
    # Previously missing 5/14 of M2_Pipeline/classes.yaml's taxonomy --
    # anything M2 reports in these classes silently fell back to "gray"
    # (same as "unknown"), indistinguishable from an unclassified LiDAR blob.
    "road_sign": "slategray", "traffic_signal": "yellow",
    "speed_bumps": "peru", "traffic_cones": "darkorange", "pothole": "saddlebrown",
}


def load_records(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def animate(path: str) -> None:
    records = load_records(path)
    if not records:
        print(f"No records found in {path}")
        return
    fig, ax = plt.subplots(figsize=(7, 7))

    def update(i):
        ax.clear()
        record = records[i]
        ego_x, ego_y, _ = record["ego_position"]
        ax.plot(ego_x, ego_y, marker="s", color="black", markersize=10, label="ego")
        for obj in record["tracked_objects"]:
            x, y, _ = obj["position"]
            color = CLASS_COLORS.get(obj["class"], "gray")
            ax.plot(x, y, marker="o", color=color)
            ax.annotate(f"{obj['track_id']}:{obj['class']}", (x, y), fontsize=7)
        ax.set_xlim(ego_x - 60, ego_x + 60)
        ax.set_ylim(ego_y - 60, ego_y + 60)
        ax.set_title(f"frame {record['frame_id']}  t={record['timestamp']:.2f}s")
        ax.set_aspect("equal")

    anim = animation.FuncAnimation(fig, update, frames=len(records), interval=50)
    plt.show()


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else "logs/m3_tracks.jsonl"
    animate(log_path)
