"""Offline bird's-eye-view animation of M4's predictions, read from
logs/m4_predictions.jsonl (written by m4_server.py or replay.py --out).
Mirrors M3_Pipeline/visualize_tracks.py's structure and CLASS_COLORS
mapping (section 48: current position, history isn't stored here so only
the predicted trajectories are drawn, multiple predicted trajectories,
uncertainty ellipses, ego vehicle, prediction confidence).

Usage: python visualization/visualize_predictions.py [path/to/m4_predictions.jsonl]
"""
from __future__ import annotations

import json
import sys

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt

CLASS_COLORS = {
    "pedestrian": "red", "animal": "brown", "car": "blue", "truck": "navy",
    "bus": "purple", "motorcycle": "orange", "bicycle": "gold",
    "rickshaw": "green", "tempo": "teal", "unknown": "gray",
    "road_sign": "slategray", "traffic_signal": "yellow",
    "speed_bumps": "peru", "traffic_cones": "darkorange", "pothole": "saddlebrown",
}

MODE_LINESTYLES = {"nominal": "-", "stop": ":", "lateral": "--"}


def load_records(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def animate(path: str) -> None:
    records = load_records(path)
    if not records:
        print(f"No records found in {path}")
        return
    fig, ax = plt.subplots(figsize=(8, 8))

    def draw(frame_idx: int):
        ax.clear()
        rec = records[frame_idx]
        ego_x, ego_y = rec["ego_position"][0], rec["ego_position"][1]
        ax.plot(ego_x, ego_y, marker="s", color="black", markersize=10, label="ego")

        for pred in rec["predictions"]:
            color = CLASS_COLORS.get(pred["class_name"], "gray")
            for traj in pred["trajectories"]:
                if traj["probability"] < 0.05:
                    continue
                pts = traj["points"]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, linestyle=MODE_LINESTYLES.get(traj["mode"], "-"),
                         color=color, alpha=max(0.2, traj["probability"]), linewidth=1.5)
                # uncertainty ellipse at the final predicted point
                if traj["along_sigma_m"] and traj["lateral_sigma_m"]:
                    a = traj["along_sigma_m"][-1] * 2  # 2-sigma
                    b = traj["lateral_sigma_m"][-1] * 2
                    import math
                    angle_deg = math.degrees(traj["heading_rad"])
                    ell = patches.Ellipse((xs[-1], ys[-1]), width=a * 2, height=b * 2,
                                           angle=angle_deg, facecolor=color, alpha=0.08, edgecolor=color)
                    ax.add_patch(ell)
            # current position marker
            p0 = pred["trajectories"][0]["points"][0]
            ax.plot(p0[0], p0[1], marker="o", color=color, markersize=8)
            ax.annotate(f"{pred['track_id']}:{pred['class_name']} conf={pred['confidence']:.2f}",
                        (p0[0], p0[1]), fontsize=7)

        ax.set_xlim(ego_x - 60, ego_x + 60)
        ax.set_ylim(ego_y - 60, ego_y + 60)
        ax.set_title(f"frame {rec['frame_id']}  t={rec['timestamp']:.2f}s")
        ax.set_aspect("equal")

    anim = animation.FuncAnimation(fig, draw, frames=len(records), interval=100, repeat=True)
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "logs/m4_predictions.jsonl"
    animate(path)
