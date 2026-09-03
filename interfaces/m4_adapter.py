import json
import math
import socket
from typing import List

from interfaces.data_types import (
    DynamicPrediction,
    PredictedTrajectory,
    EgoState,
)


class M4Adapter:

    @staticmethod
    def parse_predictions(m4_message: dict) -> List[DynamicPrediction]:
        predictions = []

        for obstacle in m4_message.get("obstacles", []):
            track_id = int(obstacle["track_id"])

            trajectories = []

            for branch in obstacle.get("trajectories", []):
                waypoints = []

                for point in branch.get("points", []):
                    waypoints.append((
                        float(point["x"]),
                        float(point["y"]),
                        float(point["t"])
                    ))

                if waypoints:
                    trajectories.append(
                        PredictedTrajectory(
                            waypoints=waypoints,
                            probability=float(
                                branch.get("probability", 0.0)
                            )
                        )
                    )

            if trajectories:
                predictions.append(
                    DynamicPrediction(
                        track_id=track_id,
                        trajectories=trajectories
                    )
                )

        return predictions

    @staticmethod
    def parse_ego_state(m4_message: dict) -> EgoState:
        ego = m4_message.get("ego", {})

        return EgoState(
            x=0.0,
            y=0.0,
            vx=float(ego.get("speed_mps", 0.0)),
            vy=0.0,
            yaw=0.0,
            accel=0.0,
            steering=0.0,
            timestamp=float(m4_message.get("timestamp", 0.0)),
        )

    @staticmethod
    def create_receiver(
        host: str = "127.0.0.1",
        port: int = 5004,
        buffer_size: int = 65535
    ):
        """
        Create a non-blocking UDP receiver for M4 predictions.
        """

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((host, port))
        sock.setblocking(False)

        print(f"M5 listening for M4 on UDP {host}:{port}...")

        return sock

    @staticmethod
    def receive_latest(sock, buffer_size=65535):
        latest_message = None

        while True:
            try:
                data, address = sock.recvfrom(buffer_size)
            except BlockingIOError:
                break

            message = json.loads(data.decode("utf-8"))

            predictions = M4Adapter.parse_predictions(message)
            ego = M4Adapter.parse_ego_state(message)

            obstacle_count = len(message.get("obstacles", []))

            print(
                f"Received {obstacle_count} obstacles "
                f"→ parsed {len(predictions)} predictions"
            )

            latest_message = {
                "predictions": predictions,
                "ego": ego,
            }

        return latest_message