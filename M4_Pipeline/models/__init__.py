"""Motion models: turn a track's current state (+ history) into a predicted
future Trajectory. Each model is a plain function, not a class -- there's no
shared state between calls, so a class hierarchy would just be ceremony.

models.trajectory.Trajectory is the shared output type every model
(constant_velocity, constant_acceleration) and every downstream consumer
(multimodal, output) uses -- defined once here to avoid every module
inventing its own (times, points) tuple shape.
"""
from models.trajectory import Trajectory, sample_times

__all__ = ["Trajectory", "sample_times"]
