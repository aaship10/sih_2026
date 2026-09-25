"""The M4 prediction core, deliberately independent of where a frame came
from. `Predictor.process_frame_packet` is the ONE entry point both
m4_server.py (live HTTP) and replay.py (recorded JSONL) call -- neither
reimplements any of the logic below, only how a `FramePacket` is obtained
differs between them.

M4 deliberately does NOT compute TTC, time-to-conflict, collision
probability, or a risk score/level -- that was explicitly cut from scope
(see README.md's "Status" section); M4 hands M5 raw predicted trajectories
+ uncertainty + confidence, and stops there.

Per-frame pipeline (section 26, with the input-quality layer added in
front, per M3 README 7B's design consequence):

    raw M3 packet
        |  tracking_input.adapter.PacketOrderer.accept()  (drop out-of-order/duplicate)
        v
    Frame (EgoState + list[TrackedObject])
        |  tracking_input.quality.annotate()               (per object)
        |  tracking_input.history.TrackHistory.update()     (per object)
        v
    models.class_specific.predict_class_specific()          (nominal trajectory)
        |
        v
    uncertainty.covariance.apply_uncertainty()               (per mode)
        |
        v
    multimodal.modes.generate_modes()                        (nominal + stop + lateral)
        |
        v
    output.schema.PredictionPacket  ->  output.sender.send_to_m5()  (fire-and-forget)
"""
from __future__ import annotations

import math

import config
from models import class_specific
from multimodal import modes as multimodal_modes
from multimodal.features import compute_motion_features
from output.schema import ObjectPrediction, PredictionPacket, TrajectoryOut
from tracking_input.adapter import Frame, PacketOrderer, TrackedObject, to_frame
from tracking_input.history import TrackHistory
from tracking_input.quality import annotate as annotate_quality
from tracking_input.schema import FramePacket
from uncertainty.covariance import apply_uncertainty, normalized_spread


class Predictor:
    def __init__(self) -> None:
        self.history = TrackHistory()
        self.orderer = PacketOrderer()
        self.frames_processed = 0
        self.objects_dropped_stale_tracks = 0

    def process_frame_packet(self, packet: FramePacket) -> PredictionPacket | None:
        """Returns None if the packet was dropped (out of order / duplicate
        -- see tracking_input.adapter.PacketOrderer). This is the single
        entry point for both live and replayed input."""
        frame = self.orderer.accept(packet)
        if frame is None:
            return None
        return self.process_frame(frame)

    def process_frame(self, frame: Frame) -> PredictionPacket:
        self.objects_dropped_stale_tracks += self.history.prune(frame.timestamp)

        predictions: list[ObjectPrediction] = []
        for obj in frame.objects:
            predictions.append(self._predict_object(obj, frame))

        self.frames_processed += 1
        return PredictionPacket(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            ego_position=frame.ego.position.tolist(),
            ego_velocity=frame.ego.velocity.tolist(),
            ego_yaw_deg=math.degrees(frame.ego.heading_rad),
            predictions=predictions,
        )

    def _predict_object(self, obj: TrackedObject, frame: Frame) -> ObjectPrediction:
        annotate_quality(obj)
        self.history.update(obj, frame.timestamp)
        history = self.history.get(obj.track_id)

        # Smoothed velocity (mean over the last config.MOTION_FEATURE_WINDOW
        # MEASURED samples) replaces the single raw current-frame velocity
        # as the input to every trajectory built below -- see
        # multimodal/features.py's docstring for the live-data failure case
        # this targets (a track whose reported velocity climbed smoothly
        # from 0.81->5.79 m/s while the real object was still stationary).
        # Below MIN_SAMPLES_FOR_FUZZY measured samples, compute_motion_
        # features itself just passes obj.velocity through unchanged, so
        # this is a no-op for brand-new tracks rather than a special case
        # here.
        features = compute_motion_features(history, obj.velocity, config.MOTION_FEATURE_WINDOW)

        profile = class_specific.get_profile(obj.class_name)

        nominal = class_specific.predict_class_specific(
            obj.track_id, obj.class_name, obj.position, features.smoothed_velocity, history, obj.is_static,
        )
        apply_uncertainty(nominal, profile, features.smoothed_velocity, obj.quality, obj.heading_rad or 0.0)

        modes = multimodal_modes.generate_modes(
            obj.track_id, obj.class_name, obj.position, features.smoothed_velocity, nominal, profile,
            history, frame.ego.position, obj.is_static, features,
        )
        for m in modes:
            if m is not nominal:
                apply_uncertainty(m, profile, features.smoothed_velocity, obj.quality, obj.heading_rad or 0.0)

        # Overall prediction confidence: how much to trust the predicted
        # path itself, distinct from input_quality (how much to trust the
        # CURRENT state it was built from -- tracking_input/quality.py).
        # Confidence additionally falls with how wide the uncertainty band
        # has grown by the end of this object's own predicted horizon -- a
        # technically "confirmed, high-quality" track whose class is
        # inherently unpredictable (an animal, at 2s out) should still
        # report lower confidence than its input_quality alone would suggest.
        spread = normalized_spread(float(nominal.along_sigma[-1]), float(nominal.lateral_sigma[-1]), config.UNCERTAINTY_MAX_M)
        confidence = max(0.05, obj.quality * (1.0 - 0.6 * spread))

        return ObjectPrediction(
            track_id=obj.track_id,
            class_name=obj.class_name,
            is_static=obj.is_static,
            trajectories=[
                TrajectoryOut(
                    mode=m.mode,
                    points=m.points.tolist(),
                    probability=m.probability,
                    along_sigma_m=m.along_sigma.tolist(),
                    lateral_sigma_m=m.lateral_sigma.tolist(),
                    heading_rad=m.heading_rad,
                )
                for m in modes
            ],
            horizon_s=profile.horizon_s,
            time_step_s=profile.time_step_s,
            input_quality=obj.quality,
            confidence=confidence,
        )
