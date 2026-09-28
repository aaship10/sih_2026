"""Tests for packet ordering and the M3-wire-format adapter (section 7.3:
M4 must drop out-of-order/duplicate packets and never let an older frame
overwrite a newer one)."""
from tracking_input.adapter import PacketOrderer
from tracking_input.schema import FramePacket


def _packet(frame_id: int, timestamp: float) -> FramePacket:
    return FramePacket(frame_id=frame_id, timestamp=timestamp, ego_position=[0, 0, 0],
                        ego_velocity=[0, 0, 0], ego_yaw_deg=0.0, tracked_objects=[])


def test_orderer_accepts_increasing_timestamps():
    orderer = PacketOrderer()
    assert orderer.accept(_packet(1, 0.05)) is not None
    assert orderer.accept(_packet(2, 0.10)) is not None
    assert orderer.dropped_count == 0


def test_orderer_drops_out_of_order_packet():
    orderer = PacketOrderer()
    assert orderer.accept(_packet(2, 0.10)) is not None
    assert orderer.accept(_packet(1, 0.05)) is None  # older -- must be dropped
    assert orderer.dropped_count == 1


def test_orderer_drops_exact_duplicate():
    orderer = PacketOrderer()
    assert orderer.accept(_packet(5, 0.25)) is not None
    assert orderer.accept(_packet(5, 0.25)) is None
    assert orderer.dropped_count == 1


def test_class_alias_round_trips():
    """M3 sends the JSON key "class" -- confirm the pydantic alias in
    tracking_input/schema.py accepts it without a KeyError/ValidationError."""
    from tracking_input.schema import TrackedObjectPacket

    raw = {
        "track_id": 1, "class": "pedestrian", "position": [1, 2, 0],
        "position_ego_relative": [1, 2, 0], "velocity": [0, 0, 0], "speed_mps": 0.0,
        "heading_deg": None, "size": [0.5, 0.5, 1.7], "confidence": 0.9, "age": 5,
        "hits": 5, "time_since_update": 0.0, "last_seen": 1.0, "sensor_sources": ["camera", "lidar"],
        "timestamp": 1.0, "track_status": "confirmed", "is_static": False,
    }
    obj = TrackedObjectPacket.model_validate(raw)
    assert obj.class_name == "pedestrian"
