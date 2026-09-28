"""Unit tests for m5_server.py's _ChunkReassembler -- the receiver half of
the chunking scheme M4_Pipeline/output/m5_udp_sender.py's docstring
describes (a dense frame's obstacle packet can exceed UDP's 65507-byte
hard limit; both sides implement the same chunk_index/chunk_count envelope
independently, this file exercises M5's half of that contract)."""
from m5_server import _ChunkReassembler


def _chunk(frame_id, chunk_index, chunk_count, obstacles, statics=None):
    return {"frame_id": frame_id, "chunk_index": chunk_index, "chunk_count": chunk_count,
            "timestamp": 0.0, "obstacles": obstacles, "statics": statics or []}


def test_single_chunk_passes_through_unchanged():
    r = _ChunkReassembler()
    merged = r.add(_chunk(1, 0, 1, obstacles=[{"track_id": 1}]))
    assert merged is not None
    assert merged["obstacles"] == [{"track_id": 1}]


def test_multi_chunk_merges_only_once_complete():
    r = _ChunkReassembler()
    assert r.add(_chunk(1, 0, 2, obstacles=[{"track_id": 1}], statics=[{"track_id": 99}])) is None
    merged = r.add(_chunk(1, 1, 2, obstacles=[{"track_id": 2}]))
    assert merged is not None
    assert [o["track_id"] for o in merged["obstacles"]] == [1, 2]
    assert merged["statics"] == [{"track_id": 99}]


def test_incomplete_frame_is_dropped_once_a_newer_frame_starts():
    r = _ChunkReassembler()
    assert r.add(_chunk(1, 0, 2, obstacles=[{"track_id": 1}])) is None  # frame 1 never completes (lost chunk)
    assert r.add(_chunk(2, 0, 1, obstacles=[{"track_id": 5}])) is not None  # frame 2 is a single chunk, completes immediately
    # frame 1's leftover chunk must not silently complete or corrupt frame 2 -- it's simply gone.
    assert 1 not in r._pending


def test_out_of_order_chunk_arrival_still_merges_correctly():
    r = _ChunkReassembler()
    assert r.add(_chunk(1, 1, 2, obstacles=[{"track_id": 2}])) is None
    merged = r.add(_chunk(1, 0, 2, obstacles=[{"track_id": 1}], statics=[{"track_id": 99}]))
    assert [o["track_id"] for o in merged["obstacles"]] == [1, 2]
