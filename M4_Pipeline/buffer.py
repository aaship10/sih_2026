"""
buffer.py
=========
M3 sends one snapshot per frame -- no history array (see the
integration notes at the top of the M4 doc). Track._next_id persisting
across frames is the only thing that lets us stitch snapshots into a
per-object history ourselves. This file is that rolling buffer: keyed
by track_id, appended every time a tracked_objects list arrives from
M3 (or dummy_m3_stream.py during standalone dev).

Get this wrong and every downstream file breaks silently (derive.py's
finite-difference acceleration, multimodal.py's lateral-trend
detection, main.py's MIN_HISTORY_TO_PREDICT check) -- so it's kept
deliberately dumb: no smoothing, no interpolation, just "remember the
last N raw snapshots M3 gave us for this track_id."
"""

from collections import defaultdict, deque


class TrackHistoryBuffer:
    """
    One fixed-length deque per track_id.

    max_len bounds memory AND caps how far back derive.py/multimodal.py
    can look. 15 frames at M3's 5 Hz (dt=0.2, matching TIME_STEP in
    interface.py) is 3 seconds -- the same as M4's own prediction
    horizon, which is plenty for finite-difference acceleration and
    lateral-velocity-trend estimation without keeping an ever-growing
    history per object.
    """

    def __init__(self, max_len=15):
        self.max_len = max_len
        self._tracks = defaultdict(lambda: deque(maxlen=self.max_len))

    def update(self, tracked_objects):
        """
        Append this frame's snapshot to each object's own buffer.

        tracked_objects: M3's to_output_dict() list (verified against
        tracker.py) -- or dummy_m3_stream.py's equivalent, which is
        built to match that schema exactly. Objects M3 didn't report
        this frame (occluded/deleted) simply don't get appended; their
        existing buffered history is left untouched so a brief
        occlusion doesn't wipe out what we already know about them.
        """
        for obj in tracked_objects:
            self._tracks[obj["track_id"]].append(obj)

    def get(self, track_id):
        """Oldest -> newest snapshots for this track, or [] if unseen."""
        return list(self._tracks.get(track_id, []))

    def known_track_ids(self):
        """All track_ids ever seen (including ones no longer updating)."""
        return list(self._tracks.keys())