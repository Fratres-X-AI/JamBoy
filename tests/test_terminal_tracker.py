import numpy as np

from jamboy.config import load_config
from jamboy.terminal_tracker import TerminalObjectTracker


def test_terminal_tracker_tracks_center_blob():
    cfg = load_config()
    tracker = TerminalObjectTracker(cfg)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2 = __import__("cv2")
    cv2.rectangle(frame, (100, 60), (220, 180), (220, 220, 220), -1)

    assert tracker.initialize(frame)
    track = tracker.update(frame, altitude_m=100.0)
    assert track is not None
    assert track.confidence > 0.0
    assert track.range_m > 0.0
