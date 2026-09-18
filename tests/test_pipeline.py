"""
Deterministic unit tests for the tracker and counter logic, using
hand-built bounding boxes instead of real video frames or YOLO detections.
No camera, video file, or model download is needed to run these. Run with:

    pytest tests/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tracker import CentroidTracker
from src.counter import LineCounter


def _det(box, class_name="person"):
    return {"box": box, "class_name": class_name}


def test_tracker_assigns_id_to_first_detection():
    tracker = CentroidTracker()
    tracked = tracker.update([_det([10, 10, 50, 50])])
    assert len(tracked) == 1
    assert 0 in tracked


def test_tracker_keeps_same_id_for_small_movement():
    tracker = CentroidTracker()
    tracker.update([_det([10, 10, 50, 50])])
    tracked = tracker.update([_det([15, 12, 55, 52])])  # small move, high IoU
    assert len(tracked) == 1
    assert 0 in tracked  # same ID reused, not re-registered


def test_tracker_assigns_new_id_to_unrelated_object():
    tracker = CentroidTracker()
    tracker.update([_det([10, 10, 50, 50])])
    tracked = tracker.update([
        _det([15, 12, 55, 52]),      # continuation of object 0
        _det([400, 400, 440, 440]),  # brand-new, far-away object
    ])
    assert len(tracked) == 2


def test_tracker_drops_object_after_max_disappeared():
    tracker = CentroidTracker(max_disappeared=2)
    tracker.update([_det([10, 10, 50, 50])])
    tracker.update([])  # missed frame 1
    tracker.update([])  # missed frame 2
    tracked = tracker.update([])  # missed frame 3 -- should now be dropped
    assert len(tracked) == 0


def test_counter_counts_a_downward_crossing():
    counter = LineCounter(frame_width=200, frame_height=200,
                           orientation="horizontal", position=0.5)  # line at y=100
    tracker = CentroidTracker()

    tracker.update([_det([0, 40, 20, 60])])              # centroid y=50 (above line)
    tracked = tracker.update([_det([0, 140, 20, 160])])  # centroid y=150 (below line)
    counter.update(tracked)

    assert counter.in_count == 1
    assert counter.out_count == 0


def test_counter_only_counts_each_object_once():
    counter = LineCounter(frame_width=200, frame_height=200,
                           orientation="horizontal", position=0.5)
    tracker = CentroidTracker()

    tracker.update([_det([0, 40, 20, 60])])
    tracked = tracker.update([_det([0, 140, 20, 160])])
    counter.update(tracked)
    counter.update(tracked)  # same positions again -- no new crossing

    assert counter.in_count == 1


def test_counter_tracks_per_class_breakdown():
    counter = LineCounter(frame_width=200, frame_height=200,
                           orientation="horizontal", position=0.5)
    tracker = CentroidTracker()

    tracker.update([_det([0, 40, 20, 60], class_name="car")])
    tracked = tracker.update([_det([0, 140, 20, 160], class_name="car")])
    counter.update(tracked)

    assert counter.per_class_counts["car"]["in"] == 1
