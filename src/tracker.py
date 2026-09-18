"""
A self-contained multi-object tracker -- no external tracking library
needed. Assigns each detection a persistent integer ID across frames by
matching new detections to existing tracked objects using IoU overlap
first (reliable when boxes actually overlap frame-to-frame), falling back
to centroid distance (better for small or fast-moving objects whose box
barely overlaps between frames).

This is intentionally a simpler, fully-explainable algorithm rather than
DeepSORT/ByteTrack -- useful for a project report where you need to explain
*how* the tracking works, not just cite a library that did it for you.
"""

from collections import OrderedDict

from . import config


def _centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def _distance(pt_a, pt_b):
    return ((pt_a[0] - pt_b[0]) ** 2 + (pt_a[1] - pt_b[1]) ** 2) ** 0.5


class TrackedObject:
    __slots__ = ("object_id", "box", "class_name", "centroid_history", "disappeared_frames")

    def __init__(self, object_id, box, class_name):
        self.object_id = object_id
        self.box = box
        self.class_name = class_name
        self.centroid_history = [_centroid(box)]
        self.disappeared_frames = 0


class CentroidTracker:
    def __init__(self, max_disappeared=None, min_iou=None, max_centroid_distance=None):
        self.next_object_id = 0
        self.objects = OrderedDict()  # object_id -> TrackedObject
        self.max_disappeared = max_disappeared if max_disappeared is not None else config.MAX_DISAPPEARED_FRAMES
        self.min_iou = min_iou if min_iou is not None else config.MIN_IOU_FOR_MATCH
        self.max_centroid_distance = max_centroid_distance if max_centroid_distance is not None else config.MAX_CENTROID_DISTANCE

    def _register(self, box, class_name):
        obj = TrackedObject(self.next_object_id, box, class_name)
        self.objects[self.next_object_id] = obj
        self.next_object_id += 1
        return obj

    def _deregister(self, object_id):
        del self.objects[object_id]

    def update(self, detections):
        """
        detections: list of {"box": [x1, y1, x2, y2], "class_name": str, ...}
        (extra keys are ignored, e.g. "confidence", "class_id" from detector.py)

        Returns the current dict of object_id -> TrackedObject after
        matching existing tracks to new detections, registering new
        objects, and deregistering ones that have been missing too long.
        """
        if len(detections) == 0:
            for obj in list(self.objects.values()):
                obj.disappeared_frames += 1
                if obj.disappeared_frames > self.max_disappeared:
                    self._deregister(obj.object_id)
            return self.objects

        if len(self.objects) == 0:
            for det in detections:
                self._register(det["box"], det["class_name"])
            return self.objects

        object_ids = list(self.objects.keys())
        existing_boxes = [self.objects[oid].box for oid in object_ids]
        existing_centroids = [_centroid(b) for b in existing_boxes]

        # Score every (existing, new) pair that's plausibly the same object,
        # preferring IoU overlap and using centroid distance as a tiebreaker
        # / fallback for fast motion.
        candidate_pairs = []
        for i, existing_box in enumerate(existing_boxes):
            for j, det in enumerate(detections):
                iou_score = _iou(existing_box, det["box"])
                dist = _distance(existing_centroids[i], _centroid(det["box"]))
                if iou_score >= self.min_iou or dist <= self.max_centroid_distance:
                    score = iou_score * 2.0 - (dist / max(self.max_centroid_distance, 1.0))
                    candidate_pairs.append((score, i, j))

        candidate_pairs.sort(key=lambda p: p[0], reverse=True)

        used_existing, used_new = set(), set()
        for score, i, j in candidate_pairs:
            if i in used_existing or j in used_new:
                continue
            oid = object_ids[i]
            det = detections[j]
            obj = self.objects[oid]
            obj.box = det["box"]
            obj.class_name = det["class_name"]
            obj.centroid_history.append(_centroid(det["box"]))
            if len(obj.centroid_history) > 30:
                obj.centroid_history.pop(0)
            obj.disappeared_frames = 0
            used_existing.add(i)
            used_new.add(j)

        # Existing objects with no matching detection this frame: mark as
        # disappeared, drop once they've been missing for too long.
        for i, oid in enumerate(object_ids):
            if i not in used_existing:
                obj = self.objects[oid]
                obj.disappeared_frames += 1
                if obj.disappeared_frames > self.max_disappeared:
                    self._deregister(oid)

        # Detections with no matching existing object: these are new arrivals.
        for j, det in enumerate(detections):
            if j not in used_new:
                self._register(det["box"], det["class_name"])

        return self.objects
