"""
Counts tracked objects as they cross a virtual line, split into "in"
crossings (moving down or right, depending on orientation) and "out"
crossings, plus a per-class breakdown. Each object is counted at most once
by remembering it in `counted_ids` the moment it crosses.
"""

from . import config


class LineCounter:
    def __init__(self, frame_width, frame_height, orientation=None, position=None):
        self.orientation = orientation or config.COUNT_LINE_ORIENTATION
        position = position if position is not None else config.COUNT_LINE_POSITION

        if self.orientation == "horizontal":
            self.line_coord = int(frame_height * position)
        else:
            self.line_coord = int(frame_width * position)

        self.frame_width = frame_width
        self.frame_height = frame_height

        self.counted_ids = set()  # object_ids already counted, so each counts once
        self.in_count = 0
        self.out_count = 0
        self.per_class_counts = {}  # class_name -> {"in": n, "out": n}

    def _side(self, centroid):
        x, y = centroid
        coord = y if self.orientation == "horizontal" else x
        return coord - self.line_coord  # negative = before the line, positive/zero = past it

    def update(self, tracked_objects):
        """
        tracked_objects: dict of object_id -> TrackedObject, as returned by
        CentroidTracker.update(). Call this once per frame, right after the
        tracker's own update() call.
        """
        for obj in tracked_objects.values():
            if len(obj.centroid_history) < 2:
                continue
            if obj.object_id in self.counted_ids:
                continue

            prev_side = self._side(obj.centroid_history[-2])
            curr_side = self._side(obj.centroid_history[-1])

            crossed = (prev_side < 0 <= curr_side) or (prev_side >= 0 > curr_side)
            if not crossed:
                continue

            direction = "in" if curr_side >= 0 else "out"
            if direction == "in":
                self.in_count += 1
            else:
                self.out_count += 1

            bucket = self.per_class_counts.setdefault(obj.class_name, {"in": 0, "out": 0})
            bucket[direction] += 1

            self.counted_ids.add(obj.object_id)

    def line_endpoints(self):
        """Returns ((x1, y1), (x2, y2)) for drawing the counting line."""
        if self.orientation == "horizontal":
            return (0, self.line_coord), (self.frame_width, self.line_coord)
        return (self.line_coord, 0), (self.line_coord, self.frame_height)
