"""Frame-level preprocessing: resizing for consistent, fast inference."""

import cv2


def resize_for_inference(frame, max_dim):
    """
    Resize `frame` so its longer side equals `max_dim`, preserving aspect
    ratio. Returns (resized_frame, scale) where scale is how much the frame
    was shrunk/grown -- divide detection box coordinates by `scale` to map
    them back onto the original frame.
    """
    h, w = frame.shape[:2]
    longer_side = max(h, w)
    scale = max_dim / float(longer_side)
    resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return resized, scale


def scale_boxes(boxes, scale):
    """Map a list of [x1, y1, x2, y2] boxes from the resized inference frame
    back to the original frame's coordinate space."""
    if scale == 1.0:
        return boxes
    inv = 1.0 / scale
    return [[coord * inv for coord in box] for box in boxes]
