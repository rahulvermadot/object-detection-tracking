"""Drawing helpers: bounding boxes, track IDs, the counting line, and a
running totals panel, all overlaid on the frame for the annotated output
video."""

import cv2

# A small fixed palette so the same class always gets the same color across
# the whole video, instead of random colors that flicker frame to frame.
_PALETTE = [
    (66, 135, 245), (245, 130, 66), (66, 245, 129), (245, 66, 194),
    (245, 221, 66), (147, 66, 245), (66, 245, 236), (245, 66, 66),
]


def _color_for(class_name):
    return _PALETTE[hash(class_name) % len(_PALETTE)]


def draw_detections(frame, tracked_objects):
    for obj in tracked_objects.values():
        x1, y1, x2, y2 = [int(v) for v in obj.box]
        color = _color_for(obj.class_name)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{obj.class_name} #{obj.object_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1, th + 8)
        cv2.rectangle(frame, (x1, label_y - th - 8), (x1 + tw + 4, label_y), color, -1)
        cv2.putText(frame, label, (x1 + 2, label_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_counting_line(frame, counter):
    p1, p2 = counter.line_endpoints()
    cv2.line(frame, p1, p2, (0, 0, 255), 2)
    return frame


def draw_counts_panel(frame, counter):
    panel_h = 44 + 22 * len(counter.per_class_counts)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (270, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, f"IN: {counter.in_count}   OUT: {counter.out_count}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    y = 44
    for class_name, counts in counter.per_class_counts.items():
        text = f"{class_name}: in {counts['in']} / out {counts['out']}"
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 22

    return frame
