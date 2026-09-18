"""
Entry point for the real-time object detection, tracking, and counting
pipeline.

Usage:
    python main.py --input 0 --output output/result.mp4
    python main.py --input sample_data/traffic.mp4 --output output/result.mp4 --no-display
"""

import argparse
import csv
import os
import time

import cv2

from src import config
from src.detector import ObjectDetector
from src.tracker import CentroidTracker
from src.counter import LineCounter
from src.preprocessing import resize_for_inference, scale_boxes
from src.visualizer import draw_detections, draw_counting_line, draw_counts_panel


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                         help="Webcam index (e.g. 0) or path to a video file.")
    parser.add_argument("--output", default="output/result.mp4",
                         help="Path to write the annotated output video.")
    parser.add_argument("--no-display", action="store_true",
                         help="Don't open a live preview window (required on headless machines).")
    parser.add_argument("--model", default=None,
                         help="Override the YOLO weights file set in src/config.py.")
    return parser.parse_args()


def open_source(input_arg):
    # A bare integer string ("0", "1") means a webcam index; anything else
    # is treated as a video file path.
    source = int(input_arg) if input_arg.isdigit() else input_arg
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {input_arg}")
    return cap


def main():
    args = parse_args()

    cap = open_source(args.input)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = config.OUTPUT_VIDEO_FPS or cap.get(cv2.CAP_PROP_FPS) or 30.0

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h)
    )

    detector = ObjectDetector(model_path=args.model)
    tracker = CentroidTracker()
    counter = LineCounter(frame_w, frame_h)

    frame_count = 0
    start_time = time.time()

    print(f"Running on: {args.input}  ->  {args.output}")
    print(f"Tracking classes: {detector.target_classes}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1

            resized, scale = resize_for_inference(frame, config.RESIZE_MAX_DIM)
            detections = detector.detect(resized)
            for det in detections:
                det["box"] = scale_boxes([det["box"]], scale)[0]

            tracked = tracker.update(detections)
            counter.update(tracked)

            frame = draw_detections(frame, tracked)
            frame = draw_counting_line(frame, counter)
            frame = draw_counts_panel(frame, counter)

            writer.write(frame)

            if not args.no_display:
                cv2.imshow("Object Detection & Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0.0
    print(f"Processed {frame_count} frames in {elapsed:.1f}s ({avg_fps:.1f} FPS)")
    print(f"IN: {counter.in_count}   OUT: {counter.out_count}")

    _write_counts_log(args.output, counter)


def _write_counts_log(output_video_path, counter):
    log_path = os.path.join(
        os.path.dirname(output_video_path) or ".", config.COUNTS_LOG_FILENAME
    )
    with open(log_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["class_name", "in_count", "out_count"])
        for class_name, counts in counter.per_class_counts.items():
            csv_writer.writerow([class_name, counts["in"], counts["out"]])
        csv_writer.writerow(["TOTAL", counter.in_count, counter.out_count])
    print(f"Counts log written to {log_path}")


if __name__ == "__main__":
    main()
