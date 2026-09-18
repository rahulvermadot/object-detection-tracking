"""
All tunable parameters for the detection + tracking + counting pipeline.
Centralizing these here means you never have to hunt through the pipeline
code to adjust behaviour -- just change a number here and re-run.
"""

# --- Model ---
# Any Ultralytics-compatible weights file works: yolov8n/s/m/l/x.pt, yolo11n.pt,
# yolo26n.pt, or a path to your own fine-tuned .pt file in models/.
# 'n' (nano) models are fastest and are what this project defaults to, so it
# runs smoothly on a laptop CPU with no GPU required.
MODEL_PATH = "yolov8n.pt"

# Minimum confidence for a detection to be kept at all.
CONFIDENCE_THRESHOLD = 0.4

# IoU threshold used by YOLO's own non-max suppression (merges overlapping
# boxes for the same object into one).
IOU_THRESHOLD = 0.45

# Only these COCO class names are detected/tracked/counted. Set to None to
# keep every one of YOLO's 80 COCO classes instead.
TARGET_CLASSES = ["person", "car", "cell phone", "bottle"]

# Device to run inference on: "cpu", "cuda" (NVIDIA GPU), or "mps" (Apple
# Silicon). "cpu" is the safe default -- it always works, just slower.
DEVICE = "cpu"

# --- Tracker (centroid-based, IoU-assisted) ---
# How many consecutive frames a tracked object can go undetected before its
# ID is dropped (handles brief occlusion / a missed detection frame).
MAX_DISAPPEARED_FRAMES = 25

# Two boxes across consecutive frames are considered the same object if
# their IoU is at least this high, OR (as a fallback for fast motion) their
# centroids are within MAX_CENTROID_DISTANCE pixels of each other.
MIN_IOU_FOR_MATCH = 0.3
MAX_CENTROID_DISTANCE = 120  # pixels, measured on the resized inference frame

# --- Counting line ---
# The line is defined as a fraction of the frame's height/width (0.0-1.0) so
# it scales automatically to any input resolution. "orientation" is
# "horizontal" (line spans left-right, counts up/down crossings) or
# "vertical" (line spans top-bottom, counts left/right crossings).
COUNT_LINE_ORIENTATION = "horizontal"
COUNT_LINE_POSITION = 0.6  # 60% of the way down the frame

# --- Preprocessing ---
# Frames are resized so the longer side equals this many pixels before
# running detection -- keeps inference speed predictable regardless of the
# source resolution. The annotated output video is still written at the
# original resolution.
RESIZE_MAX_DIM = 960

# --- Output ---
OUTPUT_VIDEO_FPS = None  # None = keep the source video's own FPS
COUNTS_LOG_FILENAME = "counts_log.csv"
