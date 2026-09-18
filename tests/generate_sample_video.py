"""
Generates a short synthetic test video: plain frames with a moving colored
square. This does NOT test detection accuracy -- YOLO won't recognize a
painted square as a real object, so expect zero detections on it. It exists
purely as a smoke test to confirm the pipeline (video I/O, resizing,
tracker, counter, video writer) runs end-to-end without crashing, without
needing a real camera, footage, or even the YOLO model to find anything.
"""

import cv2
import numpy as np

OUTPUT_PATH = "sample_data/synthetic_test_video.mp4"
WIDTH, HEIGHT, FPS, DURATION_SEC = 640, 480, 20, 4


def main():
    writer = cv2.VideoWriter(
        OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    total_frames = FPS * DURATION_SEC
    for i in range(total_frames):
        frame = np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8)
        x = int((i / total_frames) * (WIDTH - 80))
        cv2.rectangle(frame, (x, 200), (x + 80, 280), (60, 180, 220), -1)
        writer.write(frame)
    writer.release()
    print(f"Wrote {total_frames} frames to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
