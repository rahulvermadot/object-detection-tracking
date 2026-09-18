# Project Report: Real-Time Object Detection, Tracking & Counting

## 1. Abstract

This project implements an end-to-end computer-vision pipeline that
detects objects in video (webcam or file) using a pretrained YOLO
model, tracks each detected object across frames with a persistent
ID, and counts objects as they cross a user-defined virtual line. The
system is designed to run in real time on ordinary CPU hardware, using
a lightweight YOLO "nano" model and a simple, fully-explainable
tracking algorithm rather than a heavier off-the-shelf tracker.

## 2. Pipeline overview

```
Frame capture -> Resize -> YOLO detection -> Coordinate rescale
  -> Centroid/IoU tracking -> Line-crossing counting -> Overlay drawing
  -> Video/CSV output
```

1. **Frame capture.** OpenCV (`cv2.VideoCapture`) reads frames from a
   webcam or a video file.
2. **Resize.** Each frame is resized so its longer side is a fixed
   number of pixels (`RESIZE_MAX_DIM`) before detection. This bounds
   inference time regardless of the source resolution; detection boxes
   are rescaled back to the original frame's coordinates afterward for
   accurate drawing.
3. **Detection.** A YOLO model (via the Ultralytics library) runs
   inference on the resized frame, producing bounding boxes, class
   IDs, and confidence scores. Detections are filtered to a configured
   set of COCO class names (e.g. person, car, cell phone, bottle) and
   a minimum confidence.
4. **Tracking.** A custom `CentroidTracker` matches this frame's
   detections against objects tracked in the previous frame, assigning
   a stable integer ID to each. See Section 3 for the matching logic.
5. **Counting.** A `LineCounter` watches each tracked object's recent
   centroid history. When an object's centroid moves from one side of
   a configured virtual line to the other, it is counted once, split
   into "in"/"out" and broken down per class.
6. **Output.** Bounding boxes, class + ID labels, the counting line,
   and running totals are drawn onto each frame and written to an
   annotated output video; final counts are also written to a CSV log.

## 3. Tracker design rationale

Ultralytics ships built-in trackers (`model.track()`, using ByteTrack
or BoT-SORT internally), which are production-grade and handle motion
prediction, appearance embeddings, and Hungarian-algorithm assignment.
This project deliberately implements its own simpler tracker instead,
for two reasons relevant to an academic submission:

- **Explainability.** A report that says "the tracker matches boxes by
  IoU overlap, falling back to centroid distance for fast motion" is a
  complete, defensible explanation of the whole algorithm. "We called
  `model.track()`" is not -- it hides the actual assignment logic
  inside a third-party library.
- **No new dependency surface.** The tracker only needs the detections
  already produced by the detector -- no additional tracking library,
  model weights, or configuration format to learn.

**Matching algorithm (`src/tracker.py`):** for every (existing tracked
object, new detection) pair, compute Intersection-over-Union (IoU)
between their boxes and Euclidean distance between their centroids. A
pair is a *candidate* match if IoU is above `MIN_IOU_FOR_MATCH` **or**
centroid distance is below `MAX_CENTROID_DISTANCE`. Candidates are
scored (IoU weighted more heavily than inverse distance) and greedily
assigned highest-score-first, so each existing object and each new
detection is used at most once per frame. Unmatched existing objects
accumulate a "disappeared frame" counter and are dropped once it
exceeds `MAX_DISAPPEARED_FRAMES` (handling brief occlusion without
losing the ID immediately); unmatched detections become newly
registered objects with a fresh ID.

**Counting algorithm (`src/counter.py`):** for each tracked object,
compare which side of the configured line its centroid was on in the
previous frame vs. the current frame. A sign change means a crossing.
Each object ID is counted at most once (via a `counted_ids` set) to
avoid double-counting an object that lingers near the line.

## 4. Limitations

- **ID switches under heavy occlusion or fast motion.** If two objects
  of similar size cross paths, or an object moves farther between
  frames than `MAX_CENTROID_DISTANCE`, the tracker can swap or lose
  IDs. Production trackers (ByteTrack, DeepSORT) mitigate this with
  motion prediction (Kalman filters) and, for DeepSORT, learned visual
  appearance embeddings -- neither of which this tracker uses.
- **No re-identification.** If a tracked object leaves the frame
  entirely and re-enters later, it is treated as a brand-new object
  with a new ID; there is no mechanism to recognize "this is the same
  car as before."
- **Detection quality depends on the chosen model and lighting/angle.**
  The default `yolov8n.pt` ("nano") model trades some accuracy for
  speed; small, distant, partially occluded, or poorly lit objects may
  be missed or misclassified.
- **Single straight-line counting.** The counter only supports one
  straight virtual line per run. Scenarios needing multiple counting
  zones, curved boundaries, or region-based (rather than line-based)
  counting would need an extension.
- **CPU-only by default is slower than GPU inference.** Real-time
  performance on CPU depends on hardware; a busy scene with many
  objects and a larger model (e.g. `yolov8m.pt`) may fall behind live
  video frame rate.

## 5. Extension ideas

- Swap `CentroidTracker` for Ultralytics' built-in ByteTrack or
  BoT-SORT (`model.track(persist=True)`) for stronger ID stability,
  and compare ID-switch rates against the hand-built tracker as an
  experiment.
- Add multiple counting lines/zones (e.g. separate lanes) and report
  per-zone counts.
- Fine-tune YOLO on a custom class not in COCO (e.g. a specific product
  or PPE item) using Ultralytics' training API, and swap in the
  resulting weights via `MODEL_PATH`.
- Add simple speed estimation per object using centroid displacement
  over time and a real-world distance calibration.
- Push inference to GPU (`DEVICE = "cuda"`) and benchmark FPS against
  the CPU baseline as a performance comparison section in the report.

## 6. Testing approach

`tests/test_pipeline.py` verifies the tracker's ID-assignment and the
counter's crossing-detection logic using hand-built bounding box
sequences -- no camera, video file, or YOLO model is involved, so these
tests are fast, deterministic, and don't depend on what a real camera
happens to see. `tests/generate_sample_video.py` separately produces a
synthetic video (a moving colored square, not a real object) purely to
smoke-test that the full pipeline -- video I/O, resizing, the tracker,
the counter, and the video writer -- runs end-to-end without crashing.
