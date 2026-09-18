# Real-Time Object Detection, Tracking & Counting

A computer-vision pipeline that detects objects (people, cars, phones,
bottles, etc.) in webcam or video footage using YOLO, assigns each one a
persistent ID as it moves across frames, and counts how many cross a
virtual line -- with an annotated video overlay showing live boxes, IDs,
the counting line, and running totals.

Built as an academic project covering: deep-learning object detection
(YOLO via Ultralytics), multi-object tracking (a hand-built centroid +
IoU tracker -- not an off-the-shelf library, so the logic is fully
explainable), virtual line-crossing counting, and annotated video/CSV
output generation.

## What it does

```
Webcam / Video → Preprocessing (resize for consistent inference speed)
  → Object Detection (YOLO, filtered to chosen classes)
  → Multi-Object Tracking (IoU + centroid-distance matching -> stable IDs)
  → Line-Crossing Counting (per-object, per-class in/out totals)
  → Output Visualization (annotated video with boxes, IDs, line, counts)
```

Output includes:
- An annotated `.mp4` with bounding boxes, class + track-ID labels, the
  counting line, and a live IN/OUT totals panel
- A `counts_log.csv` with the final per-class and total in/out counts

## Detected classes (configurable)

By default, only these COCO classes are detected/tracked/counted (see
`src/config.py` to change this): **person, car, cell phone, bottle**.
Set `TARGET_CLASSES = None` in `src/config.py` to detect all 80 COCO
classes YOLO knows about instead.

## Why a hand-built tracker instead of DeepSORT/ByteTrack?

Ultralytics ships built-in trackers (`model.track()` with ByteTrack or
BoT-SORT), which are more robust in production. This project instead
implements its own lightweight `CentroidTracker` in `src/tracker.py` on
purpose: for a project report, being able to explain *exactly* how IDs
are assigned and kept stable (IoU overlap first, centroid distance as a
fallback for fast motion) is worth more than borrowing a black box. See
`PROJECT_REPORT.md` for the full design rationale, limitations, and
extension ideas -- including how to swap in ByteTrack/DeepSORT later.

## Project structure

```
object-detection-tracking/
├── main.py                        # Entry point -- runs the full pipeline
├── requirements.txt
├── src/
│   ├── config.py                   # All tunable thresholds/parameters
│   ├── preprocessing.py             # Resize + coordinate rescaling
│   ├── detector.py                  # Ultralytics YOLO wrapper
│   ├── tracker.py                   # Centroid + IoU multi-object tracker
│   ├── counter.py                   # Virtual line-crossing counter
│   └── visualizer.py                # Drawing overlays
├── tools/
│   ├── download_model.py            # Pre-cache YOLO weights before a demo
│   └── count_report.py              # Pretty-print counts_log.csv
├── tests/
│   ├── generate_sample_video.py     # Synthetic smoke-test video
│   └── test_pipeline.py             # Unit tests (pytest) -- synthetic boxes
├── sample_data/                     # Put your test videos here
├── models/                          # Cached/downloaded YOLO weights land here
└── output/                          # Annotated video + counts log
```

## 1. Prerequisites

- Python 3.9 or newer (Ultralytics supports a wide range of versions,
  including the latest Python 3.13 -- unlike some CV libraries, there's
  no narrow version pin to worry about here)
- pip
- A webcam, or a recorded video with the objects you want to detect
- ~250 MB free disk space for PyTorch + the YOLO nano weights on first run

Check your Python version:
```bash
python3 --version
```

## 2. Get the project onto your machine

If you received this as a zip, just extract it and `cd` into the folder.
If you're pushing it to your own GitHub repo instead:
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

## 3. Set up a virtual environment (recommended)

```bash
python3 -m venv venv

# macOS / Linux:
source venv/bin/activate
# Windows (PowerShell):
venv\Scripts\Activate.ps1
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

This pulls in Ultralytics (which brings PyTorch along with it), OpenCV,
and NumPy. The first `pip install` can take a few minutes since PyTorch
is a large package.

> **Headless machines:** if there's no display, install
> `opencv-python-headless` instead of `opencv-python`, and always pass
> `--no-display` to `main.py`.

> **No GPU? No problem.** `src/config.py` defaults `DEVICE = "cpu"`.
> The nano model (`yolov8n.pt`) runs at a usable frame rate on CPU alone.
> If you do have an NVIDIA GPU with CUDA set up, change it to `"cuda"`
> for a large speed boost.

## 5. Pre-download the model weights (optional but recommended)

```bash
python tools/download_model.py
```

Ultralytics will otherwise download `yolov8n.pt` (~6 MB) automatically
the first time you run `main.py` -- doing it ahead of time just means
your first real run or live demo isn't slowed down by a download.

## 6. Run it on your webcam

```bash
python main.py --input 0 --output output/result.mp4
```

- A live preview window opens showing bounding boxes, class + ID labels
  for each tracked object, the red counting line, and the IN/OUT totals
  in the top-left corner.
- Walk/move an object across the counting line to see the counts update.
- Press `q` to stop.

## 7. Run it on a recorded video instead

```bash
python main.py --input sample_data/my_video.mp4 --output output/result.mp4 --no-display
```

## 8. Quick smoke test (no camera or real footage needed)

To confirm the pipeline installs and runs correctly end-to-end (this does
**not** test detection accuracy -- a plain colored square isn't a real
object YOLO will recognize -- it just verifies nothing crashes and the
output files are produced correctly):
```bash
python tests/generate_sample_video.py
python main.py --input sample_data/synthetic_test_video.mp4 --output output/result.mp4 --no-display
```

## 9. Check the results

Inside `output/` you'll find:
- `result.mp4` -- annotated video with boxes, IDs, the counting line, and totals
- `counts_log.csv` -- final per-class and total in/out counts

Turn the CSV into a quick readable summary:
```bash
python tools/count_report.py output/counts_log.csv
```

## 10. Running the automated tests

```bash
pip install pytest
pytest tests/
```

These tests build synthetic bounding boxes by hand (no camera, video, or
YOLO model involved) to verify the tracker's ID-assignment logic and the
counter's line-crossing logic deterministically.

## 11. Tuning for your own setup

All thresholds live in `src/config.py`:
- `MODEL_PATH` -- swap `yolov8n.pt` for `yolov8s.pt`/`yolov8m.pt` (slower,
  more accurate) or a newer model line like `yolo11n.pt`/`yolo26n.pt`.
- `CONFIDENCE_THRESHOLD` -- raise it to cut down on false positives,
  lower it if real objects aren't being detected.
- `TARGET_CLASSES` -- the list of COCO class names to keep; set to `None`
  for all 80 classes.
- `MAX_DISAPPEARED_FRAMES` -- how many frames an object can go undetected
  (e.g. briefly occluded) before its track ID is dropped.
- `MIN_IOU_FOR_MATCH` / `MAX_CENTROID_DISTANCE` -- how the tracker decides
  two boxes in consecutive frames are the same object.
- `COUNT_LINE_ORIENTATION` / `COUNT_LINE_POSITION` -- where the virtual
  counting line sits and which direction it counts.
- `RESIZE_MAX_DIM` -- the inference resolution; lower it for more speed,
  raise it for better detection of small/distant objects.

## 12. Troubleshooting

- **`Could not open video source`** -- check the `--input` path, or that
  a webcam is connected (`--input 0`).
- **Very slow first run** -- Ultralytics is downloading the model weights
  and PyTorch is initializing; run `tools/download_model.py` ahead of
  time to avoid this during a live demo.
- **No objects ever detected** -- make sure the object is well-lit, large
  enough in the frame, and actually one of the `TARGET_CLASSES`; try
  lowering `CONFIDENCE_THRESHOLD` in `src/config.py`.
- **IDs keep changing for the same object (ID switching)** -- this is a
  known limitation of a simple centroid/IoU tracker under fast motion or
  heavy occlusion; see `PROJECT_REPORT.md` for mitigations and stronger
  alternatives (ByteTrack, DeepSORT).
- **Counts look wrong** -- check `COUNT_LINE_ORIENTATION`/`POSITION`
  actually crosses the path objects take; an object that never crosses
  the line is never counted.
- **Live window doesn't appear / crashes on a server** -- use
  `--no-display`, and make sure `opencv-python-headless` is installed
  instead of `opencv-python`.

## License

MIT -- see [LICENSE](LICENSE).
