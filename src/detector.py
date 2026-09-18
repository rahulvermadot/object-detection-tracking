"""
Thin wrapper around Ultralytics YOLO so the rest of the pipeline only ever
deals with plain lists of dicts, never with Ultralytics' own Results
objects. This keeps main.py, the tracker, and the counter fully decoupled
from which specific YOLO version or model file is loaded.
"""

from ultralytics import YOLO

from . import config


class ObjectDetector:
    def __init__(self, model_path=None, target_classes=None, confidence=None,
                 iou=None, device=None):
        self.model = YOLO(model_path or config.MODEL_PATH)
        self.confidence = confidence if confidence is not None else config.CONFIDENCE_THRESHOLD
        self.iou = iou if iou is not None else config.IOU_THRESHOLD
        self.device = device or config.DEVICE
        self.target_classes = target_classes if target_classes is not None else config.TARGET_CLASSES

        # Build a name -> class_id lookup from the model's own label map, so
        # TARGET_CLASSES in config.py can list plain English names
        # ("car", "cell phone") instead of magic COCO class numbers.
        self.class_names = self.model.names  # {id: name}
        if self.target_classes is not None:
            wanted = set(self.target_classes)
            known_names = set(self.class_names.values())
            unknown = wanted - known_names
            if unknown:
                raise ValueError(
                    f"TARGET_CLASSES contains names the model doesn't know: "
                    f"{sorted(unknown)}. Valid names include: "
                    f"{sorted(known_names)[:15]} ... ({len(known_names)} total)"
                )
            self.allowed_class_ids = [
                cid for cid, name in self.class_names.items() if name in wanted
            ]
        else:
            self.allowed_class_ids = None

    def detect(self, frame):
        """
        Run detection on a single BGR frame (as read by cv2.VideoCapture).

        Returns a list of dicts:
            {"box": [x1, y1, x2, y2], "class_id": int, "class_name": str,
             "confidence": float}
        """
        results = self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            classes=self.allowed_class_ids,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "box": [float(v) for v in box.xyxy[0].tolist()],
                "class_id": cls_id,
                "class_name": self.class_names[cls_id],
                "confidence": float(box.conf[0]),
            })
        return detections
