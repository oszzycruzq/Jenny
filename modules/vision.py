"""
Jenny - Vision Module (YOLOv8 upgrade)
=======================================
Detects all 80 COCO classes with YOLOv8.
Vehicles get extra detail:
  • Type  — Sports Car, Sedan, SUV, Minivan, Truck, Pickup, Bus, Motorcycle
  • Color — Red, Blue, Green, Yellow, Orange, White, Black, Silver, Gray, Purple, Pink, Cyan
  • Shape inference — from bounding-box aspect ratio + YOLO class

Primary:  YOLOv8 (ultralytics) — auto-downloads yolov8s.pt on first run (~22 MB)
Fallback: Caffe SSD → MediaPipe face detection
"""

import os
import cv2
import numpy as np

# ── YOLOv8 ────────────────────────────────────────────────────────── #
_YOLO_OK = False
try:
    from ultralytics import YOLO
    _YOLO_OK = True
except ImportError:
    pass

# ── MediaPipe (fallback) ──────────────────────────────────────────── #
try:
    import mediapipe as mp
    _MP_OK = True
except ImportError:
    _MP_OK = False

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..")

# ── COCO class IDs we care about ──────────────────────────────────── #
_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# ── Box-draw colors per detected color name (BGR) ─────────────────── #
_BOX_COLORS = {
    "Red":      ( 30,  30, 220),
    "Orange":   (  0, 140, 255),
    "Yellow":   (  0, 220, 220),
    "Green":    ( 30, 200,  50),
    "Cyan":     (200, 200,   0),
    "Blue":     (220, 100,   0),
    "Purple":   (200,  50, 200),
    "Pink":     (180, 100, 220),
    "White":    (220, 220, 220),
    "Silver":   (190, 190, 205),
    "Gray":     (140, 140, 145),
    "Black":    ( 80,  80,  80),
}
_DEFAULT_BOX = (0, 200, 255)


class VisionModule:
    """
    Vehicle + object detection.
    Primary: YOLOv8  |  Fallback: Caffe SSD → MediaPipe.
    """

    # SSD labels (legacy fallback)
    _SSD_CLASSES = [
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
        "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
    ]

    def __init__(self, model_dir=None, yolo_model="yolov8s.pt"):
        self._model_dir = os.path.abspath(model_dir or _MODEL_DIR)
        self._net       = None
        self._mode      = None
        self._mp_face   = None
        self._yolo      = None

        if _YOLO_OK:
            self._load_yolo(yolo_model)
        else:
            print("[Vision] ultralytics not found — using legacy detector")
            self._load_legacy()

    # ── Model loading ─────────────────────────────────────────────── #

    def _load_yolo(self, model_name):
        try:
            # Weights auto-download to ~/.cache/ultralytics/ on first use
            self._yolo = YOLO(model_name, verbose=False)
            self._mode = "yolo"
            print(f"[Vision] YOLOv8 ready ({model_name})")
        except Exception as exc:
            print(f"[Vision] YOLOv8 load failed ({exc}) — falling back")
            self._load_legacy()

    def _load_legacy(self):
        face_proto  = os.path.join(self._model_dir, "deploy.prototxt")
        face_model  = os.path.join(self._model_dir,
                                    "res10_300x300_ssd_iter_140000.caffemodel")
        ssd_proto   = os.path.join(self._model_dir, "MobileNetSSD_deploy.prototxt")
        ssd_model   = os.path.join(self._model_dir, "MobileNetSSD_deploy.caffemodel")

        if os.path.exists(ssd_proto) and os.path.exists(ssd_model):
            self._net  = cv2.dnn.readNetFromCaffe(ssd_proto, ssd_model)
            self._mode = "ssd"
            print("[Vision] MobileNetSSD loaded (legacy)")
            return
        if os.path.exists(face_proto) and os.path.exists(face_model):
            self._net  = cv2.dnn.readNetFromCaffe(face_proto, face_model)
            self._mode = "face"
            print("[Vision] res10 face detector loaded (legacy)")
            return
        if _MP_OK:
            self._mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5)
            self._mode = "mediapipe"
            print("[Vision] MediaPipe face detection (legacy)")
            return
        self._mode = "none"
        print("[Vision] WARNING: no detection model available")

    # ── Public API ────────────────────────────────────────────────── #

    def process_frame(self, frame) -> tuple:
        """
        Returns (annotated_frame, list_of_label_strings).
        Labels follow the format:  "Red Sedan  92%"
        """
        if self._mode == "yolo":
            return self._run_yolo(frame)
        if self._mode in ("ssd", "face"):
            return self._run_ssd(frame, face_only=(self._mode == "face"))
        if self._mode == "mediapipe":
            return self._run_mediapipe(frame)
        return frame, []

    @property
    def mode(self) -> str:
        return self._mode

    def shutdown(self):
        if self._mp_face is not None:
            self._mp_face.close()

    # ── YOLOv8 path ───────────────────────────────────────────────── #

    def _run_yolo(self, frame) -> tuple:
        results = self._yolo(frame, verbose=False, conf=0.40)[0]
        labels  = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame.shape[1]-1, x2)
            y2 = min(frame.shape[0]-1, y2)

            cls_name = results.names[cls_id]
            is_vehicle = cls_id in _VEHICLES

            if is_vehicle:
                v_type = self._vehicle_type(cls_name, x1, y1, x2, y2)
                color  = self._detect_color(frame, x1, y1, x2, y2)
                label  = f"{color} {v_type}"
                box_bgr = _BOX_COLORS.get(color, _DEFAULT_BOX)
            else:
                label  = cls_name.capitalize()
                box_bgr = _DEFAULT_BOX

            conf_str  = f"{conf*100:.0f}%"
            full_label = f"{label}  {conf_str}"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_bgr, 2)

            # Label with filled background for readability
            (tw, th), _ = cv2.getTextSize(
                full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ly = max(y1 - 6, th + 6)
            cv2.rectangle(frame,
                          (x1,     ly - th - 6),
                          (x1 + tw + 8, ly + 2),
                          box_bgr, -1)
            cv2.putText(frame, full_label, (x1 + 4, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)

            labels.append(label)

        return frame, labels

    # ── Vehicle type from bounding-box shape ──────────────────────── #

    def _vehicle_type(self, cls_name: str, x1, y1, x2, y2) -> str:
        """
        Infer vehicle sub-type from YOLO class + bounding-box aspect ratio.

        Aspect ratio  =  width / height
        Sports cars: very wide, low          → asp > 2.1
        Sedans:      moderately wide          → 1.75 – 2.1
        SUVs/Crossovers: squarish – moderate → 1.3 – 1.75
        Minivans / large SUVs: taller        → < 1.3
        Trucks:      YOLO 'truck' class       → further split by asp
        Buses:       YOLO 'bus' class
        """
        asp = (x2 - x1) / max(y2 - y1, 1)

        if cls_name == "truck":
            if asp < 1.6:   return "Pickup Truck"
            elif asp < 2.8: return "Truck"
            else:           return "Semi-Truck"

        elif cls_name == "bus":
            if asp > 2.2:   return "Bus"
            else:           return "Van"

        elif cls_name == "motorcycle":
            return "Motorcycle"

        elif cls_name == "bicycle":
            return "Bicycle"

        elif cls_name == "car":
            if asp > 2.1:    return "Sports Car"
            elif asp > 1.75: return "Sedan"
            elif asp > 1.3:  return "SUV"
            else:            return "Minivan"

        return cls_name.title()

    # ── Color detection ───────────────────────────────────────────── #

    def _detect_color(self, frame, x1, y1, x2, y2) -> str:
        """
        Identify the dominant color of a vehicle by sampling the
        central body region of its bounding box (avoids road, wheels, sky).
        """
        bh = y2 - y1
        bw = x2 - x1

        # Sample the middle horizontal band of the bounding box
        ys = y1 + bh // 5
        ye = y1 + bh * 3 // 5
        xs = x1 + bw // 8
        xe = x2 - bw // 8

        crop = frame[ys:ye, xs:xe]
        if crop.size == 0:
            return "Unknown"

        # Downsample for speed
        small = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
        hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3).astype(int)

        # Vote across all pixels
        votes: dict[str, int] = {}
        for h, s, v in pixels:
            name = self._px_color(h, s, v)
            votes[name] = votes.get(name, 0) + 1

        return max(votes, key=votes.get) if votes else "Unknown"

    def _px_color(self, h: int, s: int, v: int) -> str:
        """
        Map an HSV pixel (OpenCV scale: H 0-179, S/V 0-255) to a color name.
        """
        # Achromatic first (saturation low)
        if v < 45:
            return "Black"
        if s < 35:
            if v > 210: return "White"
            if v > 130: return "Silver"
            return "Gray"

        # Chromatic by hue
        if   h <  10 or h > 165: return "Red"
        elif h <  20:             return "Orange"
        elif h <  35:             return "Yellow"
        elif h <  80:             return "Green"
        elif h < 100:             return "Cyan"
        elif h < 130:             return "Blue"
        elif h < 150:             return "Purple"
        else:                     return "Pink"

    # ── Legacy Caffe SSD ──────────────────────────────────────────── #

    def _run_ssd(self, frame, face_only: bool) -> tuple:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self._net.setInput(blob)
        detections = self._net.forward()
        labels = []

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < 0.5:
                continue
            idx = int(detections[0, 0, i, 1])
            if face_only:
                label_text = f"Face: {confidence*100:.1f}%"
            else:
                if idx < 0 or idx >= len(self._SSD_CLASSES):
                    continue
                label_text = f"{self._SSD_CLASSES[idx]}: {confidence*100:.1f}%"

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), _DEFAULT_BOX, 2)
            label_y = max(y1 - 10, 15)
            cv2.putText(frame, label_text, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            labels.append(label_text)

        return frame, labels

    # ── Legacy MediaPipe ──────────────────────────────────────────── #

    def _run_mediapipe(self, frame) -> tuple:
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mp_face.process(rgb)
        h, w    = frame.shape[:2]
        labels  = []

        if results.detections:
            for det in results.detections:
                score = det.score[0]
                bb    = det.location_data.relative_bounding_box
                x1 = max(0, int(bb.xmin * w))
                y1 = max(0, int(bb.ymin * h))
                x2 = min(w-1, int((bb.xmin + bb.width)  * w))
                y2 = min(h-1, int((bb.ymin + bb.height) * h))
                label_text = f"Face: {score*100:.1f}%"
                cv2.rectangle(frame, (x1, y1), (x2, y2), _DEFAULT_BOX, 2)
                label_y = max(y1 - 10, 15)
                cv2.putText(frame, label_text, (x1, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                labels.append(label_text)

        return frame, labels
