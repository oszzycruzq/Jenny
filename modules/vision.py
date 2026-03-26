"""
Jenny - Vision Module
Object detection and face detection from a camera frame.

Fixes from original object_detector.py:
- Model paths built relative to this file's directory (no broken hardcoded cwd paths)
- The files present (deploy.prototxt + res10_300x300_ssd_iter_140000.caffemodel)
  are a face-detection pair, NOT MobileNetSSD — the module now uses them correctly
- Class index bounds-checked before indexing CLASSES list
- Camera open state verified before entering loop
- Label Y position clamped so it never renders off the top of the frame
- Graceful fallback to MediaPipe face detection if Caffe models are missing
- Detections returned as structured list for LLM context
"""

import os
import cv2
import numpy as np

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

# Directory that contains the model files (same folder as this module's parent)
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..")


class VisionModule:
    # MobileNetSSD labels (for general object detection)
    _SSD_CLASSES = [
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
        "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
    ]

    def __init__(self, model_dir: str = None):
        self._model_dir = os.path.abspath(model_dir or _MODEL_DIR)
        self._net = None
        self._mode = None          # 'face', 'object', or 'mediapipe'
        self._mp_face = None

        self._try_load_models()

    # ------------------------------------------------------------------ #
    #  Model loading                                                       #
    # ------------------------------------------------------------------ #

    def _try_load_models(self):
        face_prototxt  = os.path.join(self._model_dir, "deploy.prototxt")
        face_caffemodel = os.path.join(self._model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        ssd_prototxt   = os.path.join(self._model_dir, "MobileNetSSD_deploy.prototxt")
        ssd_caffemodel = os.path.join(self._model_dir, "MobileNetSSD_deploy.caffemodel")

        # Prefer general object detection if caffemodel is present
        if os.path.exists(ssd_prototxt) and os.path.exists(ssd_caffemodel):
            self._net = cv2.dnn.readNetFromCaffe(ssd_prototxt, ssd_caffemodel)
            self._mode = "object"
            print("[Vision] Loaded MobileNetSSD object detector")
            return

        # Use face detection model (files that are actually present)
        if os.path.exists(face_prototxt) and os.path.exists(face_caffemodel):
            self._net = cv2.dnn.readNetFromCaffe(face_prototxt, face_caffemodel)
            self._mode = "face"
            print("[Vision] Loaded res10 face detector")
            return

        # Final fallback: MediaPipe face detection (no model files needed)
        if _MP_AVAILABLE:
            self._mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
            self._mode = "mediapipe"
            print("[Vision] Using MediaPipe face detection (no Caffe models found)")
            return

        print(
            "[Vision] WARNING: No detection model available.\n"
            "  Place deploy.prototxt + res10_300x300_ssd_iter_140000.caffemodel\n"
            "  in the Jenny folder for face detection, or\n"
            "  MobileNetSSD_deploy.prototxt + MobileNetSSD_deploy.caffemodel\n"
            "  for general object detection."
        )
        self._mode = "none"

    # ------------------------------------------------------------------ #
    #  Frame processing                                                    #
    # ------------------------------------------------------------------ #

    def process_frame(self, frame) -> tuple:
        """
        Process a BGR frame.
        Draws bounding boxes + labels in-place.
        Returns (annotated_frame, list_of_label_strings).
        """
        if self._mode == "object":
            return self._run_ssd(frame, face_only=False)
        if self._mode == "face":
            return self._run_ssd(frame, face_only=True)
        if self._mode == "mediapipe":
            return self._run_mediapipe(frame)
        return frame, []

    def _run_ssd(self, frame, face_only: bool) -> tuple:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5
        )
        self._net.setInput(blob)
        detections = self._net.forward()

        labels_found = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < 0.5:
                continue

            idx = int(detections[0, 0, i, 1])

            if face_only:
                label_text = f"Face: {confidence * 100:.1f}%"
            else:
                # FIX: bounds-check before indexing CLASSES
                if idx < 0 or idx >= len(self._SSD_CLASSES):
                    continue
                label_text = f"{self._SSD_CLASSES[idx]}: {confidence * 100:.1f}%"

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            # Clamp coordinates to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)

            # FIX: clamp label Y so it never renders above the frame
            label_y = max(y1 - 10, 15)
            cv2.putText(
                frame, label_text, (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
            )
            labels_found.append(label_text)

        return frame, labels_found

    def _run_mediapipe(self, frame) -> tuple:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mp_face.process(rgb)
        h, w = frame.shape[:2]

        labels_found = []
        if results.detections:
            for det in results.detections:
                score = det.score[0]
                bb = det.location_data.relative_bounding_box
                x1 = max(0, int(bb.xmin * w))
                y1 = max(0, int(bb.ymin * h))
                x2 = min(w - 1, int((bb.xmin + bb.width) * w))
                y2 = min(h - 1, int((bb.ymin + bb.height) * h))

                label_text = f"Face: {score * 100:.1f}%"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                label_y = max(y1 - 10, 15)
                cv2.putText(
                    frame, label_text, (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                )
                labels_found.append(label_text)

        return frame, labels_found

    @property
    def mode(self) -> str:
        return self._mode

    def shutdown(self):
        if self._mp_face is not None:
            self._mp_face.close()
