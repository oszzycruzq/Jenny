"""
Jenny - Gesture Module
Real-time hand gesture recognition via MediaPipe.

Fixes from original "Real-Time Hand Gesture.py":
- np.arccos clipped to [-1, 1] to prevent domain errors on float edge cases
- Smoothing rewritten: used a candidate/counter approach so last_gesture
  actually transitions (original counter reset to 0 on every change so it
  never reached the SMOOTHING threshold)
- Live Long (Vulcan salute) checked BEFORE Four — previously unreachable
  because Four's broader condition always matched first
- Unused `pip` parameter removed from is_finger_extended
- hands.close() called on shutdown
- All logic encapsulated in a class for integration
"""

import math
import cv2
import mediapipe as mp
import numpy as np


class GestureModule:
    SMOOTHING = 3  # consecutive frames required to confirm a gesture change

    def __init__(self):
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self._drawing = mp.solutions.drawing_utils

        # Smoothing state
        self._confirmed_gesture = "No Hand"
        self._candidate_gesture = "No Hand"
        self._candidate_count = 0

    # ------------------------------------------------------------------ #
    #  Geometry helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _dist(p1, p2) -> float:
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    @staticmethod
    def _angle(p1, p2, p3) -> float:
        """Angle at p2 formed by p1-p2-p3. Safe against float domain errors."""
        a, b, c = np.array(p1), np.array(p2), np.array(p3)
        ba, bc = a - b, c - b
        denom = np.linalg.norm(ba) * np.linalg.norm(bc)
        if denom == 0:
            return 0.0
        # FIX: clip cosine to [-1, 1] before arccos to avoid NaN
        cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))

    # ------------------------------------------------------------------ #
    #  Gesture detection                                                   #
    # ------------------------------------------------------------------ #

    def _detect(self, landmarks, shape) -> str:
        h, w, _ = shape
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark]

        wrist       = pts[0]
        thumb_tip   = pts[4];  thumb_mcp   = pts[2]
        index_tip   = pts[8];  index_mcp   = pts[5]
        middle_tip  = pts[12]; middle_mcp  = pts[9]
        ring_tip    = pts[16]; ring_mcp    = pts[13]
        pinky_tip   = pts[20]; pinky_mcp   = pts[17]

        def extended(tip, mcp, thresh=50):
            return self._dist(tip, mcp) > thresh

        thumb_ext  = self._angle(thumb_tip, thumb_mcp, wrist) > 30
        index_ext  = extended(index_tip,  index_mcp)
        middle_ext = extended(middle_tip, middle_mcp)
        ring_ext   = extended(ring_tip,   ring_mcp)
        pinky_ext  = extended(pinky_tip,  pinky_mcp)

        ti_dist = self._dist(thumb_tip, index_tip)
        im_dist = self._dist(index_tip, middle_tip)

        fingers = [thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext]

        # --- ordered from most-specific to least-specific ---
        if not any(fingers):
            return "Fist"
        if all(fingers):
            return "Open Palm"

        # Thumb-only gestures
        if thumb_ext and not any(fingers[1:]):
            return "Thumbs Up" if thumb_tip[1] < wrist[1] else "Thumbs Down"

        # Pinch / OK (thumb + index close together)
        if ti_dist < 30:
            if not middle_ext and not ring_ext and not pinky_ext:
                return "Pinch"
            if middle_ext and ring_ext and pinky_ext:
                return "OK"

        # Phone Call (thumb + pinky)
        if thumb_ext and pinky_ext and not index_ext and not middle_ext and not ring_ext:
            return "Phone Call"

        # Gun Shape (thumb + index)
        if thumb_ext and index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return "Gun Shape"

        # Pointing (index only)
        if index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return "Pointing"

        # Peace (index + middle, fingers apart)
        if index_ext and middle_ext and not ring_ext and not pinky_ext:
            return "Peace"

        # Rock On (index + pinky)
        if index_ext and pinky_ext and not middle_ext and not ring_ext:
            return "Rock On"

        # Spider-Man (index + middle + pinky)
        if index_ext and middle_ext and pinky_ext and not ring_ext:
            return "Spider-Man"

        # Three (index + middle + ring)
        if index_ext and middle_ext and ring_ext and not pinky_ext:
            return "Three"

        # Four fingers (no thumb)
        if index_ext and middle_ext and ring_ext and pinky_ext and not thumb_ext:
            # FIX: check Live Long BEFORE Four — Vulcan salute has a wide
            # spread between middle and ring fingers
            m_r_dist = self._dist(middle_tip, ring_tip)
            if m_r_dist > 40:
                return "Live Long"
            return "Four"

        return "Unknown"

    # ------------------------------------------------------------------ #
    #  Smoothing                                                           #
    # ------------------------------------------------------------------ #

    def _smooth(self, current: str) -> str:
        """
        FIX: original code reset counter to 0 on every change so last_gesture
        never transitioned. New approach: track a candidate and its consecutive
        count independently of confirmed_gesture.
        """
        if current == self._candidate_gesture:
            self._candidate_count += 1
        else:
            self._candidate_gesture = current
            self._candidate_count = 1

        if self._candidate_count >= self.SMOOTHING:
            self._confirmed_gesture = self._candidate_gesture

        return self._confirmed_gesture

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def process_frame(self, frame) -> tuple:
        """
        Process a BGR camera frame.
        Draws hand landmarks in-place.
        Returns (annotated_frame, gesture_string).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        current = "No Hand"
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                self._drawing.draw_landmarks(
                    frame, hand_lm, self._mp_hands.HAND_CONNECTIONS,
                    self._drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self._drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
                )
                current = self._detect(hand_lm, frame.shape)

        gesture = self._smooth(current)
        return frame, gesture

    @property
    def confirmed_gesture(self) -> str:
        return self._confirmed_gesture

    def shutdown(self):
        self._hands.close()
