"""
Jenny - Unified AI Assistant
==============================
Combines speech recognition, hand gesture detection, object/face detection,
and a Claude LLM brain into a single interactive assistant.

Controls (header buttons or keyboard when window is focused):
  G  – switch to Gesture mode
  V  – switch to Vision mode
  Q  – quit

Usage:
  1. Set your Anthropic API key:
       Windows:  set ANTHROPIC_API_KEY=sk-ant-...
       Mac/Linux: export ANTHROPIC_API_KEY=sk-ant-...
  2. Run:
       python jenny.py
"""

import os
import sys

# Must be set before any pydantic/anthropic import
os.environ.setdefault("PYDANTIC_DISABLE_PLUGINS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import threading
import queue
import time

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from modules.speech  import SpeechModule
from modules.gesture import GestureModule
from modules.vision  import VisionModule
from modules.ui      import JennyUI

try:
    import anthropic
    _ANTHROPIC_OK = True
except ImportError:
    _ANTHROPIC_OK = False
    print("[Jenny] WARNING: 'anthropic' package not found. Run: pip install anthropic")


# ══════════════════════════════════════════════════════════════════════════ #
#  Jenny                                                                     #
# ══════════════════════════════════════════════════════════════════════════ #

SYSTEM_PROMPT = """You are Jenny, a friendly and helpful AI assistant.
You can see through a camera — you may be shown what gestures the user is making
or what objects are visible. Keep your responses concise (1-3 sentences) and
conversational, since they will be read aloud. Do not use markdown or bullet points."""


class Jenny:
    def __init__(self):
        print("[Jenny] Initialising modules...")
        self.speech  = SpeechModule()
        self.gesture = GestureModule()
        self.vision  = VisionModule()

        # LLM client (optional)
        self._llm = None
        if _ANTHROPIC_OK:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if api_key:
                self._llm = anthropic.Anthropic(api_key=api_key)
                print("[Jenny] Claude API connected")
            else:
                print("[Jenny] WARNING: ANTHROPIC_API_KEY not set — LLM disabled")
        self._history = []

        # Shared state (read by camera thread, written by various threads)
        self._mode            = "gesture"
        self._current_gesture = "No Hand"
        self._current_dets    = []
        self._running         = True

        # Queues for thread communication
        self._speech_q   = queue.Queue()   # transcribed text → LLM worker
        self._response_q = queue.Queue()   # (user_text, jenny_text) → UI

        # Build UI (must run mainloop on main thread later)
        self._ui = JennyUI(
            on_mode_change=self._on_mode_change,
            on_quit=self._on_quit,
        )

    # ------------------------------------------------------------------ #
    #  Callbacks from UI                                                   #
    # ------------------------------------------------------------------ #

    def _on_mode_change(self, mode: str):
        self._mode = mode
        self._ui.set_mode(mode)
        print(f"[Jenny] Switched to {mode.upper()} mode")

    def _on_quit(self):
        self._running = False

    # ------------------------------------------------------------------ #
    #  Background threads                                                  #
    # ------------------------------------------------------------------ #

    def _camera_worker(self):
        """Capture frames, run active vision module, push to UI."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[Jenny] No camera found — camera feed unavailable.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)   # mirror for natural selfie view

            if self._mode == "gesture":
                frame, gesture = self.gesture.process_frame(frame)
                self._current_gesture = gesture
                self._ui.set_gesture(gesture)
            else:
                frame, dets = self.vision.process_frame(frame)
                self._current_dets = dets
                self._ui.set_detections(dets)

            self._ui.push_frame(frame)

            # Pull conversation updates and post to UI
            try:
                user_t, jenny_t = self._response_q.get_nowait()
                self._ui.add_turn("you",   user_t)
                self._ui.add_turn("jenny", jenny_t)
            except queue.Empty:
                pass

        cap.release()

    def _speech_worker(self):
        """Continuously listen for speech and queue transcriptions."""
        while self._running:
            self._ui.set_listening(True)
            text = self.speech.listen(timeout=4.0, phrase_limit=12.0)
            self._ui.set_listening(False)
            if text:
                print(f"[You] {text}")
                self._speech_q.put(text)

    def _llm_worker(self):
        """Process speech queue through Claude and speak responses."""
        while self._running:
            try:
                user_text = self._speech_q.get(timeout=1.0)
            except queue.Empty:
                continue

            response = self._get_response(user_text)
            self.speech.speak(response)
            self._response_q.put((user_text, response))

    def _get_response(self, user_text: str) -> str:
        """Call Claude with conversation history. Falls back to echo if no API."""
        if self._llm is None:
            return f"You said: {user_text}. (Connect an API key for real responses.)"

        context_parts = []
        if self._mode == "gesture" and self._current_gesture not in ("No Hand", "Unknown"):
            context_parts.append(f"[Camera: gesture detected — {self._current_gesture}]")
        elif self._mode == "vision" and self._current_dets:
            context_parts.append(f"[Camera: {', '.join(self._current_dets[:3])}]")

        full_user = " ".join(context_parts + [user_text])
        self._history.append({"role": "user", "content": full_user})

        if len(self._history) > 20:
            self._history = self._history[-20:]

        try:
            result = self._llm.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=256,
                system=SYSTEM_PROMPT,
                messages=self._history,
            )
            reply = result.content[0].text.strip()
        except Exception as exc:
            reply = f"Sorry, I had trouble thinking there. ({exc})"

        self._history.append({"role": "assistant", "content": reply})
        return reply

    # ------------------------------------------------------------------ #
    #  Entry point                                                         #
    # ------------------------------------------------------------------ #

    def run(self):
        # Start background threads
        threads = [
            threading.Thread(target=self._camera_worker, daemon=True),
            threading.Thread(target=self._speech_worker, daemon=True),
            threading.Thread(target=self._llm_worker,    daemon=True),
        ]
        for t in threads:
            t.start()

        self.speech.speak(
            "Hi! I'm Jenny, your AI assistant. "
            "Press G for gesture mode or V for vision mode. Just talk to me!"
        )

        # Tkinter mainloop runs on the main thread (required by Tkinter)
        self._ui.run()

        # Cleanup after window closes
        self._running = False
        self.speech.speak("Goodbye!")
        time.sleep(1.5)
        self.speech.shutdown()
        self.gesture.shutdown()
        self.vision.shutdown()


# ══════════════════════════════════════════════════════════════════════════ #
#  Entry point                                                               #
# ══════════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    jenny = Jenny()
    try:
        jenny.run()
    except KeyboardInterrupt:
        print("\n[Jenny] Interrupted by user.")
        jenny._running = False
