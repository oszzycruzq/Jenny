"""
Jenny - Speech Module
Handles speech recognition (mic -> text) and text-to-speech (text -> voice).

Fixes from original jennytalk.py:
- pyttsx3 engine initialized once per TTS thread (not on every speak() call)
- listen() has timeout so it never hangs forever
- UnknownValueError now says "Could not understand" instead of "Unknown error"
- TTS runs in a dedicated background thread so speak() is non-blocking
- Proper shutdown support
"""

import threading
import queue
import speech_recognition as sr
import pyttsx3


class SpeechModule:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self._tts_queue = queue.Queue()
        self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self._tts_thread.start()

    # ------------------------------------------------------------------ #
    #  TTS                                                                 #
    # ------------------------------------------------------------------ #

    def _tts_worker(self):
        """Runs in a dedicated thread. Initialises the engine once."""
        engine = pyttsx3.init()

        # Try to pick a female voice (Jenny should sound like Jenny)
        voices = engine.getProperty("voices")
        for voice in voices:
            name = voice.name.lower()
            if "zira" in name or "female" in name or "hazel" in name:
                engine.setProperty("voice", voice.id)
                break

        engine.setProperty("rate", 175)

        while True:
            text = self._tts_queue.get()
            if text is None:        # shutdown signal
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:
                print(f"[TTS Error] {exc}")
            finally:
                self._tts_queue.task_done()

    def speak(self, text: str):
        """Queue text to be spoken. Returns immediately (non-blocking)."""
        print(f"[Jenny] {text}")
        self._tts_queue.put(text)

    # ------------------------------------------------------------------ #
    #  Speech Recognition                                                  #
    # ------------------------------------------------------------------ #

    def listen(self, timeout: float = 5.0, phrase_limit: float = 10.0) -> str | None:
        """
        Listen for one utterance from the microphone.

        Returns the transcribed string (lowercase), or None if nothing was
        heard / understood / service was unavailable.

        timeout      – seconds to wait for speech to start before giving up
        phrase_limit – max seconds to record once speech has started
        """
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_limit,
                )
            text = self.recognizer.recognize_google(audio)
            return text.lower()

        except sr.WaitTimeoutError:
            return None  # no speech detected within timeout - normal, not an error

        except sr.UnknownValueError:
            return None  # audio heard but couldn't be understood

        except sr.RequestError as exc:
            print(f"[Speech] Recognition service unavailable: {exc}")
            return None

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def shutdown(self):
        """Stop the TTS thread cleanly."""
        self._tts_queue.put(None)
        self._tts_thread.join(timeout=3)
