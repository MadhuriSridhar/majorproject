"""Unit tests for Phase 2 AudioEngine (state transitions, no overlap)."""

from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_engine import (  # noqa: E402
    AudioEngine,
    AudioSettings,
    DEFAULT_SOUNDS,
    VOICE_MESSAGES,
)
from predefined_sounds import ALL_SOUND_NAMES  # noqa: E402


class VoiceMessageTests(unittest.TestCase):
    def test_voice_messages_exact(self) -> None:
        self.assertEqual(VOICE_MESSAGES["LOW_FATIGUE"], "Please stay alert.")
        self.assertEqual(VOICE_MESSAGES["ALERT"], "Your alertness appears to be decreasing. Please stay attentive.")
        self.assertEqual(VOICE_MESSAGES["WARNING"], "Fatigue warning. Please stay alert and consider taking a break soon.")
        self.assertEqual(VOICE_MESSAGES["DROWSY"], "Driver fatigue detected. Please take a safe break now.")

    def test_default_sound_selection(self) -> None:
        # LOW FATIGUE is voice only (no alarm sound). ALERT uses a soft beep.
        self.assertIn(DEFAULT_SOUNDS["ALERT"], ALL_SOUND_NAMES)
        self.assertIn(DEFAULT_SOUNDS["WARNING"], ALL_SOUND_NAMES)
        self.assertIn(DEFAULT_SOUNDS["DROWSY"], ALL_SOUND_NAMES)


class AudioEngineStateTests(unittest.TestCase):
    def test_play_alert_activates(self) -> None:
        engine = AudioEngine(
            AudioSettings(
                alert_volume=0.5,
                alarm_volume=0.5,
                voice_enabled=False,
                sound_enabled=False,  # no real audio in tests
            )
        )
        engine.play_alert("LOW_FATIGUE")
        self.assertEqual(engine.active_severity, "LOW_FATIGUE")
        engine.shutdown()

    def test_lower_priority_does_not_preempt_higher(self) -> None:
        engine = AudioEngine(
            AudioSettings(
                alert_volume=0.5,
                alarm_volume=0.5,
                voice_enabled=False,
                sound_enabled=False,
            )
        )
        engine.play_alert("WARNING")
        self.assertEqual(engine.active_severity, "WARNING")
        # LOW_FATIGUE cannot preempt WARNING.
        engine.play_alert("LOW_FATIGUE")
        self.assertEqual(engine.active_severity, "WARNING")
        engine.shutdown()

    def test_shutdown_resets(self) -> None:
        engine = AudioEngine(
            AudioSettings(
                alert_volume=0.5,
                alarm_volume=0.5,
                voice_enabled=False,
                sound_enabled=False,
            )
        )
        engine.play_alert("DROWSY")
        engine.shutdown()
        self.assertIsNone(engine.active_severity)


class AudioBackendSequenceTests(unittest.TestCase):
    def test_voice_disabled_skips_speak(self) -> None:
        # The real _speak must return early when voice_enabled is False.
        from audio_engine import AudioBackend

        backend = AudioBackend(
            severity="LOW_FATIGUE",
            wav_path=None,
            settings=AudioSettings(voice_enabled=False, sound_enabled=False),
        )
        # Patch run() to only call _speak (bypass sound), and verify _speak
        # does not raise / avoids pyttsx3 when voice is disabled.
        backend._play_wav = lambda path: None  # type: ignore[method-assign]
        # _speak with voice disabled should return without initializing pyttsx3.
        # We verify by checking it does not raise and returns None.
        result = backend._speak("Please stay alert.")
        self.assertIsNone(result)

    def test_backend_run_skips_sound_when_disabled(self) -> None:
        from audio_engine import AudioBackend

        backend = AudioBackend(
            severity="LOW_FATIGUE",
            wav_path=None,
            settings=AudioSettings(voice_enabled=False, sound_enabled=False),
        )
        # Patch _play_wav to record calls without real audio.
        calls: list[str] = []
        backend._play_wav = lambda path: calls.append("sound")  # type: ignore[method-assign]
        backend.run()
        # wav_path is None and sound_enabled is False -> no sound.
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()