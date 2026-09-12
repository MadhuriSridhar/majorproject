"""
Unit tests for the audio priority / interruption / volume pass.

Covers:
- Higher-priority alerts replace lower ones (WARNING replaces LOW,
  DROWSY replaces WARNING).
- Lower-priority alerts never replace higher ones.
- Interrupted alarms do not play their old voice message.
- Audio cancellation works.
- Volume settings are applied (alert/alarm/voice).
- Audio failure does not crash the app.
- Natural completion releases the priority slot so re-alerts can play.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_engine import (  # noqa: E402
    AudioBackend,
    AudioEngine,
    AudioSettings,
    _scale_wav,
    _wav_duration,
)


def _make_wav(duration_s: float = 1.0, peak: int = 10000, rate: int = 8000) -> bytes:
    """Build a minimal 16-bit mono PCM WAV for testing."""
    import math
    import struct

    num_samples = int(rate * duration_s)
    pcm = b"".join(
        struct.pack("<h", int(peak * math.sin(2 * math.pi * 440 * i / rate)))
        for i in range(num_samples)
    )
    data_size = len(pcm)
    byte_rate = rate * 2  # mono, 16-bit
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,          # PCM
        1,          # mono
        rate,
        byte_rate,
        2,          # block align
        16,         # bits per sample
        b"data",
        data_size,
    )
    return header + pcm


class WavHelperTests(unittest.TestCase):
    def test_wav_duration(self) -> None:
        wav = _make_wav(duration_s=1.0, rate=8000)
        self.assertAlmostEqual(_wav_duration(wav), 1.0, places=2)

    def test_scale_wav_half_volume(self) -> None:
        wav = _make_wav(duration_s=0.01, peak=10000, rate=8000)
        # Read original PCM samples.
        import array
        import struct

        data_size = struct.unpack_from("<I", wav, 40)[0]
        original = array.array("h")
        original.frombytes(wav[44 : 44 + data_size])

        scaled = _scale_wav(wav, 0.5)
        scaled_samples = array.array("h")
        scaled_samples.frombytes(scaled[44 : 44 + data_size])

        self.assertEqual(len(original), len(scaled_samples))
        for orig, sc in zip(original, scaled_samples):
            # 50% amplitude (allow rounding).
            self.assertLessEqual(abs(sc), abs(orig))
            if abs(orig) >= 100:
                self.assertAlmostEqual(sc / orig, 0.5, delta=0.06)

    def test_scale_wav_full_volume_unchanged(self) -> None:
        wav = _make_wav(duration_s=0.01, peak=10000, rate=8000)
        self.assertEqual(_scale_wav(wav, 1.0), wav)
        self.assertEqual(_scale_wav(wav, 0.999), wav)

    def test_scale_wav_zero_volume_mutes(self) -> None:
        wav = _make_wav(duration_s=0.01, peak=10000, rate=8000)
        import array
        import struct

        data_size = struct.unpack_from("<I", wav, 40)[0]
        scaled = _scale_wav(wav, 0.0)
        samples = array.array("h")
        samples.frombytes(scaled[44 : 44 + data_size])
        self.assertTrue(all(s == 0 for s in samples))


class VolumeSelectionTests(unittest.TestCase):
    def test_alert_volume_used_for_low_fatigue(self) -> None:
        backend = AudioBackend(
            severity="LOW_FATIGUE",
            wav_path=None,
            settings=AudioSettings(alert_volume=0.30, alarm_volume=0.80),
        )
        self.assertAlmostEqual(backend._volume_for("LOW_FATIGUE"), 0.30)

    def test_alert_volume_used_for_alert(self) -> None:
        backend = AudioBackend(
            severity="ALERT",
            wav_path=None,
            settings=AudioSettings(alert_volume=0.30, alarm_volume=0.80),
        )
        self.assertAlmostEqual(backend._volume_for("ALERT"), 0.30)

    def test_alarm_volume_used_for_warning(self) -> None:
        backend = AudioBackend(
            severity="WARNING",
            wav_path=None,
            settings=AudioSettings(alert_volume=0.30, alarm_volume=0.80),
        )
        self.assertAlmostEqual(backend._volume_for("WARNING"), 0.80)

    def test_alarm_volume_used_for_drowsy(self) -> None:
        backend = AudioBackend(
            severity="DROWSY",
            wav_path=None,
            settings=AudioSettings(alert_volume=0.30, alarm_volume=0.80),
        )
        self.assertAlmostEqual(backend._volume_for("DROWSY"), 0.80)

    def test_alert_and_alarm_volumes_are_independent(self) -> None:
        settings = AudioSettings(alert_volume=0.20, alarm_volume=0.90)
        self.assertAlmostEqual(
            AudioBackend("LOW_FATIGUE", None, settings)._volume_for("LOW_FATIGUE"), 0.20
        )
        self.assertAlmostEqual(
            AudioBackend("WARNING", None, settings)._volume_for("WARNING"), 0.90
        )

    def test_volume_clamped_to_range(self) -> None:
        backend = AudioBackend(
            severity="DROWSY",
            wav_path=None,
            settings=AudioSettings(alert_volume=-1.0, alarm_volume=5.0),
        )
        self.assertEqual(backend._volume_for("DROWSY"), 1.0)  # 5 clamped to 1
        backend2 = AudioBackend(
            severity="LOW_FATIGUE",
            wav_path=None,
            settings=AudioSettings(alert_volume=-1.0, alarm_volume=5.0),
        )
        self.assertEqual(backend2._volume_for("LOW_FATIGUE"), 0.0)  # -1 clamped to 0

    def test_voice_volume_respects_alert_volume(self) -> None:
        # LOW_FATIGUE voice uses alert_volume (not alarm_volume).
        backend = AudioBackend(
            severity="LOW_FATIGUE",
            wav_path=None,
            settings=AudioSettings(alert_volume=0.45, alarm_volume=0.90, voice_enabled=True),
        )
        applied: list[float] = []

        class FakeEngine:
            def setProperty(self, name, value):  # noqa: N802
                if name == "volume":
                    applied.append(float(value))

            def say(self, message):  # noqa: A002
                pass

            def runAndWait(self):  # noqa: N802
                pass

            def stop(self):
                pass

        class FakePyttsx3:
            @staticmethod
            def init():
                return FakeEngine()

        import sys as _sys

        real_pyttsx3 = _sys.modules.get("pyttsx3")
        _sys.modules["pyttsx3"] = FakePyttsx3()  # type: ignore[assignment]
        try:
            backend._speak("Please stay alert.")
        finally:
            if real_pyttsx3 is not None:
                _sys.modules["pyttsx3"] = real_pyttsx3
            else:
                _sys.modules.pop("pyttsx3", None)

        self.assertTrue(applied, "volume property should have been set")
        self.assertAlmostEqual(applied[0], 0.45)

    def test_subsequent_alert_uses_updated_volume(self) -> None:
        engine = AudioEngine(
            AudioSettings(alert_volume=0.2, alarm_volume=0.4, voice_enabled=False, sound_enabled=False)
        )
        engine.play_alert("WARNING")
        first = engine._current
        assert first is not None
        self.assertAlmostEqual(first._volume_for("WARNING"), 0.4)
        engine._on_job_done("WARNING")
        engine.settings = AudioSettings(
            alert_volume=0.9, alarm_volume=0.8, voice_enabled=False, sound_enabled=False
        )
        engine.play_alert("WARNING")
        second = engine._current
        assert second is not None
        self.assertAlmostEqual(second._volume_for("WARNING"), 0.8)
        engine.shutdown()


class InterruptionTests(unittest.TestCase):
    def test_interrupted_alarm_does_not_play_old_voice(self) -> None:
        backend = AudioBackend(
            severity="WARNING",
            wav_path=Path("dummy.wav"),
            settings=AudioSettings(voice_enabled=True, sound_enabled=True),
        )
        order: list[str] = []
        backend._play_wav = lambda path: order.append("sound") or None  # type: ignore[method-assign]
        backend._speak = lambda msg: order.append("voice")  # type: ignore[method-assign]
        # Simulate interruption during the alarm.
        def play_then_interrupt(path):
            order.append("sound")
            backend._sound_interrupted = True

        backend._play_wav = play_then_interrupt  # type: ignore[method-assign]
        backend.run()
        self.assertEqual(order, ["sound"])  # voice skipped

    def test_cancellation_before_voice_skips_voice(self) -> None:
        backend = AudioBackend(
            severity="WARNING",
            wav_path=Path("dummy.wav"),
            settings=AudioSettings(voice_enabled=True, sound_enabled=True),
        )
        order: list[str] = []
        backend._play_wav = lambda path: order.append("sound") or None  # type: ignore[method-assign]
        backend._speak = lambda msg: order.append("voice")  # type: ignore[method-assign]
        # Cancel the job before it reaches the voice stage.
        backend.interrupted.set()
        backend.run()
        self.assertEqual(order, ["sound"])  # voice skipped

    def test_interrupted_voice_never_starts(self) -> None:
        backend = AudioBackend(
            severity="LOW_FATIGUE",
            wav_path=None,
            settings=AudioSettings(voice_enabled=True, sound_enabled=True),
        )
        backend.interrupted.set()
        spoken: list[str] = []

        class FakeEngine:
            def setProperty(self, name, value):  # noqa: N802
                pass

            def say(self, message):  # noqa: A002
                spoken.append(message)

            def stop(self):
                pass

            def iterate(self):
                pass

        class FakePyttsx3:
            @staticmethod
            def init():
                return FakeEngine()

        real_pyttsx3 = sys.modules.get("pyttsx3")
        sys.modules["pyttsx3"] = FakePyttsx3()  # type: ignore[assignment]
        try:
            backend._speak("Please stay alert.")
        finally:
            if real_pyttsx3 is not None:
                sys.modules["pyttsx3"] = real_pyttsx3
            else:
                sys.modules.pop("pyttsx3", None)

        self.assertEqual(spoken, [])

    def test_audio_failure_does_not_crash(self) -> None:
        errors: list[str] = []
        backend = AudioBackend(
            severity="WARNING",
            wav_path=Path("missing.wav"),
            settings=AudioSettings(voice_enabled=True, sound_enabled=True),
            on_error=errors.append,
        )

        def boom(path):  # noqa: A002
            raise RuntimeError("playback failed")

        backend._play_wav = boom  # type: ignore[method-assign]
        # Should not raise.
        backend.run()
        self.assertTrue(errors, "on_error should have been called")


class AudioEnginePriorityTests(unittest.TestCase):
    def _engine(self) -> AudioEngine:
        # No real audio in tests.
        return AudioEngine(
            AudioSettings(
                alert_volume=0.5,
                alarm_volume=0.5,
                voice_enabled=False,
                sound_enabled=False,
            )
        )

    def test_warning_replaces_low_fatigue(self) -> None:
        engine = self._engine()
        engine.play_alert("LOW_FATIGUE")
        self.assertEqual(engine.active_severity, "LOW_FATIGUE")
        engine.play_alert("WARNING")
        self.assertEqual(engine.active_severity, "WARNING")
        engine.shutdown()

    def test_drowsy_replaces_warning(self) -> None:
        engine = self._engine()
        engine.play_alert("WARNING")
        self.assertEqual(engine.active_severity, "WARNING")
        engine.play_alert("DROWSY")
        self.assertEqual(engine.active_severity, "DROWSY")
        engine.shutdown()

    def test_drowsy_cannot_be_replaced_by_warning(self) -> None:
        engine = self._engine()
        engine.play_alert("DROWSY")
        engine.play_alert("WARNING")
        self.assertEqual(engine.active_severity, "DROWSY")
        engine.shutdown()

    def test_drowsy_cannot_be_replaced_by_low_fatigue(self) -> None:
        engine = self._engine()
        engine.play_alert("DROWSY")
        engine.play_alert("LOW_FATIGUE")
        self.assertEqual(engine.active_severity, "DROWSY")
        engine.shutdown()

    def test_natural_completion_releases_priority_slot(self) -> None:
        engine = self._engine()
        engine.play_alert("DROWSY")
        self.assertEqual(engine.active_severity, "DROWSY")
        # Simulate the DROWSY job finishing naturally.
        engine._on_job_done("DROWSY")
        self.assertIsNone(engine.active_severity)
        # A lower-priority alert may now play.
        engine.play_alert("LOW_FATIGUE")
        self.assertEqual(engine.active_severity, "LOW_FATIGUE")
        engine.shutdown()

    def test_drowsy_can_trigger_again_after_completion(self) -> None:
        engine = self._engine()
        engine.play_alert("DROWSY")
        engine._on_job_done("DROWSY")
        self.assertIsNone(engine.active_severity)
        engine.play_alert("DROWSY")
        self.assertEqual(engine.active_severity, "DROWSY")
        engine.shutdown()

    def test_recovery_then_reentry_allows_audio(self) -> None:
        # Full chain: DROWSY -> recovers (job done) -> DROWSY again.
        engine = self._engine()
        engine.play_alert("DROWSY")
        engine._on_job_done("DROWSY")
        engine.play_alert("DROWSY")
        self.assertEqual(engine.active_severity, "DROWSY")
        engine.shutdown()


if __name__ == "__main__":
    unittest.main()