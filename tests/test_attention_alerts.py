"""
Focused unit tests for attention-layer alert behavior (no overlapping audio).
"""

from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alert_escalation import AlertEscalationController  # noqa: E402
from audio_engine import AudioBackend, AudioEngine, AudioSettings  # noqa: E402


def _make_wav(duration_s: float = 0.05, peak: int = 1000, rate: int = 8000) -> bytes:
    import math
    import struct

    num_samples = max(1, int(rate * duration_s))
    pcm = b"".join(
        struct.pack("<h", int(peak * math.sin(2 * math.pi * 440 * i / rate)))
        for i in range(num_samples)
    )
    data_size = len(pcm)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        rate,
        rate * 2,
        2,
        16,
        b"data",
        data_size,
    )
    return header + pcm


class AttentionAlertBehaviorTests(unittest.TestCase):
    def test_same_alert_does_not_repeat_every_frame(self) -> None:
        ctl = AlertEscalationController()
        first = ctl.update("WARNING", 0.0)
        self.assertTrue(first.should_play_audio)
        for t in (0.05, 0.1, 0.2, 0.5, 1.0):
            r = ctl.update("WARNING", t)
            self.assertFalse(r.should_play_audio)

    def test_cooldown_works(self) -> None:
        ctl = AlertEscalationController(warning_cooldown_s=10.0, warning_display_s=0.5)
        ctl.update("WARNING", 0.0)
        # Recover, then re-enter before cooldown elapses.
        ctl.update("NORMAL", 1.0)
        blocked = ctl.update("WARNING", 2.0)
        self.assertFalse(blocked.should_play_audio)
        # Recover again, then re-enter after cooldown.
        ctl.update("NORMAL", 3.0)
        allowed = ctl.update("WARNING", 12.0)
        self.assertTrue(allowed.should_play_audio)

    def test_different_priority_can_escalate(self) -> None:
        ctl = AlertEscalationController()
        low = ctl.update("LOW_FATIGUE", 0.0)
        self.assertTrue(low.should_play_audio)
        high = ctl.update("WARNING", 0.5)
        self.assertTrue(high.should_play_audio)
        self.assertEqual(high.audio_severity, "WARNING")
        critical = ctl.update("DROWSY", 1.0)
        self.assertTrue(critical.should_play_audio)
        self.assertEqual(critical.audio_severity, "DROWSY")

    def test_audio_does_not_overlap_same_severity(self) -> None:
        engine = AudioEngine(AudioSettings(voice_enabled=False, sound_enabled=True, alert_volume=0.2))
        wav = _make_wav(0.2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "beep.wav"
            path.write_bytes(wav)

            def _sound_for(_severity: str) -> str:
                return "beep"

            with patch.object(engine, "_sound_for", side_effect=_sound_for), patch(
                "audio_engine.sound_path", return_value=path
            ), patch.object(AudioBackend, "_play_wav", lambda self, p: time.sleep(0.05)), patch.object(
                AudioBackend, "_speak", lambda self, msg: None
            ):
                engine.play_alert("WARNING")
                first = engine._current
                engine.play_alert("WARNING")
                # Same severity while still playing must not start a second job.
                self.assertIs(engine._current, first)
                if first is not None:
                    first.interrupted.set()
                    first.join(timeout=1.0)
        engine.shutdown()

    def test_volume_behavior_intact(self) -> None:
        settings = AudioSettings(alert_volume=0.35, alarm_volume=0.9, voice_enabled=True)
        self.assertAlmostEqual(settings.master_volume, 0.35, places=5)
        backend = AudioBackend("DROWSY", None, settings)
        self.assertAlmostEqual(backend._volume_for("DROWSY"), 0.35, places=5)
        self.assertAlmostEqual(backend._volume_for("LOW_FATIGUE"), 0.35, places=5)


if __name__ == "__main__":
    unittest.main()
