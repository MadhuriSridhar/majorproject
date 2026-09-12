"""
Focused tests for the alertness stabilization pass (voice volume, multi-series
graph data, alert-decision logging, alertness semantics).

No webcam / MediaPipe / network / real audio required.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alert_logging import format_alert_decision  # noqa: E402
from attention_engine import AttentionEngine  # noqa: E402
from audio_engine import (  # noqa: E402
    VOICE_VOLUME,
    AudioBackend,
    AudioSettings,
)
from session_manager import AlertnessSample, trend_series  # noqa: E402


class VoiceVolumeTests(unittest.TestCase):
    def test_voice_volume_is_central_and_separate(self) -> None:
        settings = AudioSettings()
        self.assertEqual(settings.voice_volume, VOICE_VOLUME)
        self.assertNotEqual(settings.alert_volume, settings.alarm_volume)
        custom = AudioSettings(alert_volume=0.2, alarm_volume=0.9, voice_volume=0.55)
        self.assertEqual(custom.voice_volume, 0.55)
        self.assertEqual(custom.alert_volume, 0.2)
        self.assertEqual(custom.alarm_volume, 0.9)

    def test_speak_applies_single_voice_volume(self) -> None:
        captured: dict[str, object] = {}

        class _FakeEngine:
            def setProperty(self, name, value):  # noqa: N802
                captured[name] = value

            def say(self, _msg):  # noqa: N802
                return None

            def runAndWait(self):  # noqa: N802
                return None

            def stop(self):  # noqa: N802
                return None

        class _FakeTTS:
            def init(self):
                return _FakeEngine()

        backend = AudioBackend(
            "WARNING", None, AudioSettings(voice_volume=0.42, sound_enabled=False)
        )
        backend._play_wav = lambda path: None  # type: ignore[method-assign]
        with patch.dict(sys.modules, {"pyttsx3": _FakeTTS()}):
            backend.run()
        self.assertEqual(captured.get("volume"), 0.42)


class TrendSeriesTests(unittest.TestCase):
    def _samples(self):
        return [
            AlertnessSample(0.0, 100.0, 0.0, 5.0, 2.0, 0.0, "SAFE", ""),
            AlertnessSample(1.0, 80.0, 20.0, 30.0, 40.0, 10.0, "MODERATE", "x"),
            AlertnessSample(2.0, 60.0, 40.0, 55.0, 60.0, 20.0, "HIGH", "y"),
        ]

    def test_trend_series_produces_distinct_series(self) -> None:
        data = trend_series(self._samples())
        self.assertEqual(len(data["x"]), 3)
        self.assertEqual(len(data["alertness"]), 3)
        self.assertEqual(len(data["fatigue"]), 3)
        self.assertEqual(len(data["distraction"]), 3)
        self.assertEqual(len(data["environmental"]), 3)
        self.assertEqual(data["alertness"], [100.0, 80.0, 60.0])
        self.assertEqual(data["fatigue"], [5.0, 30.0, 55.0])
        self.assertEqual(data["distraction"], [2.0, 40.0, 60.0])

    def test_trend_series_preserves_missing_values(self) -> None:
        samples = self._samples()
        samples[1] = AlertnessSample(1.0, None, None, None, None, None, "SAFE", "")
        data = trend_series(samples)
        self.assertIsNone(data["alertness"][1])
        self.assertIsNone(data["fatigue"][1])
        self.assertIsNotNone(data["alertness"][0])


class AlertDecisionLoggingTests(unittest.TestCase):
    def test_format_contains_required_fields(self) -> None:
        line = format_alert_decision(
            severity="WARNING",
            reason="sustained head-turn distraction",
            alertness=62.0,
            fatigue=20.0,
            distraction=58.0,
            context_urgency=0.85,
            head_direction="left",
            eye_state="OPEN",
            face_detected=True,
            audio="PLAY",
        )
        for token in (
            "ALERT DECISION",
            "severity=WARNING",
            "reason=sustained head-turn distraction",
            "alertness=62",
            "fatigue=20",
            "distraction=58",
            "context_urgency=0.85",
            "head=left",
            "eye=OPEN",
            "face_detected=True",
            "audio=PLAY",
        ):
            self.assertIn(token, line)

    def test_format_suppressed_marks_suppressed(self) -> None:
        line = format_alert_decision(
            severity="WARNING",
            reason="x",
            alertness=62.0,
            fatigue=20.0,
            distraction=58.0,
            context_urgency=1.0,
            head_direction="forward",
            eye_state="OPEN",
            face_detected=True,
            audio="NONE",
            suppressed=True,
            suppression_reason="same_event_cooldown",
        )
        self.assertIn("ALERT SUPPRESSED", line)
        self.assertIn("suppression_reason=same_event_cooldown", line)


class AlertnessSemanticsTests(unittest.TestCase):
    def test_alertness_0_to_100_and_complements_risk(self) -> None:
        eng = AttentionEngine()
        out = eng.update(fatigue_risk=70.0, distraction=None, now_s=0.0)
        self.assertGreaterEqual(out.alertness_score, 0.0)
        self.assertLessEqual(out.alertness_score, 100.0)
        self.assertGreaterEqual(out.combined_risk, 0.0)
        self.assertLessEqual(out.combined_risk, 100.0)
        self.assertAlmostEqual(out.alertness_score + out.combined_risk, 100.0, places=1)

    def test_brief_improvement_does_not_collapse_alertness(self) -> None:
        eng = AttentionEngine()
        first = eng.update(fatigue_risk=70.0, distraction=None, now_s=0.0)
        low = first.alertness_score
        second = eng.update(fatigue_risk=0.0, distraction=None, now_s=0.2)
        self.assertLess(second.alertness_score, 80.0)
        self.assertEqual(second.alertness_score, low)


if __name__ == "__main__":
    unittest.main()