"""
Unit tests for the 5-level progressive alert system.

Covers the ALERT severity level (between LOW FATIGUE and WARNING):
- FatigueEngine severity mapping to ALERT.
- AlertEscalationController ALERT entry / priority.
- AudioEngine ALERT priority replacement.
- Weak events (one yawn, night, duration) do not reach DROWSY.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alert_escalation import AlertEscalationController  # noqa: E402
from audio_engine import AudioEngine, AudioSettings  # noqa: E402
from fatigue_engine import FatigueEngine, FatigueEvidence  # noqa: E402


def E(**kwargs) -> FatigueEvidence:
    return FatigueEvidence(**kwargs)


def _feed(engine: FatigueEngine, evidence: FatigueEvidence) -> "object":
    """Drive the engine with two updates so a severity change latches."""
    engine.update(evidence, 0.0)
    return engine.update(evidence, 1.0)


class FatigueSeverityMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = FatigueEngine(
            drowsy_enter_hold_s=0.0, drowsy_exit_hold_s=0.0
        )

    def test_normal_band(self) -> None:
        # Score 0-19 -> NORMAL.
        self.assertEqual(_feed(self.engine, E()).severity, "NORMAL")
        self.assertEqual(_feed(self.engine, E(night_context=True)).severity, "NORMAL")  # 5

    def test_low_fatigue_band(self) -> None:
        # night(5) + duration(10) + yawn(15) = 30 -> LOW_FATIGUE.
        d = _feed(
            self.engine,
            E(yawn_detected=True, yawn_count=1, night_context=True,
              driving_duration_risk=True, driving_duration_seconds=5 * 3600),
        )
        self.assertEqual(d.fatigue_score, 30)
        self.assertEqual(d.severity, "LOW_FATIGUE")

    def test_alert_band(self) -> None:
        # yawn(15) + night(5) + duration(10) + blink_risk(9.6) = ~40 -> ALERT.
        d = _feed(
            self.engine,
            E(yawn_detected=True, yawn_count=10, recent_yawn_frequency=15.0,
              night_context=True, driving_duration_risk=True,
              driving_duration_seconds=5 * 3600,
              blink_rate=30.0, recent_blink_rate=30.0),
        )
        self.assertGreaterEqual(d.fatigue_score, 35)
        self.assertLess(d.fatigue_score, 50)
        self.assertEqual(d.severity, "ALERT")

    def test_warning_band(self) -> None:
        # yawn(15) + blink(20) + night(5) + duration(10) = 50 -> WARNING.
        d = _feed(
            self.engine,
            E(yawn_detected=True, yawn_count=10, recent_yawn_frequency=15.0,
              night_context=True, driving_duration_risk=True,
              driving_duration_seconds=5 * 3600,
              blink_rate=100.0, recent_blink_rate=100.0),
        )
        self.assertGreaterEqual(d.fatigue_score, 50)
        self.assertLess(d.fatigue_score, 70)
        self.assertEqual(d.severity, "WARNING")

    def test_drowsy_band(self) -> None:
        # Prolonged eye closure (70) -> DROWSY.
        d = _feed(self.engine, E(prolonged_eye_closure=True, current_eye_closure_seconds=2.5))
        self.assertEqual(d.fatigue_score, 70)
        self.assertEqual(d.severity, "DROWSY")

    def test_weak_events_do_not_reach_drowsy(self) -> None:
        # One normal blink.
        self.assertEqual(_feed(self.engine, E(blink_count=1)).severity, "NORMAL")
        # One small yawn.
        self.assertNotEqual(
            _feed(self.engine, E(yawn_detected=True, yawn_count=1)).severity, "DROWSY"
        )
        # Night only.
        self.assertNotEqual(_feed(self.engine, E(night_context=True)).severity, "DROWSY")
        # 3-hour duration only.
        self.assertNotEqual(
            _feed(self.engine, E(driving_duration_risk=True, driving_duration_seconds=3 * 3600)).severity,
            "DROWSY",
        )


class AlertEscalationALERTTests(unittest.TestCase):
    def test_alert_entry_triggers_audio_once(self) -> None:
        ctl = AlertEscalationController()
        r = ctl.update("ALERT", 1.0)
        self.assertTrue(r.should_play_audio)
        self.assertEqual(r.audio_severity, "ALERT")

    def test_alert_continues_no_repeat(self) -> None:
        ctl = AlertEscalationController()
        ctl.update("ALERT", 1.0)
        for t in (1.1, 1.5, 2.0, 3.0):
            r = ctl.update("ALERT", t)
            self.assertFalse(r.should_play_audio)

    def test_alert_replaces_low_fatigue(self) -> None:
        ctl = AlertEscalationController()
        ctl.update("LOW_FATIGUE", 0.0)
        r = ctl.update("ALERT", 1.0)
        self.assertEqual(r.active_severity, "ALERT")
        self.assertTrue(r.should_play_audio)

    def test_warning_replaces_alert(self) -> None:
        ctl = AlertEscalationController()
        ctl.update("ALERT", 0.0)
        r = ctl.update("WARNING", 1.0)
        self.assertEqual(r.active_severity, "WARNING")
        self.assertTrue(r.should_play_audio)

    def test_low_fatigue_does_not_override_alert(self) -> None:
        ctl = AlertEscalationController()
        ctl.update("ALERT", 0.0)
        r = ctl.update("LOW_FATIGUE", 1.0)
        self.assertFalse(r.should_play_audio)


class AudioEngineALERTTests(unittest.TestCase):
    def _engine(self) -> AudioEngine:
        return AudioEngine(
            AudioSettings(
                alert_volume=0.5,
                alarm_volume=0.5,
                voice_enabled=False,
                sound_enabled=False,
            )
        )

    def test_alert_replaces_low_fatigue_in_audio(self) -> None:
        engine = self._engine()
        engine.play_alert("LOW_FATIGUE")
        self.assertEqual(engine.active_severity, "LOW_FATIGUE")
        engine.play_alert("ALERT")
        self.assertEqual(engine.active_severity, "ALERT")
        engine.shutdown()

    def test_warning_replaces_alert_in_audio(self) -> None:
        engine = self._engine()
        engine.play_alert("ALERT")
        self.assertEqual(engine.active_severity, "ALERT")
        engine.play_alert("WARNING")
        self.assertEqual(engine.active_severity, "WARNING")
        engine.shutdown()

    def test_drowsy_cannot_be_replaced_by_alert(self) -> None:
        engine = self._engine()
        engine.play_alert("DROWSY")
        engine.play_alert("ALERT")
        self.assertEqual(engine.active_severity, "DROWSY")
        engine.shutdown()

    def test_alert_does_not_replace_drowsy(self) -> None:
        engine = self._engine()
        engine.play_alert("DROWSY")
        engine.play_alert("ALERT")
        self.assertEqual(engine.active_severity, "DROWSY")
        engine.shutdown()

    def test_alert_duration_is_five_seconds(self) -> None:
        from predefined_sounds import ALERT_ALARM_DURATION_SECONDS

        self.assertEqual(ALERT_ALARM_DURATION_SECONDS, 1.5)

    def test_low_fatigue_uses_soft_beep_then_voice(self) -> None:
        engine = AudioEngine(
            AudioSettings(
                alert_volume=0.5,
                alarm_volume=0.5,
                voice_enabled=True,
                sound_enabled=True,
            )
        )
        self.assertEqual(engine._sound_for("LOW_FATIGUE"), "soft_beep_2")
        engine.shutdown()


if __name__ == "__main__":
    unittest.main()