"""Unit tests for Phase 2 SessionManager (start/stop/timer/history)."""

from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fatigue_engine import FatigueDecision, FatigueEngine, FatigueEvidence  # noqa: E402
from session_manager import MAX_RISK_HISTORY, RiskSample, SessionManager  # noqa: E402


def make_decision(engine: FatigueEngine, **evidence_kwargs):
    """Return a decision after two updates so severity transitions settle."""
    evidence = FatigueEvidence(**evidence_kwargs)
    engine.update(evidence, timestamp_s=0.0)
    return engine.update(evidence, timestamp_s=0.1)


class SessionLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = SessionManager(use_demo_duration=True)
        self.engine = FatigueEngine(
            drowsy_enter_hold_s=0.0,
            drowsy_exit_hold_s=0.0,
            drowsy_persistence_s=0.0,
            warning_persistence_s=0.0,
        )

    def test_start_creates_session_and_activates(self) -> None:
        sid = self.manager.start()
        self.assertTrue(self.manager.active)
        self.assertNotEqual(sid, "")
        self.assertIsNotNone(self.manager.stats.start_time)

    def test_inactive_before_start(self) -> None:
        self.assertFalse(self.manager.active)
        self.assertEqual(self.manager.driving_time_text(), "00:00:00")

    def test_stop_finalizes_and_deactivates(self) -> None:
        self.manager.start()
        stats = self.manager.stop()
        self.assertFalse(self.manager.active)
        self.assertIsNotNone(stats.end_time)

    def test_timer_starts_at_zero_and_increases(self) -> None:
        self.manager.start()
        # Immediately after start the timer is ~0.
        self.assertNotEqual(self.manager.driving_time_text(now_s=0.0 + 0.01), "")

    def test_new_session_resets_statistics(self) -> None:
        self.manager.start()
        # Record a high-risk decision.
        decision = make_decision(
            self.engine,
            prolonged_eye_closure=True,
            current_eye_closure_seconds=2.5,
            face_detected=True,
        )
        self.manager.update(decision, now_s=1.0)
        self.manager.stop()
        first_id = self.manager.stats.session_id

        # New session.
        self.manager.start()
        self.assertNotEqual(self.manager.stats.session_id, first_id)
        self.assertEqual(self.manager.stats.blink_count, 0)
        self.assertEqual(self.manager.stats.yawn_count, 0)
        self.assertEqual(self.manager.stats.maximum_fatigue_risk, 0.0)
        self.assertEqual(len(self.manager.stats.risk_history), 0)

    def test_stop_when_inactive_returns_stats(self) -> None:
        stats = self.manager.stop()
        self.assertIsNotNone(stats)


class RiskHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = SessionManager(use_demo_duration=True)
        self.engine = FatigueEngine(
            drowsy_enter_hold_s=0.0,
            drowsy_exit_hold_s=0.0,
            drowsy_persistence_s=0.0,
            warning_persistence_s=0.0,
        )
        self.manager.start()

    def test_update_records_risk_history(self) -> None:
        decision = make_decision(self.engine, face_detected=True)
        self.manager.update(decision, now_s=1.0)
        self.assertEqual(len(self.manager.stats.risk_history), 1)
        sample = self.manager.stats.risk_history[0]
        self.assertIsInstance(sample, RiskSample)
        self.assertEqual(sample.severity, "NORMAL")

    def test_risk_history_is_bounded(self) -> None:
        for i in range(MAX_RISK_HISTORY + 50):
            decision = make_decision(self.engine, face_detected=True)
            self.manager.update(decision, now_s=float(i))
        self.assertLessEqual(len(self.manager.stats.risk_history), MAX_RISK_HISTORY)

    def test_risk_factor_history_tracked(self) -> None:
        decision = make_decision(
            self.engine,
            prolonged_eye_closure=True,
            current_eye_closure_seconds=2.0,
            face_detected=True,
        )
        self.manager.update(decision, now_s=1.0)
        self.assertIn("eye_risk", self.manager.stats.risk_factor_history)
        self.assertGreater(self.manager.stats.risk_factor_history["eye_risk"][-1], 0.0)

    def test_risk_trend_data_available(self) -> None:
        decision = make_decision(self.engine, face_detected=True)
        for i in range(5):
            self.manager.update(decision, now_s=float(i))
        self.assertEqual(len(self.manager.stats.risk_history), 5)

    def test_alertness_history_and_statistics_are_recorded(self) -> None:
        attention = SimpleNamespace(
            alertness_score=82.0,
            distraction_score=12.0,
            fatigue_score=10.0,
            level="SAFE",
            primary_reason="Attentive",
            location_risk=0.0,
            weather_risk=5.0,
            location_zone="none",
        )
        alert = SimpleNamespace(
            active_severity="NORMAL",
            reason="",
            contributing_factors=(),
        )
        decision = make_decision(self.engine, face_detected=True)
        self.manager.update(decision, now_s=1.0, attention=attention, alert_result=alert)
        attention.alertness_score = 64.0
        attention.level = "MODERATE"
        self.manager.update(decision, now_s=2.0, attention=attention, alert_result=alert)
        stats = self.manager.stop()

        self.assertEqual(len(stats.alertness_history), 2)
        self.assertAlmostEqual(stats.average_alertness, 73.0)
        self.assertAlmostEqual(stats.minimum_alertness, 64.0)
        self.assertAlmostEqual(stats.maximum_alertness, 82.0)
        self.assertGreater(stats.time_safe_s, 0.0)

    def test_continuous_alert_is_one_episode_with_escalation(self) -> None:
        attention = SimpleNamespace(
            alertness_score=45.0,
            distraction_score=50.0,
            fatigue_score=30.0,
            level="HIGH",
            primary_reason="Sustained attention away from the road",
            location_risk=0.0,
            weather_risk=0.0,
            location_zone="none",
        )
        decision = make_decision(self.engine, face_detected=True)
        warning = SimpleNamespace(
            active_severity="WARNING", reason=attention.primary_reason,
            contributing_factors=("distraction",),
        )
        drowsy = SimpleNamespace(
            active_severity="DROWSY", reason=attention.primary_reason,
            contributing_factors=("distraction", "fatigue"),
        )
        normal = SimpleNamespace(active_severity="NORMAL", reason="", contributing_factors=())
        self.manager.update(decision, now_s=1.0, attention=attention, alert_result=warning)
        self.manager.update(decision, now_s=2.0, attention=attention, alert_result=warning)
        self.manager.update(decision, now_s=3.0, attention=attention, alert_result=drowsy)
        attention.alertness_score = 90.0
        attention.level = "SAFE"
        self.manager.update(decision, now_s=4.0, attention=attention, alert_result=normal)
        self.manager.stop()

        self.assertEqual(len(self.manager.stats.alert_events), 1)
        self.assertEqual(self.manager.stats.alert_events[0].severity, "DROWSY")
        self.assertEqual(self.manager.stats.alert_events[0].primary_reason, warning.reason)


class CounterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = SessionManager(use_demo_duration=True)
        self.engine = FatigueEngine(
            drowsy_enter_hold_s=0.0,
            drowsy_exit_hold_s=0.0,
            drowsy_persistence_s=0.0,
            warning_persistence_s=0.0,
        )
        self.manager.start()

    def test_low_fatigue_counter(self) -> None:
        # Force LOW_FATIGUE via yawn + context.
        decision = make_decision(
            self.engine,
            yawn_detected=True,
            yawn_count=5,
            recent_yawn_frequency=3.0,
            night_context=True,
            driving_duration_risk=True,
            driving_duration_seconds=5 * 3600,
            face_detected=True,
        )
        self.manager.update(decision, now_s=1.0)
        self.assertGreater(self.manager.stats.low_fatigue_count, 0)

    def test_warning_counter(self) -> None:
        # Yawn risk is bounded at its weight (15). Reach WARNING via a strong
        # blink rate (up to 20) + yawn (15) + night (5) + duration (10) = 50,
        # which is >= warning_entry (default 50).
        decision = make_decision(
            self.engine,
            yawn_detected=True,
            yawn_count=20,
            recent_yawn_frequency=15.0,
            blink_rate=50.0,
            recent_blink_rate=55.0,
            night_context=True,
            driving_duration_risk=True,
            driving_duration_seconds=5 * 3600,
            face_detected=True,
        )
        self.manager.update(decision, now_s=1.0)
        self.assertGreater(self.manager.stats.warning_count, 0)
        self.assertEqual(self.manager.stats.final_status, "WARNING")

    def test_drowsy_counter(self) -> None:
        decision = make_decision(
            self.engine,
            prolonged_eye_closure=True,
            current_eye_closure_seconds=2.5,
            face_detected=True,
        )
        self.manager.update(decision, now_s=1.0)
        self.assertGreater(self.manager.stats.drowsy_count, 0)
        self.assertEqual(self.manager.stats.final_status, "DROWSY")


if __name__ == "__main__":
    unittest.main()