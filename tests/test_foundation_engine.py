"""Regression tests for the multi-factor fatigue foundation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fatigue_engine import FatigueEngine, FatigueEvidence  # noqa: E402


def E(**kwargs) -> FatigueEvidence:
    return FatigueEvidence(**kwargs)


class FoundationRiskTests(unittest.TestCase):
    def test_one_blink_is_negligible(self) -> None:
        engine = FatigueEngine()
        decision = engine.decide(E(blink_count=1, face_detected=True))
        self.assertEqual(decision.status, "SAFE")
        self.assertEqual(decision.severity, "NORMAL")
        self.assertLessEqual(decision.fatigue_score, 10)

    def test_one_yawn_is_small_and_not_drowsy(self) -> None:
        engine = FatigueEngine()
        decision = engine.decide(E(yawn_detected=True, yawn_count=1, face_detected=True))
        self.assertEqual(decision.status, "SAFE")
        self.assertIn("yawn", " ".join(decision.reasons).lower())

    def test_repeated_yawns_raise_risk(self) -> None:
        engine = FatigueEngine()
        decision = engine.decide(E(yawn_detected=True, yawn_count=6, recent_yawn_frequency=2.5, face_detected=True))
        self.assertGreater(decision.fatigue_score, 10)
        self.assertGreater(decision.smoothed_risk or 0, 10)

    def test_prolonged_closure_is_highest_direct_risk(self) -> None:
        engine = FatigueEngine()
        prolonged = engine.decide(E(prolonged_eye_closure=True, current_eye_closure_seconds=2.5, face_detected=True))
        blink_only = engine.decide(E(blink_count=3, face_detected=True))
        yawn_only = engine.decide(E(yawn_detected=True, yawn_count=1, face_detected=True))
        self.assertGreater(prolonged.fatigue_score or 0, max(blink_only.fatigue_score or 0, yawn_only.fatigue_score or 0))
        self.assertGreater(prolonged.breakdown["eye_risk"], prolonged.breakdown["blink_risk"])
        self.assertGreater(prolonged.breakdown["eye_risk"], prolonged.breakdown["yawn_risk"])

    def test_abnormal_blink_rate_is_supporting_risk(self) -> None:
        engine = FatigueEngine()
        decision = engine.decide(E(blink_count=12, blink_rate=30.0, recent_blink_rate=30.0, face_detected=True))
        self.assertGreater(decision.breakdown["blink_risk"], 0.0)
        self.assertEqual(decision.status, "SAFE")

    def test_yawn_decay_is_not_permanent(self) -> None:
        engine = FatigueEngine()
        first = engine.update(E(yawn_detected=True, yawn_count=3, recent_yawn_frequency=2.5, face_detected=True), 0.0)
        second = engine.update(E(yawn_detected=False, yawn_count=3, recent_yawn_frequency=0.0, face_detected=True), 4.0)
        self.assertLess(second.breakdown["yawn_risk"], first.breakdown["yawn_risk"])

    def test_no_face_reports_unavailable_risk(self) -> None:
        engine = FatigueEngine()
        decision = engine.decide(E(face_detected=False))
        self.assertIsNone(decision.fatigue_score)
        self.assertEqual(decision.status, "SAFE")
        self.assertEqual(decision.severity, "NORMAL")

    def test_face_return_resets_stale_state(self) -> None:
        engine = FatigueEngine()
        engine.update(E(prolonged_eye_closure=True, current_eye_closure_seconds=3.0, face_detected=True), 0.0)
        engine.update(E(face_detected=False), 0.2)
        decision = engine.update(E(face_detected=True), 0.4)
        self.assertEqual(decision.status, "SAFE")
        self.assertIsNone(decision.fatigue_score)

    def test_warning_persists_before_entering_warning(self) -> None:
        engine = FatigueEngine(
            warning_entry=20,
            warning_persistence_s=1.0,
            drowsy_entry=70,
            drowsy_persistence_s=1.0,
        )
        decision = engine.update(E(yawn_detected=True, yawn_count=2, night_context=True, driving_duration_risk=True, driving_duration_seconds=5 * 3600, face_detected=True), 0.0)
        self.assertEqual(decision.status, "SAFE")
        self.assertEqual(decision.severity, "NORMAL")

        decision = engine.update(E(yawn_detected=True, yawn_count=2, night_context=True, driving_duration_risk=True, driving_duration_seconds=5 * 3600, face_detected=True), 0.5)
        self.assertEqual(decision.status, "SAFE")
        decision = engine.update(E(yawn_detected=True, yawn_count=2, night_context=True, driving_duration_risk=True, driving_duration_seconds=5 * 3600, face_detected=True), 1.5)
        self.assertEqual(decision.status, "WARNING")

    def test_prolonged_eye_closure_respects_drowsy_persistence(self) -> None:
        engine = FatigueEngine(drowsy_persistence_s=1.0, warning_persistence_s=0.0)
        decision = engine.update(E(prolonged_eye_closure=True, current_eye_closure_seconds=2.5, face_detected=True), 0.0)
        self.assertEqual(decision.status, "SAFE")
        decision = engine.update(E(prolonged_eye_closure=True, current_eye_closure_seconds=2.5, face_detected=True), 0.5)
        self.assertEqual(decision.status, "SAFE")
        decision = engine.update(E(prolonged_eye_closure=True, current_eye_closure_seconds=2.5, face_detected=True), 1.5)
        self.assertEqual(decision.status, "DROWSY")

    def test_duration_ramps_after_three_hours(self) -> None:
        engine = FatigueEngine()
        at_three_hours = engine.decide(E(driving_duration_risk=True, driving_duration_seconds=3 * 3600, face_detected=True))
        at_four_hours = engine.decide(E(driving_duration_risk=True, driving_duration_seconds=4 * 3600, face_detected=True))
        at_five_hours = engine.decide(E(driving_duration_risk=True, driving_duration_seconds=5 * 3600, face_detected=True))
        self.assertGreaterEqual(at_three_hours.breakdown["duration_risk"], 2.0)
        self.assertGreater(at_four_hours.breakdown["duration_risk"], at_three_hours.breakdown["duration_risk"])
        self.assertGreater(at_five_hours.breakdown["duration_risk"], at_four_hours.breakdown["duration_risk"])

    def test_old_yawns_do_not_create_high_current_yawn_risk(self) -> None:
        engine = FatigueEngine()
        decision = engine.decide(E(yawn_detected=False, yawn_count=40, recent_yawn_frequency=0.0, face_detected=True))
        self.assertEqual(decision.breakdown["yawn_risk"], 0.0)

    def test_lifetime_yawn_count_is_not_sufficient_for_severe_fatigue(self) -> None:
        engine = FatigueEngine()
        decision = engine.decide(E(yawn_detected=False, yawn_count=50, face_detected=True))
        self.assertLess(decision.breakdown["yawn_risk"], 8.0)
        self.assertEqual(decision.status, "SAFE")

    def test_drowsy_recovery_requires_sustained_recovery(self) -> None:
        engine = FatigueEngine(
            warning_entry=50,
            drowsy_entry=70,
            warning_recovery=35,
            drowsy_recovery=50,
            drowsy_persistence_s=0.0,
            drowsy_recovery_s=1.0,
        )
        engine.update(E(prolonged_eye_closure=True, current_eye_closure_seconds=2.5, face_detected=True), 0.0)
        decision = engine.update(E(face_detected=True), 0.5)
        self.assertEqual(decision.status, "DROWSY")
        decision = engine.update(E(face_detected=True), 1.5)
        self.assertEqual(decision.status, "SAFE")

    def test_risk_holds_on_brief_improvement(self) -> None:
        engine = FatigueEngine(risk_decrease_hold_s=0.40, risk_decrease_tau_s=1.0)
        high = engine.update(
            E(prolonged_eye_closure=True, current_eye_closure_seconds=2.0, face_detected=True),
            0.0,
        )
        self.assertGreaterEqual(high.smoothed_risk or 0, 70)
        brief = engine.update(E(face_detected=True), 0.2)
        self.assertAlmostEqual(brief.smoothed_risk or 0, high.smoothed_risk or 0, places=2)

    def test_risk_decreases_gradually_after_sustained_improvement(self) -> None:
        engine = FatigueEngine(risk_decrease_hold_s=0.40, risk_decrease_tau_s=1.0)
        engine.update(
            E(prolonged_eye_closure=True, current_eye_closure_seconds=2.0, face_detected=True),
            0.0,
        )
        held = engine.update(E(face_detected=True), 0.2)
        after = engine.update(E(face_detected=True), 1.0)
        later = engine.update(E(face_detected=True), 2.0)
        self.assertLess(after.smoothed_risk or 0, held.smoothed_risk or 0)
        self.assertLess(later.smoothed_risk or 0, after.smoothed_risk or 0)
        self.assertGreaterEqual(later.smoothed_risk or 0, 0.0)
        self.assertLessEqual(later.smoothed_risk or 0, 100.0)

    def test_risk_increase_is_not_blocked(self) -> None:
        engine = FatigueEngine(risk_decrease_hold_s=0.40, risk_decrease_tau_s=1.0)
        low = engine.update(E(face_detected=True), 0.0)
        high = engine.update(
            E(prolonged_eye_closure=True, current_eye_closure_seconds=2.0, face_detected=True),
            0.1,
        )
        self.assertGreater(high.smoothed_risk or 0, low.smoothed_risk or 0)

    def test_danger_return_stops_risk_decrease(self) -> None:
        engine = FatigueEngine(risk_decrease_hold_s=0.40, risk_decrease_tau_s=1.0)
        engine.update(
            E(prolonged_eye_closure=True, current_eye_closure_seconds=2.0, face_detected=True),
            0.0,
        )
        engine.update(E(yawn_detected=True, yawn_count=1, face_detected=True), 0.5)
        mid = engine.update(E(face_detected=True), 1.0)
        # Partial danger returns (yawn again) while still below peak — freeze decrease.
        stopped = engine.update(E(yawn_detected=True, yawn_count=2, face_detected=True), 1.2)
        self.assertGreaterEqual(stopped.smoothed_risk or 0, mid.smoothed_risk or 0)


if __name__ == "__main__":
    unittest.main()
