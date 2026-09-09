"""Unit tests for Stage 7A standalone fatigue decision engine."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fatigue_engine import (  # noqa: E402
    BLINK_ANOMALY_MAX,
    BLINK_HIGH_RATE_THRESHOLD,
    BLINK_LOW_RATE_THRESHOLD,
    RECENT_YAWN_MAX,
    SAFE_SCORE_MAX,
    WARNING_SCORE_MAX,
    WEIGHT_DRIVING_DURATION_RISK,
    WEIGHT_NIGHT_CONTEXT,
    WEIGHT_PROLONGED_EYE_CLOSURE,
    WEIGHT_YAWN_EVENT,
    FatigueEngine,
    FatigueEvidence,
)


def E(**kwargs) -> FatigueEvidence:
    return FatigueEvidence(**kwargs)


class ScoreAndThresholdTests(unittest.TestCase):
    def setUp(self) -> None:
        # Zero hold times for score/threshold tests (except flicker suite).
        self.engine = FatigueEngine(drowsy_enter_hold_s=0.0, drowsy_exit_hold_s=0.0)

    def test_normal_eyes_safe(self) -> None:
        d = self.engine.decide(E())
        self.assertEqual(d.status, "SAFE")
        self.assertEqual(d.fatigue_score, 0)
        self.assertIn("Eyes normal", d.reasons)

    def test_one_normal_blink_safe(self) -> None:
        d = self.engine.decide(E(blink_count=1))
        self.assertEqual(d.status, "SAFE")
        self.assertEqual(d.fatigue_score, 0)

    def test_yawn_alone_not_drowsy(self) -> None:
        d = self.engine.decide(E(yawn_detected=True, yawn_count=1))
        self.assertNotEqual(d.status, "DROWSY")
        self.assertEqual(d.fatigue_score, WEIGHT_YAWN_EVENT)
        self.assertIn("Yawning detected", d.reasons)

    def test_night_alone_not_drowsy(self) -> None:
        d = self.engine.decide(E(night_context=True))
        self.assertNotEqual(d.status, "DROWSY")
        self.assertEqual(d.fatigue_score, WEIGHT_NIGHT_CONTEXT)
        self.assertEqual(d.status, "SAFE")

    def test_driving_alone_not_drowsy(self) -> None:
        d = self.engine.decide(E(driving_duration_risk=True, driving_duration_seconds=0.0))
        self.assertNotEqual(d.status, "DROWSY")
        self.assertEqual(d.fatigue_score, 0)
        self.assertEqual(d.status, "SAFE")

    def test_night_plus_driving_not_drowsy(self) -> None:
        d = self.engine.decide(E(night_context=True, driving_duration_risk=True, driving_duration_seconds=0.0))
        self.assertNotEqual(d.status, "DROWSY")
        expected = WEIGHT_NIGHT_CONTEXT
        self.assertEqual(d.fatigue_score, expected)
        self.assertEqual(d.status, "SAFE")

    def test_prolonged_eye_closure_drowsy(self) -> None:
        d = self.engine.decide(
            E(prolonged_eye_closure=True, current_eye_closure_seconds=2.0)
        )
        self.assertEqual(d.status, "DROWSY")
        self.assertEqual(d.fatigue_score, WEIGHT_PROLONGED_EYE_CLOSURE)
        self.assertTrue(any("Prolonged eye closure" in r for r in d.reasons))

    def test_prolonged_plus_yawn_drowsy(self) -> None:
        d = self.engine.decide(
            E(prolonged_eye_closure=True, yawn_detected=True, yawn_count=1)
        )
        self.assertEqual(d.status, "DROWSY")
        self.assertEqual(
            d.fatigue_score,
            WEIGHT_PROLONGED_EYE_CLOSURE + WEIGHT_YAWN_EVENT,
        )

    def test_prolonged_plus_night_drowsy(self) -> None:
        d = self.engine.decide(E(prolonged_eye_closure=True, night_context=True))
        self.assertEqual(d.status, "DROWSY")
        self.assertIn("Prolonged eye closure + night context", d.reasons)

    def test_prolonged_plus_driving_drowsy(self) -> None:
        d = self.engine.decide(
            E(prolonged_eye_closure=True, driving_duration_risk=True)
        )
        self.assertEqual(d.status, "DROWSY")
        self.assertIn("Prolonged eye closure + extended driving", d.reasons)

    def test_all_risk_factors_drowsy(self) -> None:
        d = self.engine.decide(
            E(
                prolonged_eye_closure=True,
                current_eye_closure_seconds=2.5,
                yawn_detected=True,
                yawn_count=3,
                night_context=True,
                driving_duration_risk=True,
                driving_duration_seconds=5 * 3600,
                blink_count=40,
            )
        )
        self.assertEqual(d.status, "DROWSY")
        expected = (
            WEIGHT_PROLONGED_EYE_CLOSURE
            + WEIGHT_YAWN_EVENT
            + WEIGHT_NIGHT_CONTEXT
            + WEIGHT_DRIVING_DURATION_RISK
        )
        self.assertEqual(d.fatigue_score, expected)
        self.assertIn(
            "Prolonged eye closure + night + extended driving",
            d.reasons,
        )

    def test_score_thresholds(self) -> None:
        self.assertEqual(SAFE_SCORE_MAX, 30)
        self.assertEqual(WARNING_SCORE_MAX, 60)
        eng = self.engine
        self.assertEqual(eng.status_from_score(0), "SAFE")
        self.assertEqual(eng.status_from_score(29), "SAFE")
        self.assertEqual(eng.status_from_score(30), "WARNING")
        self.assertEqual(eng.status_from_score(59), "WARNING")
        self.assertEqual(eng.status_from_score(60), "DROWSY")

    def test_repeated_frames_do_not_accumulate_infinite_score(self) -> None:
        evidence = E(
            prolonged_eye_closure=True,
            yawn_detected=True,
            night_context=True,
            driving_duration_risk=True,
            driving_duration_seconds=5 * 3600,
        )
        expected = (
            WEIGHT_PROLONGED_EYE_CLOSURE
            + WEIGHT_YAWN_EVENT
            + WEIGHT_NIGHT_CONTEXT
            + WEIGHT_DRIVING_DURATION_RISK
        )
        scores = []
        for i in range(50):
            d = self.engine.update(evidence, timestamp_s=float(i))
            scores.append(d.fatigue_score)
        self.assertTrue(all(s == expected for s in scores))
        self.assertEqual(scores[-1], expected)


class WarningBandTests(unittest.TestCase):
    def test_yawn_plus_context_can_reach_warning(self) -> None:
        # 15+5+10 = 30 -> WARNING boundary with initial weights
        eng = FatigueEngine(drowsy_enter_hold_s=0.0, drowsy_exit_hold_s=0.0)
        d = eng.decide(
            E(yawn_detected=True, night_context=True, driving_duration_risk=True, driving_duration_seconds=5 * 3600)
        )
        self.assertEqual(d.fatigue_score, 30)
        self.assertEqual(d.status, "WARNING")
        self.assertNotEqual(d.status, "DROWSY")


class TemporalStabilityTests(unittest.TestCase):
    def test_prolonged_enters_drowsy_immediately(self) -> None:
        eng = FatigueEngine(drowsy_enter_hold_s=0.0, drowsy_exit_hold_s=0.8, drowsy_persistence_s=0.0)
        d = eng.update(E(prolonged_eye_closure=True), timestamp_s=1.0)
        self.assertEqual(d.status, "DROWSY")
        self.assertEqual(d.raw_status, "DROWSY")

    def test_recovery_hold_before_leaving_drowsy(self) -> None:
        eng = FatigueEngine(drowsy_enter_hold_s=0.0, drowsy_exit_hold_s=0.80)
        eng.update(E(prolonged_eye_closure=True), timestamp_s=0.0)
        # Eyes open again — still DROWSY during recovery window
        d1 = eng.update(E(), timestamp_s=0.20)
        self.assertEqual(d1.raw_status, "SAFE")
        self.assertEqual(d1.status, "DROWSY")
        d2 = eng.update(E(), timestamp_s=0.99)
        self.assertEqual(d2.status, "DROWSY")
        # Recovery started at t=0.20; exit hold is 0.80s -> ready at t=1.00
        d3 = eng.update(E(), timestamp_s=1.00)
        self.assertEqual(d3.status, "SAFE")

    def test_status_does_not_flicker_on_brief_recovery(self) -> None:
        eng = FatigueEngine(drowsy_enter_hold_s=0.0, drowsy_exit_hold_s=1.0)
        eng.update(E(prolonged_eye_closure=True), timestamp_s=0.0)
        # Brief clear then prolonged again should stay DROWSY
        eng.update(E(), timestamp_s=0.10)
        d = eng.update(E(prolonged_eye_closure=True), timestamp_s=0.20)
        self.assertEqual(d.status, "DROWSY")

    def test_non_prolonged_drowsy_requires_enter_hold(self) -> None:
        # Force a DROWSY raw score without prolonged by raising yawn weight.
        eng = FatigueEngine(
            weight_yawn=70,
            drowsy_enter_hold_s=0.40,
            drowsy_exit_hold_s=0.0,
        )
        d0 = eng.update(E(yawn_detected=True), timestamp_s=0.0)
        self.assertEqual(d0.raw_status, "DROWSY")
        self.assertEqual(d0.status, "SAFE")  # hold not yet elapsed
        d1 = eng.update(E(yawn_detected=True), timestamp_s=0.39)
        self.assertEqual(d1.status, "SAFE")
        d2 = eng.update(E(yawn_detected=True), timestamp_s=0.40)
        self.assertEqual(d2.status, "DROWSY")


class DecayingRecentEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = FatigueEngine(drowsy_enter_hold_s=0.0, drowsy_exit_hold_s=0.0)

    def test_single_yawn_contributes_single_weight(self) -> None:
        # First frame establishes the yawn-count baseline.
        self.engine.update(E(yawn_count=0, face_detected=True), timestamp_s=0.0)
        d = self.engine.update(E(yawn_count=1, face_detected=True), timestamp_s=10.0)
        self.assertEqual(d.breakdown["yawn_risk"], float(WEIGHT_YAWN_EVENT))

    def test_repeated_yawns_strengthen_recent_evidence(self) -> None:
        # Two yawns within the recent window -> stronger supporting evidence.
        self.engine.update(E(yawn_count=0, face_detected=True), timestamp_s=0.0)
        self.engine.update(E(yawn_count=1, face_detected=True), timestamp_s=10.0)
        d = self.engine.update(E(yawn_count=2, face_detected=True), timestamp_s=20.0)
        self.assertGreater(d.breakdown["yawn_risk"], float(WEIGHT_YAWN_EVENT))
        self.assertLessEqual(d.breakdown["yawn_risk"], float(RECENT_YAWN_MAX))

    def test_yawn_evidence_decays_over_time(self) -> None:
        self.engine.update(E(yawn_count=0, face_detected=True), timestamp_s=0.0)
        self.engine.update(E(yawn_count=1, face_detected=True), timestamp_s=10.0)
        d1 = self.engine.update(E(yawn_count=1, face_detected=True), timestamp_s=20.0)
        # After ~5 minutes the evidence should be negligible.
        d2 = self.engine.update(E(yawn_count=1, face_detected=True), timestamp_s=5 * 60 + 20.0)
        self.assertLess(d2.breakdown["yawn_risk"], d1.breakdown["yawn_risk"])
        self.assertLess(d2.breakdown["yawn_risk"], float(WEIGHT_YAWN_EVENT))

    def test_yawn_evidence_does_not_accumulate_forever(self) -> None:
        # Many yawns capped at RECENT_YAWN_MAX.
        self.engine.update(E(yawn_count=0, face_detected=True), timestamp_s=0.0)
        self.engine.update(E(yawn_count=10, face_detected=True), timestamp_s=1.0)
        d = self.engine.update(E(yawn_count=10, face_detected=True), timestamp_s=2.0)
        self.assertLessEqual(d.breakdown["yawn_risk"], float(RECENT_YAWN_MAX))

    def test_abnormal_blink_evidence_accumulates_and_decays(self) -> None:
        # High blink rate sampled every 10s builds decaying evidence.
        high = E(blink_rate=BLINK_HIGH_RATE_THRESHOLD + 10.0)
        normal_eyes = E(blink_rate=12.0)
        self.engine.update(high, timestamp_s=0.0)
        self.engine.update(high, timestamp_s=10.0)
        self.engine.update(high, timestamp_s=20.0)
        # Blink behaviour returns to normal: current-rate contribution is 0,
        # only the decaying recent evidence remains.
        self.engine.update(normal_eyes, timestamp_s=30.0)
        d1 = self.engine.update(normal_eyes, timestamp_s=40.0)
        self.assertGreater(d1.breakdown["blink_risk"], 0.0)
        # After ~5 minutes the evidence should be negligible.
        d2 = self.engine.update(normal_eyes, timestamp_s=5 * 60 + 40.0)
        self.assertLess(d2.breakdown["blink_risk"], d1.breakdown["blink_risk"])

    def test_low_blink_rate_is_abnormal_evidence(self) -> None:
        low = E(blink_rate=BLINK_LOW_RATE_THRESHOLD - 4.0)
        d = self.engine.update(low, timestamp_s=0.0)
        self.assertGreater(d.breakdown["blink_risk"], 0.0)

    def test_normal_blink_rate_does_not_accumulate(self) -> None:
        normal = E(blink_rate=12.0)
        d = self.engine.update(normal, timestamp_s=0.0)
        self.assertEqual(d.breakdown["blink_risk"], 0.0)

    def test_face_loss_preserves_recent_history(self) -> None:
        # Build yawn evidence, then lose the face, then resume.
        self.engine.update(E(yawn_count=0, face_detected=True), timestamp_s=0.0)
        self.engine.update(E(yawn_count=2, face_detected=True), timestamp_s=5.0)
        d_no_face = self.engine.update(E(face_detected=False), timestamp_s=7.0)
        self.assertIsNone(d_no_face.fatigue_score)
        # Resume: recent evidence should still be present (decayed slightly).
        d_resume = self.engine.update(E(yawn_count=2, face_detected=True), timestamp_s=20.0)
        self.assertGreater(d_resume.breakdown["yawn_risk"], 0.0)

    def test_prolonged_closure_remains_immediate_drowsy(self) -> None:
        # Prolonged eye closure must still produce the strong DROWSY response.
        d = self.engine.update(
            E(prolonged_eye_closure=True, current_eye_closure_seconds=2.0),
            timestamp_s=0.0,
        )
        self.assertEqual(d.status, "DROWSY")
        self.assertEqual(d.breakdown["eye_risk"], float(WEIGHT_PROLONGED_EYE_CLOSURE))


class IndependenceTests(unittest.TestCase):
    def test_module_has_no_cv_mediapipe_imports(self) -> None:
        import fatigue_engine as fe

        source = Path(fe.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import cv2", source)
        self.assertNotIn("import mediapipe", source)
        self.assertNotIn("from mediapipe", source)


if __name__ == "__main__":
    unittest.main()
