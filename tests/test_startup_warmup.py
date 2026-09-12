"""Regression tests for startup evidence isolation and warm-up."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alert_escalation import AlertEscalationController  # noqa: E402
from attention_engine import AttentionEngine  # noqa: E402
from app_ui import CameraWorker  # noqa: E402
from fatigue_engine import FatigueEngine, FatigueEvidence  # noqa: E402
from session_manager import SessionManager  # noqa: E402
from temporal_eye import EyeClosureTracker  # noqa: E402
from temporal_mouth import YawnTracker  # noqa: E402


class StartupWarmupTests(unittest.TestCase):
    def _worker(self) -> CameraWorker:
        return CameraWorker(
            session_manager=SessionManager(),
            alert_controller=AlertEscalationController(),
            audio_engine=Mock(),
        )

    def test_valid_face_evidence_needs_time_and_frames(self) -> None:
        worker = self._worker()

        self.assertFalse(worker._evidence_ready(0.0, True))
        self.assertFalse(worker._evidence_ready(0.1, True))
        self.assertFalse(worker._evidence_ready(0.2, True))
        self.assertFalse(worker._evidence_ready(0.3, True))
        self.assertFalse(worker._evidence_ready(0.4, True))
        self.assertFalse(worker._evidence_ready(1.0, True))
        self.assertTrue(worker._evidence_ready(1.5, True))

    def test_invalid_face_restarts_warmup(self) -> None:
        worker = self._worker()
        for timestamp in (0.0, 0.4, 0.8, 1.2, 1.6):
            worker._evidence_ready(timestamp, True)
        self.assertFalse(worker._evidence_ready(1.7, False))
        self.assertFalse(worker._evidence_ready(1.8, True))

    def test_session_reset_clears_warmup_and_context(self) -> None:
        worker = self._worker()
        worker._evidence_ready(0.0, True)
        worker._location_ctx = object()
        worker._weather_ctx = object()

        worker.reset_session_state()

        self.assertEqual(worker._valid_evidence_frames, 0)
        self.assertIsNone(worker._valid_evidence_since)
        self.assertIsNone(worker._location_ctx)
        self.assertIsNone(worker._weather_ctx)

    def test_new_session_clears_decaying_fatigue_history(self) -> None:
        worker = self._worker()
        worker.fatigue_engine.update(
            FatigueEvidence(yawn_detected=False, yawn_count=0), 0.0
        )
        worker.fatigue_engine.update(
            FatigueEvidence(yawn_detected=True, yawn_count=1), 1.0
        )
        self.assertGreater(worker.fatigue_engine._recent_yawn_evidence, 0.0)

        worker.reset_session_state()

        self.assertEqual(worker.fatigue_engine._recent_yawn_evidence, 0.0)
        self.assertEqual(worker.fatigue_engine._recent_blink_evidence, 0.0)

    def test_normal_driver_pipeline_stays_safe_after_warmup(self) -> None:
        eyes = EyeClosureTracker()
        mouth = YawnTracker()
        fatigue = FatigueEngine()
        attention = AttentionEngine()
        decision = None
        result = None
        for timestamp in (0.0, 0.5, 1.0, 1.5, 2.0):
            eye = eyes.update(0.30, timestamp)
            mouth_state = mouth.update(0.10, timestamp)
            decision = fatigue.update(
                FatigueEvidence(
                    prolonged_eye_closure=eye.prolonged_closure,
                    current_eye_closure_seconds=eye.current_closure_s,
                    blink_count=eye.blink_count,
                    blink_rate=eye.blink_rate,
                    recent_blink_rate=eye.recent_blink_rate,
                    yawn_detected=mouth_state.yawn,
                    yawn_count=mouth_state.yawn_count,
                ),
                timestamp,
            )
            result = attention.update(decision.smoothed_risk or 0.0, None, timestamp)

        self.assertEqual(eye.eye, "OPEN")
        self.assertEqual(decision.fatigue_score, 0)
        self.assertEqual(result.distraction_score, 0.0)
        self.assertEqual(result.alert_severity, "NORMAL")
        self.assertGreaterEqual(result.alertness_score, 95.0)


if __name__ == "__main__":
    unittest.main()
