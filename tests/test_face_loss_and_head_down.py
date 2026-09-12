"""
Focused tests for:
  * head-down must NOT become fatigue
  * face-loss grace / temporary-unavailable handling

No webcam / MediaPipe / network.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alert_escalation import AlertEscalationController  # noqa: E402
from app_ui import (  # noqa: E402
    FACE_LOSS_GRACE_S,
    CameraWorker,
    classify_face_state,
    head_is_down,
)
from distraction_monitor import DistractionMonitor  # noqa: E402
from fatigue_engine import (  # noqa: E402
    WEIGHT_PROLONGED_EYE_CLOSURE,
    FatigueEngine,
    FatigueEvidence,
)
from head_pose import HeadPoseObservation  # noqa: E402
from session_manager import SessionManager  # noqa: E402


def _head(direction: str, looking_away: bool) -> HeadPoseObservation:
    return HeadPoseObservation(
        direction=direction,
        yaw_deg=0.0,
        pitch_deg=30.0 if direction == "down" else 0.0,
        looking_away=looking_away,
        confidence=1.0,
        duration_s=5.0 if looking_away else 0.0,
        active=looking_away,
    )


class _ScriptedHead:
    def __init__(self) -> None:
        self.obs = _head("forward", False)

    def update(self, landmarks, frame_w, frame_h, now_s):
        return self.obs


class HeadDownIsNotFatigueTests(unittest.TestCase):
    def test_looking_down_suppresses_eye_closure_fatigue(self) -> None:
        eng = FatigueEngine(drowsy_enter_hold_s=0.0, drowsy_exit_hold_s=0.0)
        d = eng.decide(
            FatigueEvidence(
                prolonged_eye_closure=True,
                current_eye_closure_seconds=2.0,
                head_down=True,
            )
        )
        self.assertNotEqual(d.status, "DROWSY")
        self.assertEqual(d.breakdown["eye_risk"], 0.0)
        self.assertEqual(d.fatigue_score, 0)

    def test_prolonged_closure_without_head_down_still_drowsy(self) -> None:
        eng = FatigueEngine(drowsy_enter_hold_s=0.0, drowsy_exit_hold_s=0.0)
        d = eng.decide(
            FatigueEvidence(prolonged_eye_closure=True, current_eye_closure_seconds=2.0)
        )
        self.assertEqual(d.status, "DROWSY")
        self.assertEqual(d.breakdown["eye_risk"], float(WEIGHT_PROLONGED_EYE_CLOSURE))

    def test_head_down_alone_never_creates_fatigue(self) -> None:
        eng = FatigueEngine()
        d = eng.decide(FatigueEvidence(head_down=True))
        self.assertEqual(d.status, "SAFE")
        self.assertEqual(d.fatigue_score, 0)


class HeadDownIsDistractionTests(unittest.TestCase):
    def test_sustained_down_builds_distraction_not_fatigue(self) -> None:
        tracker = _ScriptedHead()
        mon = DistractionMonitor(head_tracker=tracker)
        tracker.obs = _head("down", True)
        out = None
        for t in (0.0, 0.2, 0.4, 0.6):
            out = mon.update(None, 1000, 800, t)
        self.assertTrue(out.looking_down.active)
        self.assertGreater(out.distraction_risk, 20.0)
        self.assertEqual(out.primary, "looking_down")

    def test_brief_down_does_not_activate_distraction(self) -> None:
        tracker = _ScriptedHead()
        mon = DistractionMonitor(head_tracker=tracker)
        tracker.obs = _head("down", True)
        mon.update(None, 1000, 800, 0.0)
        tracker.obs = _head("forward", False)
        out = None
        for t in (0.2, 0.4, 0.8):
            out = mon.update(None, 1000, 800, t)
        self.assertFalse(out.looking_down.active)
        self.assertEqual(out.primary, "none")


class FaceStateHelpersTests(unittest.TestCase):
    def test_head_is_down_only_when_active_and_down(self) -> None:
        self.assertTrue(head_is_down(_head("down", True)))
        self.assertFalse(head_is_down(_head("left", True)))
        self.assertFalse(head_is_down(_head("forward", False)))
        self.assertFalse(head_is_down(_head("down", False)))
        self.assertFalse(head_is_down(None))

    def test_classify_face_state_transitions(self) -> None:
        g = FACE_LOSS_GRACE_S
        self.assertEqual(classify_face_state(True, None, g), "FACE_DETECTED")
        self.assertEqual(classify_face_state(False, 0.0, g), "FACE_TEMPORARILY_LOST")
        self.assertEqual(classify_face_state(False, g - 0.01, g), "FACE_TEMPORARILY_LOST")
        self.assertEqual(classify_face_state(False, g + 0.01, g), "FACE_NOT_DETECTED")


class CameraWorkerFaceLossStateTests(unittest.TestCase):
    def _worker(self) -> CameraWorker:
        return CameraWorker(
            session_manager=SessionManager(),
            alert_controller=AlertEscalationController(),
            audio_engine=Mock(),
        )

    def test_default_grace_period_and_reset(self) -> None:
        w = self._worker()
        self.assertAlmostEqual(w.face_loss_grace_s, FACE_LOSS_GRACE_S)
        w._face_lost_since = 1.0
        w._last_distraction_obs = object()
        w.reset_session_state()
        self.assertIsNone(w._face_lost_since)
        self.assertIsNone(w._last_distraction_obs)


if __name__ == "__main__":
    unittest.main()
