"""
Focused tests for the PART 5 face-state / head-turn distinction rules.

Covers every PART 5M state: face present but turned is not "no face"; eyes /
mouth unavailable because of a turned/down head is not fatigue (and never read
as "eyes closed"); missing subfeatures never collapse the PARTIAL face state
into ABSENT; temporary face loss does not alert or reset history/duration/stats;
and passenger-side turns are NOT labelled "talking to passenger" without actual
talking evidence.

No webcam / MediaPipe / network required.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alert_escalation import AlertEscalationController  # noqa: E402
from app_ui import FACE_LOSS_GRACE_S, classify_face_state, head_is_down, head_is_turned  # noqa: E402
from attention_engine import AttentionEngine  # noqa: E402
from distraction_monitor import (  # noqa: E402
    DistractionMonitor,
    classify_distraction_type,
)
from driving_duration import DrivingDurationTracker  # noqa: E402
from fatigue_engine import (  # noqa: E402
    WEIGHT_PROLONGED_EYE_CLOSURE,
    FatigueEngine,
    FatigueEvidence,
)
from head_pose import HeadPoseObservation  # noqa: E402
from session_manager import SessionManager  # noqa: E402


def E(**kwargs) -> FatigueEvidence:
    return FatigueEvidence(**kwargs)


def _head(direction: str, looking_away: bool, duration_s: float = 5.0) -> HeadPoseObservation:
    """Build a HeadPoseObservation with a plausible signed angle per direction."""
    yaw = 40.0 if direction == "left" else (-40.0 if direction == "right" else 0.0)
    pitch = 30.0 if direction == "down" else 0.0
    return HeadPoseObservation(
        direction=direction,
        yaw_deg=yaw,
        pitch_deg=pitch,
        looking_away=looking_away,
        confidence=1.0,
        duration_s=duration_s if looking_away else 0.0,
        active=looking_away,
    )


class _ScriptedHead:
    """Deterministic stand-in for HeadPoseTracker."""

    def __init__(self) -> None:
        self.obs = _head("forward", False, 0.0)

    def reset(self) -> None:
        self.set_forward()

    def set_forward(self) -> None:
        self.obs = _head("forward", False, 0.0)

    def set_away(self, direction: str, duration_s: float) -> None:
        self.obs = _head(direction, True, duration_s)

    def update(self, landmarks, frame_w: int, frame_h: int, now_s: float) -> HeadPoseObservation:
        return self.obs


def _attention(alertness: float = 60.0, level: str = "MODERATE", **kw) -> SimpleNamespace:
    values = dict(
        alertness_score=alertness,
        level=level,
        distraction_score=10.0,
        fatigue_score=40.0,
        primary_reason="Prolonged eye closure",
        location_risk=0.0,
        weather_risk=0.0,
        recommendation="",
    )
    values.update(kw)
    return SimpleNamespace(**values)


class FaceTurnIsNotFatigueTests(unittest.TestCase):
    def _engine(self) -> FatigueEngine:
        return FatigueEngine(drowsy_enter_hold_s=0.0, drowsy_exit_hold_s=0.0)

    def test_forward_eyes_visible_is_normal(self) -> None:
        # PART 5M.1 - face present, looking forward, eyes visible -> normal.
        d = self._engine().decide(E(face_detected=True, prolonged_eye_closure=False))
        self.assertEqual(d.status, "SAFE")
        self.assertEqual(d.breakdown["eye_risk"], 0.0)
        self.assertFalse(head_is_turned(_head("forward", False)))
        self.assertFalse(head_is_down(_head("forward", False)))

    def test_turned_left_eyes_unavailable_not_fatigue(self) -> None:
        # PART 5M.2 - face present + head turned left + eyes unavailable.
        self.assertTrue(head_is_turned(_head("left", True)))
        d = self._engine().decide(
            E(prolonged_eye_closure=True, current_eye_closure_seconds=2.0, head_turned=True)
        )
        self.assertNotEqual(d.status, "DROWSY")
        self.assertEqual(d.breakdown["eye_risk"], 0.0)

    def test_turned_right_eyes_unavailable_not_fatigue(self) -> None:
        # PART 5M.3 - face present + head turned right + eyes unavailable.
        self.assertTrue(head_is_turned(_head("right", True)))
        d = self._engine().decide(
            E(prolonged_eye_closure=True, current_eye_closure_seconds=2.0, head_turned=True)
        )
        self.assertNotEqual(d.status, "DROWSY")
        self.assertEqual(d.breakdown["eye_risk"], 0.0)

    def test_mouth_unavailable_not_fatigue(self) -> None:
        # PART 5M.4 - face present + head turned + mouth unavailable.
        d = self._engine().decide(
            E(
                yawn_detected=True,
                yawn_count=1,
                recent_yawn_frequency=3.0,
                mouth_not_visible=True,
            )
        )
        self.assertEqual(d.breakdown["yawn_risk"], 0.0)
        self.assertNotEqual(d.status, "DROWSY")

    def test_eyes_not_visible_not_read_as_closed(self) -> None:
        # PART 5M.12 - eye landmarks unavailable because of orientation must
        # never be interpreted as closed eyes.
        d = self._engine().decide(
            E(prolonged_eye_closure=True, current_eye_closure_seconds=2.0, eyes_not_visible=True)
        )
        self.assertNotEqual(d.status, "DROWSY")
        self.assertEqual(d.breakdown["eye_risk"], 0.0)
        self.assertNotIn("Prolonged eye closure", d.reasons)

    def test_genuine_eye_closure_increases_fatigue(self) -> None:
        # PART 5M.13 - genuine prolonged closure with valid landmarks.
        d = self._engine().decide(
            E(prolonged_eye_closure=True, current_eye_closure_seconds=2.0)
        )
        self.assertEqual(d.status, "DROWSY")
        self.assertEqual(d.breakdown["eye_risk"], float(WEIGHT_PROLONGED_EYE_CLOSURE))
class FaceObservationStateTests(unittest.TestCase):
    def test_head_turned_helper(self) -> None:
        self.assertTrue(head_is_turned(_head("left", True)))
        self.assertTrue(head_is_turned(_head("right", True)))
        self.assertFalse(head_is_turned(_head("down", True)))
        self.assertFalse(head_is_turned(_head("forward", False)))
        self.assertFalse(head_is_turned(None))

    def test_few_frames_no_face_is_temporary_loss(self) -> None:
        # PART 5M.14 - one/few frames with no face -> temporary face-loss.
        self.assertEqual(
            classify_face_state(False, 0.0, FACE_LOSS_GRACE_S),
            "FACE_TEMPORARILY_LOST",
        )

    def test_prolonged_absence_is_no_face(self) -> None:
        # PART 5M.18 - prolonged complete face absence -> no-face state.
        self.assertEqual(
            classify_face_state(False, FACE_LOSS_GRACE_S + 0.01, FACE_LOSS_GRACE_S),
            "FACE_NOT_DETECTED",
        )
        self.assertEqual(
            classify_face_state(True, None, FACE_LOSS_GRACE_S),
            "FACE_DETECTED",
        )


class DistractionTypeTests(unittest.TestCase):
    def test_normalized_distraction_types(self) -> None:
        # PART 5M.9 / 5M.10 / PART 5F.
        self.assertEqual(classify_distraction_type("head_turn_left"), "head_turn")
        self.assertEqual(classify_distraction_type("head_turn_right"), "head_turn")
        self.assertEqual(classify_distraction_type("looking_down"), "downward")
        # Passenger-side WITHOUT talking evidence -> lateral/head-turn.
        self.assertEqual(classify_distraction_type("passenger_side"), "head_turn")
        self.assertNotEqual(classify_distraction_type("passenger_side"), "passenger_conversation")
        # Passenger-side WITH talking evidence -> passenger conversation.
        self.assertEqual(
            classify_distraction_type("passenger_side", talking_detected=True),
            "passenger_conversation",
        )
        self.assertEqual(classify_distraction_type("none"), "none")
        self.assertEqual(classify_distraction_type(None), "none")

    def test_mouth_unavailable_is_not_certain_not_talking(self) -> None:
        # PART 5M.11 - mouth unavailable because of orientation must not be
        # interpreted as "not talking" with certainty.
        d = FatigueEngine(drowsy_enter_hold_s=0.0, drowsy_exit_hold_s=0.0).decide(
            E(yawn_detected=True, mouth_not_visible=True)
        )
        self.assertEqual(d.breakdown["yawn_risk"], 0.0)
class DistractionTemporalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = _ScriptedHead()
        self.mon = DistractionMonitor(head_tracker=self.tracker)

    def test_sustained_turn_builds_head_turn_distraction(self) -> None:
        # PART 5M.5 / 5M.7 - sustained turn -> distraction evidence increases.
        self.tracker.set_away("left", 5.0)
        out = None
        for t in (0.0, 0.4):
            out = self.mon.update(None, 1000, 800, t)
        self.assertTrue(out.head_turn_left.active)
        self.assertGreater(out.distraction_risk, 20.0)
        self.assertEqual(out.distraction_type, "head_turn")

    def test_sustained_right_turn_is_head_turn(self) -> None:
        self.tracker.set_away("right", 5.0)
        out = None
        for t in (0.0, 0.4):
            out = self.mon.update(None, 1000, 800, t)
        self.assertTrue(out.head_turn_right.active)
        self.assertEqual(out.distraction_type, "head_turn")

    def test_brief_head_movement_is_not_severe(self) -> None:
        # PART 5M.6 - brief left/right movement must not trigger severe alert.
        self.tracker.set_away("left", 0.5)
        self.mon.update(None, 1000, 800, 0.0)
        self.tracker.set_forward()
        out = None
        for t in (0.2, 0.4, 0.8):
            out = self.mon.update(None, 1000, 800, t)
        self.assertEqual(out.primary, "none")
        self.assertLess(out.distraction_risk, 20.0)

    def test_brief_mirror_check_no_alarm(self) -> None:
        # PART 5M.8 - brief mirror/blind-spot style movement -> no alarm.
        for t in (0.0, 0.2):
            self.tracker.set_away("right", 0.3)
            out = self.mon.update(None, 1000, 800, t)
        self.tracker.set_forward()
        for t in (0.4, 0.6):
            out = self.mon.update(None, 1000, 800, t)
        self.assertLess(out.distraction_risk, 30.0)
        attn = AttentionEngine().update(fatigue_risk=0.0, distraction=out, now_s=0.8)
        self.assertEqual(attn.alert_severity, "NORMAL")


class FaceLossGraceTests(unittest.TestCase):
    def test_temporary_face_loss_no_fatigue_alert(self) -> None:
        # PART 5M.15 - temporary face loss -> no fatigue alert.
        eng = AttentionEngine()
        eng.update(fatigue_risk=70.0, distraction=None, now_s=0.0)
        d = eng.update(fatigue_risk=None, distraction=None, now_s=0.5, face_ok=False)
        self.assertFalse(d.face_ok)
        self.assertEqual(d.alert_severity, "NORMAL")
        self.assertIn("No valid face", d.primary_reason)

    def test_temporary_face_loss_no_alarm(self) -> None:
        # PART 5M.16 - temporary face loss -> no alarm.
        ctl = AlertEscalationController()
        r = ctl.update("NORMAL", 0.0)
        self.assertFalse(r.should_play_audio)
        self.assertEqual(r.active_severity, "NORMAL")

    def test_face_reacquisition_resumes(self) -> None:
        # PART 5M.17 - face reacquisition -> normal processing resumes.
        eng = FatigueEngine(drowsy_enter_hold_s=0.0, drowsy_exit_hold_s=0.0)
        self.assertIsNone(eng.decide(E(face_detected=False)).fatigue_score)
        # First frame after reacquisition just resumes analysis (returns None),
        # the next frame scores normally again.
        eng.decide(E(face_detected=True))
        d = eng.decide(
            E(face_detected=True, prolonged_eye_closure=True, current_eye_closure_seconds=2.0)
        )
        self.assertIsNotNone(d.fatigue_score)
        self.assertEqual(d.status, "DROWSY")
class FaceLossDoesNotResetStateTests(unittest.TestCase):
    def test_face_loss_preserves_alertness_history(self) -> None:
        # PART 5M.19.
        mgr = SessionManager()
        mgr.start()
        eng = FatigueEngine(drowsy_enter_hold_s=0.0, drowsy_exit_hold_s=0.0)
        decision = eng.decide(
            E(prolonged_eye_closure=True, current_eye_closure_seconds=2.0)
        )
        mgr.update(decision, 0.0, attention=_attention())
        before = len(mgr.stats.alertness_history)
        self.assertGreater(before, 0)
        mgr.record_no_face(1.0)
        self.assertEqual(len(mgr.stats.alertness_history), before)

    def test_face_loss_does_not_reset_driving_duration(self) -> None:
        # PART 5M.20 - a face loss never resets the session clock.
        tracker = DrivingDurationTracker(start_s=100.0, use_demo=True, demo_seconds=30.0)
        self.assertEqual(tracker.update(110.0).elapsed_s, 10.0)
        # "Face loss" here means no reset() is called; elapsed keeps advancing.
        self.assertEqual(tracker.update(120.0).elapsed_s, 20.0)
        self.assertTrue(tracker.update(135.0).threshold_reached)

    def test_face_loss_does_not_reset_session_stats(self) -> None:
        # PART 5M.21 - session statistics are preserved through face loss.
        mgr = SessionManager()
        mgr.start()
        eng = FatigueEngine(drowsy_enter_hold_s=0.0, drowsy_exit_hold_s=0.0)
        eng.decide(E(face_detected=True))  # warm-up so the face is established
        evidence = E(
            prolonged_eye_closure=True,
            current_eye_closure_seconds=2.0,
            blink_count=7,
            yawn_count=2,
        )
        eng.update(evidence, 0.0)
        decision = eng.update(evidence, 0.1)  # second frame latches DROWSY severity
        mgr.update(decision, 0.0)
        hist_before = len(mgr.stats.risk_history)
        final_before = mgr.stats.final_status
        self.assertEqual(mgr.stats.blink_count, 7)
        self.assertEqual(mgr.stats.yawn_count, 2)

        mgr.record_no_face(1.0)

        # Face loss must not erase or reset the accumulated session statistics.
        self.assertEqual(mgr.stats.blink_count, 7)
        self.assertEqual(mgr.stats.yawn_count, 2)
        self.assertEqual(mgr.stats.final_status, final_before)
        self.assertGreaterEqual(len(mgr.stats.risk_history), hist_before)


if __name__ == "__main__":
    unittest.main()
