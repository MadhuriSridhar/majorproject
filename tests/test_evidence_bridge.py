"""Integration-mapping tests for Stage 7B (no webcam required)."""

from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from driving_duration import DrivingDurationResult  # noqa: E402
from evidence_bridge import build_fatigue_evidence  # noqa: E402
from fatigue_engine import FatigueEngine  # noqa: E402
from temporal_eye import TemporalEyeState  # noqa: E402
from temporal_mouth import TemporalMouthState  # noqa: E402
from time_context import TimeContextResult  # noqa: E402


def _time_ctx(night: bool = False) -> TimeContextResult:
    return TimeContextResult(
        local_time=datetime(2026, 8, 11, 23 if night else 14, 0, 0),
        time_text="23:00:00" if night else "14:00:00",
        context="NIGHT" if night else "DAY",
        night_context_active=night,
        using_forced_time=False,
    )


def _driving(elevated: bool = False) -> DrivingDurationResult:
    return DrivingDurationResult(
        elapsed_s=40.0 if elevated else 5.0,
        driving_time_text="00:00:40" if elevated else "00:00:05",
        threshold_s=30.0,
        duration_risk="ELEVATED" if elevated else "NORMAL",
        threshold_reached=elevated,
        demo_mode=True,
    )


class EvidenceBridgeTests(unittest.TestCase):
    def test_maps_prolonged_and_yawn_flags(self) -> None:
        eye = TemporalEyeState(
            ear=0.10,
            eye="CLOSED",
            blink_count=3,
            current_closure_s=2.2,
            prolonged_closure=True,
        )
        mouth = TemporalMouthState(
            mar=0.80,
            mouth="OPEN",
            current_mouth_open_s=1.5,
            yawn_count=2,
            yawn=True,
        )
        evidence = build_fatigue_evidence(eye, mouth, _time_ctx(True), _driving(True))
        self.assertTrue(evidence.prolonged_eye_closure)
        self.assertEqual(evidence.current_eye_closure_seconds, 2.2)
        self.assertEqual(evidence.blink_count, 3)
        self.assertTrue(evidence.yawn_detected)
        self.assertEqual(evidence.yawn_count, 2)
        self.assertTrue(evidence.night_context)
        self.assertTrue(evidence.driving_duration_risk)

    def test_no_face_like_unknown_states_are_safe_inputs(self) -> None:
        eye = TemporalEyeState(
            ear=None,
            eye="UNKNOWN",
            blink_count=0,
            current_closure_s=0.0,
            prolonged_closure=False,
        )
        mouth = TemporalMouthState(
            mar=None,
            mouth="UNKNOWN",
            current_mouth_open_s=0.0,
            yawn_count=0,
            yawn=False,
        )
        evidence = build_fatigue_evidence(eye, mouth, _time_ctx(False), _driving(False))
        eng = FatigueEngine(drowsy_enter_hold_s=0.0, drowsy_exit_hold_s=0.0)
        decision = eng.decide(evidence)
        self.assertEqual(decision.status, "SAFE")
        self.assertFalse(evidence.prolonged_eye_closure)
        self.assertFalse(evidence.yawn_detected)

    def test_context_alone_not_drowsy_through_bridge(self) -> None:
        eye = TemporalEyeState(0.30, "OPEN", 1, 0.0, False)
        mouth = TemporalMouthState(0.20, "CLOSED", 0.0, 0, False)
        evidence = build_fatigue_evidence(eye, mouth, _time_ctx(True), _driving(True))
        eng = FatigueEngine(drowsy_enter_hold_s=0.0, drowsy_exit_hold_s=0.0)
        decision = eng.decide(evidence)
        self.assertNotEqual(decision.status, "DROWSY")


if __name__ == "__main__":
    unittest.main()