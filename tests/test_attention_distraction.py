"""
Focused unit tests for driver attention / distraction monitoring.

Uses a scripted head-pose tracker — no webcam, MediaPipe, or network.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attention_engine import AttentionEngine, fuse_raw_risk  # noqa: E402
from distraction_monitor import (  # noqa: E402
    CHANNEL_CLEAR_S,
    DistractionMonitor,
    HEAD_TURN_MIN_S,
    LOOK_DOWN_MIN_S,
    PASSENGER_MIN_S,
)
from head_pose import HeadPoseObservation, HeadPoseTracker  # noqa: E402
from head_pose import direction_from_angles, estimate_head_angles  # noqa: E402


class ScriptedHeadTracker:
    """Deterministic stand-in for HeadPoseTracker."""

    def __init__(self) -> None:
        self._obs = HeadPoseObservation(
            direction="forward",
            yaw_deg=0.0,
            pitch_deg=0.0,
            looking_away=False,
            confidence=1.0,
            duration_s=0.0,
            active=False,
        )

    def reset(self) -> None:
        self.set_forward()

    def set_forward(self) -> None:
        self._obs = HeadPoseObservation(
            direction="forward",
            yaw_deg=0.0,
            pitch_deg=0.0,
            looking_away=False,
            confidence=1.0,
            duration_s=0.0,
            active=False,
        )

    def set_away(self, direction: str, duration_s: float) -> None:
        self._obs = HeadPoseObservation(
            direction=direction,
            yaw_deg=40.0 if direction == "left" else (-40.0 if direction == "right" else 0.0),
            pitch_deg=30.0 if direction == "down" else 0.0,
            looking_away=True,
            confidence=1.0,
            duration_s=float(duration_s),
            active=True,
        )

    def update(self, landmarks, frame_w: int, frame_h: int, now_s: float) -> HeadPoseObservation:
        return self._obs


def _fake_distraction(
    risk: float,
    primary: str = "head_turn_left",
    direction: str = "left",
):
    """Build a DistractionObservation with an exact risk value."""
    from distraction_monitor import DistractionChannel, DistractionObservation

    head = HeadPoseObservation(
        direction=direction,
        yaw_deg=35.0,
        pitch_deg=0.0,
        looking_away=True,
        confidence=1.0,
        duration_s=5.0,
        active=True,
    )
    ch = DistractionChannel(primary, True, 5.0, float(risk))
    empty = DistractionChannel("none", False, 0.0, 0.0)
    return DistractionObservation(
        head=head,
        head_turn_left=ch if primary == "head_turn_left" else empty,
        head_turn_right=ch if primary == "head_turn_right" else empty,
        looking_down=ch if primary == "looking_down" else empty,
        passenger_side=empty,
        distraction_risk=float(risk),
        primary=primary,
        active=True,
        phone_available=False,
    )


class DistractionTemporalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = ScriptedHeadTracker()
        self.mon = DistractionMonitor(head_tracker=self.tracker)

    def test_brief_head_turn_not_severe(self) -> None:
        # Channel activates after 0.4s, but duration stays below HEAD_TURN_MIN_S.
        self.tracker.set_away("left", 0.0)
        self.mon.update(None, 640, 480, 0.0)
        self.tracker.set_away("left", 0.3)
        obs = self.mon.update(None, 640, 480, 0.5)
        self.assertLess(obs.distraction_risk, 28.0)
        self.assertNotEqual(obs.primary, "passenger_side")
        # Brief glance must not be treated as severe distraction.
        self.assertLess(obs.distraction_risk, 40.0)

    def test_sustained_head_turn_becomes_active(self) -> None:
        t = 0.0
        self.tracker.set_away("left", 0.0)
        self.mon.update(None, 640, 480, t)
        for step in range(1, 12):
            t = step * 0.5
            self.tracker.set_away("left", t)
            obs = self.mon.update(None, 640, 480, t)
        self.assertTrue(obs.head_turn_left.active)
        self.assertGreaterEqual(obs.distraction_risk, 28.0)
        self.assertIn(obs.primary, ("head_turn_left", "passenger_side"))

    def test_looking_down_off_road_persists(self) -> None:
        for t in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0):
            self.tracker.set_away("down", t)
            obs = self.mon.update(None, 640, 480, t)
        self.assertTrue(obs.looking_down.active)
        self.assertGreaterEqual(obs.looking_down.duration_s, LOOK_DOWN_MIN_S)
        self.assertGreaterEqual(obs.distraction_risk, 20.0)
        self.assertEqual(obs.primary, "looking_down")

    def test_passenger_side_attention_tracked(self) -> None:
        for t in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
            self.tracker.set_away("right", t)
            obs = self.mon.update(None, 640, 480, t)
        self.assertTrue(obs.passenger_side.active)
        self.assertGreaterEqual(obs.passenger_side.duration_s, PASSENGER_MIN_S)
        self.assertGreaterEqual(obs.passenger_side.score, 20.0)

    def test_returning_forward_clears_gradually(self) -> None:
        for t in (0.0, 1.0, 2.0, 3.0):
            self.tracker.set_away("left", t)
            self.mon.update(None, 640, 480, t)
        active = self.mon.update(None, 640, 480, 3.5)
        self.assertTrue(active.head_turn_left.active)

        self.tracker.set_forward()
        # Immediately after returning forward, channel still held (clear hysteresis).
        held = self.mon.update(None, 640, 480, 3.6)
        self.assertTrue(held.head_turn_left.active)

        # After clear hold, distraction channel clears.
        cleared = self.mon.update(None, 640, 480, 3.6 + CHANNEL_CLEAR_S + 0.05)
        self.assertFalse(cleared.head_turn_left.active)
        self.assertLess(cleared.distraction_risk, active.distraction_risk)


class LandmarkHeadPoseTests(unittest.TestCase):
    def _landmarks(self, nose_x: float):
        from types import SimpleNamespace

        points = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(264)]
        points[1] = SimpleNamespace(x=nose_x, y=0.50, z=0.0)
        points[152] = SimpleNamespace(x=0.50, y=0.82, z=0.0)
        points[33] = SimpleNamespace(x=0.38, y=0.40, z=0.0)
        points[263] = SimpleNamespace(x=0.62, y=0.40, z=0.0)
        return points

    def test_normalized_landmark_geometry_reaches_lateral_gate(self) -> None:
        yaw, pitch, confidence = estimate_head_angles(self._landmarks(0.58), 1000, 800)
        self.assertGreater(confidence, 0.0)
        self.assertEqual(direction_from_angles(yaw, pitch), "left")

    def test_small_landmark_motion_stays_forward(self) -> None:
        yaw, pitch, _confidence = estimate_head_angles(self._landmarks(0.53), 1000, 800)
        self.assertEqual(direction_from_angles(yaw, pitch), "forward")

    def _landmarks_with_pitch(self, pitch_deg: float):
        from types import SimpleNamespace

        points = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(264)]
        points[1] = SimpleNamespace(x=0.50, y=0.40 + (pitch_deg / 55.0) * 0.42, z=0.0)
        points[152] = SimpleNamespace(x=0.50, y=0.82, z=0.0)
        points[33] = SimpleNamespace(x=0.38, y=0.40, z=0.0)
        points[263] = SimpleNamespace(x=0.62, y=0.40, z=0.0)
        return points

    def test_camera_neutral_pitch_is_calibrated_as_forward(self) -> None:
        tracker = HeadPoseTracker(activate_hold_s=1.2, clear_hold_s=0.55)
        neutral = self._landmarks_with_pitch(24.5)
        for _ in range(10):
            tracker.calibrate(neutral, 1000, 800)

        observed = tracker.update(neutral, 1000, 800, 0.0)

        self.assertTrue(tracker.calibrated)
        self.assertAlmostEqual(observed.relative_pitch_deg, 0.0, places=3)
        self.assertEqual(observed.direction, "forward")
        self.assertFalse(observed.looking_away)

    def test_relative_pitch_requires_persistence_and_clears(self) -> None:
        tracker = HeadPoseTracker(activate_hold_s=1.2, clear_hold_s=0.55)
        neutral = self._landmarks_with_pitch(24.5)
        down = self._landmarks_with_pitch(34.5)
        for _ in range(10):
            tracker.calibrate(neutral, 1000, 800)

        first = tracker.update(down, 1000, 800, 0.0)
        held = tracker.update(down, 1000, 800, 1.0)
        active = tracker.update(down, 1000, 800, 1.3)
        self.assertAlmostEqual(active.relative_pitch_deg, 10.0, places=2)
        self.assertEqual(first.direction, "forward")
        self.assertEqual(held.direction, "forward")
        self.assertEqual(active.direction, "down")

        returned = tracker.update(neutral, 1000, 800, 1.4)
        cleared = tracker.update(neutral, 1000, 800, 2.0)
        self.assertEqual(returned.direction, "down")
        self.assertEqual(cleared.direction, "forward")
        self.assertFalse(cleared.looking_away)

    def test_neutral_camera_pitch_produces_no_distraction(self) -> None:
        tracker = HeadPoseTracker()
        monitor = DistractionMonitor(head_tracker=tracker)
        neutral = self._landmarks_with_pitch(24.5)
        for _ in range(10):
            tracker.calibrate(neutral, 1000, 800)

        observation = monitor.update(neutral, 1000, 800, 0.0)

        self.assertEqual(observation.head.direction, "forward")
        self.assertFalse(observation.looking_down.active)
        self.assertEqual(observation.distraction_risk, 0.0)


class AttentionRiskContributionTests(unittest.TestCase):
    def test_fatigue_contributes_to_attention_risk(self) -> None:
        eng = AttentionEngine()
        low = eng.update(fatigue_risk=5.0, distraction=None, now_s=0.0)
        high = AttentionEngine().update(fatigue_risk=70.0, distraction=None, now_s=0.0)
        self.assertGreater(high.combined_risk, low.combined_risk)
        self.assertGreaterEqual(high.combined_risk, 70.0 * 0.9)

    def test_risk_bounded_0_to_100(self) -> None:
        eng = AttentionEngine()
        d = _fake_distraction(95.0)
        out = eng.update(
            fatigue_risk=100.0,
            distraction=d,
            now_s=0.0,
            location_risk=100.0,
            weather_risk=100.0,
            location_zone="school",
            weather_condition="heavy_rain",
        )
        self.assertGreaterEqual(out.combined_risk, 0.0)
        self.assertLessEqual(out.combined_risk, 100.0)
        self.assertGreaterEqual(fuse_raw_risk(0, 0), 0.0)
        self.assertLessEqual(fuse_raw_risk(200, 200), 100.0)


if __name__ == "__main__":
    unittest.main()
