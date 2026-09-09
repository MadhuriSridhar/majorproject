"""
Focused unit tests for context-aware attention risk fusion.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attention_engine import (  # noqa: E402
    AttentionEngine,
    apply_context_boost,
    fuse_raw_risk,
    risk_level_from_score,
)
from distraction_monitor import DistractionChannel, DistractionObservation  # noqa: E402
from head_pose import HeadPoseObservation  # noqa: E402


def _dist(risk: float) -> DistractionObservation:
    head = HeadPoseObservation(
        direction="left",
        yaw_deg=35.0,
        pitch_deg=0.0,
        looking_away=True,
        confidence=1.0,
        duration_s=4.0,
        active=True,
    )
    ch = DistractionChannel("head_turn_left", True, 4.0, risk)
    empty = DistractionChannel("none", False, 0.0, 0.0)
    return DistractionObservation(
        head=head,
        head_turn_left=ch,
        head_turn_right=empty,
        looking_down=empty,
        passenger_side=empty,
        distraction_risk=float(risk),
        primary="head_turn_left",
        active=True,
        phone_available=False,
    )


class ContextAwareRiskTests(unittest.TestCase):
    def test_school_zone_raises_above_normal_location(self) -> None:
        d = 45.0
        base = fuse_raw_risk(10.0, d)
        normal = apply_context_boost(base, d, location_risk=5.0, weather_risk=5.0, zone_type="none")
        school = apply_context_boost(base, d, location_risk=70.0, weather_risk=5.0, zone_type="school")
        self.assertLess(normal, school)
        self.assertLessEqual(school, 100.0)

    def test_heavy_rain_raises_above_clear(self) -> None:
        d = 45.0
        base = fuse_raw_risk(10.0, d)
        clear = apply_context_boost(base, d, location_risk=5.0, weather_risk=5.0, zone_type="none")
        rain = apply_context_boost(base, d, location_risk=5.0, weather_risk=80.0, zone_type="none")
        self.assertLess(clear, rain)
        self.assertLessEqual(rain, 100.0)

    def test_combined_high_risk_context_can_escalate(self) -> None:
        eng = AttentionEngine()
        out = eng.update(
            fatigue_risk=10.0,
            distraction=_dist(45.0),
            now_s=0.0,
            location_risk=70.0,
            weather_risk=80.0,
            location_zone="school",
            weather_condition="heavy_rain",
        )
        self.assertIn(out.level, ("HIGH", "CRITICAL"))
        self.assertGreaterEqual(out.combined_risk, 70.0)
        self.assertLessEqual(out.combined_risk, 100.0)
        if out.level == "CRITICAL":
            self.assertTrue(out.safe_stop)

    def test_final_score_remains_0_to_100(self) -> None:
        eng = AttentionEngine()
        out = eng.update(
            fatigue_risk=100.0,
            distraction=_dist(100.0),
            now_s=0.0,
            location_risk=100.0,
            weather_risk=100.0,
            location_zone="school",
            weather_condition="storm",
        )
        self.assertGreaterEqual(out.combined_risk, 0.0)
        self.assertLessEqual(out.combined_risk, 100.0)
        self.assertEqual(risk_level_from_score(out.combined_risk), out.level)


if __name__ == "__main__":
    unittest.main()
