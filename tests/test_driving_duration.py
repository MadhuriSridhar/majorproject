"""Unit tests for Stage 6B driving-duration tracking."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from driving_duration import (  # noqa: E402
    DEMO_DRIVING_DURATION_THRESHOLD_SECONDS,
    DRIVING_DURATION_THRESHOLD_HOURS,
    DrivingDurationTracker,
    active_threshold_seconds,
    format_hms,
)


class FormatTests(unittest.TestCase):
    def test_format_hms(self) -> None:
        self.assertEqual(format_hms(0), "00:00:00")
        self.assertEqual(format_hms(65), "00:01:05")
        self.assertEqual(format_hms(3661), "01:01:01")


class ThresholdSelectionTests(unittest.TestCase):
    def test_demo_threshold(self) -> None:
        self.assertEqual(
            active_threshold_seconds(use_demo=True, demo_seconds=30, real_hours=3),
            30.0,
        )

    def test_real_threshold(self) -> None:
        self.assertEqual(
            active_threshold_seconds(use_demo=False, demo_seconds=30, real_hours=3),
            3.0 * 3600.0,
        )


class DrivingDurationTrackerTests(unittest.TestCase):
    def test_below_threshold_is_normal(self) -> None:
        tracker = DrivingDurationTracker(start_s=1000.0, use_demo=True, demo_seconds=30.0)
        state = tracker.update(1020.0)  # 20 s
        self.assertEqual(state.duration_risk, "NORMAL")
        self.assertFalse(state.threshold_reached)
        self.assertEqual(state.driving_time_text, "00:00:20")
        self.assertTrue(state.demo_mode)

    def test_at_threshold_is_elevated(self) -> None:
        tracker = DrivingDurationTracker(start_s=1000.0, use_demo=True, demo_seconds=30.0)
        state = tracker.update(1030.0)
        self.assertEqual(state.duration_risk, "ELEVATED")
        self.assertTrue(state.threshold_reached)

    def test_above_threshold_stays_elevated(self) -> None:
        tracker = DrivingDurationTracker(start_s=1000.0, use_demo=True, demo_seconds=30.0)
        state = tracker.update(1100.0)
        self.assertEqual(state.duration_risk, "ELEVATED")

    def test_timer_does_not_reset_across_updates(self) -> None:
        tracker = DrivingDurationTracker(start_s=0.0, use_demo=True, demo_seconds=30.0)
        a = tracker.update(5.0)
        b = tracker.update(12.0)
        self.assertEqual(a.driving_time_text, "00:00:05")
        self.assertEqual(b.driving_time_text, "00:00:12")
        self.assertEqual(tracker.start_s, 0.0)

    def test_real_three_hour_threshold(self) -> None:
        tracker = DrivingDurationTracker(
            start_s=0.0,
            use_demo=False,
            real_hours=DRIVING_DURATION_THRESHOLD_HOURS,
        )
        below = tracker.update(2.9 * 3600.0)
        self.assertEqual(below.duration_risk, "NORMAL")
        above = tracker.update(3.0 * 3600.0)
        self.assertEqual(above.duration_risk, "ELEVATED")
        self.assertFalse(above.demo_mode)

    def test_demo_constant_default(self) -> None:
        self.assertEqual(DEMO_DRIVING_DURATION_THRESHOLD_SECONDS, 30.0)


if __name__ == "__main__":
    unittest.main()