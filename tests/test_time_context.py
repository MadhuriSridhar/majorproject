"""Unit tests for Stage 6A day/night time context."""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from time_context import get_time_context, is_night  # noqa: E402


class NightDayClassificationTests(unittest.TestCase):
    def test_day_afternoon(self) -> None:
        moment = datetime(2026, 8, 10, 14, 30, 10)
        self.assertFalse(is_night(moment))
        result = get_time_context(now=moment, force_local_time=None)
        self.assertEqual(result.context, "DAY")
        self.assertFalse(result.night_context_active)
        self.assertEqual(result.time_text, "14:30:10")

    def test_day_just_after_night_end(self) -> None:
        moment = datetime(2026, 8, 10, 6, 0, 0)
        self.assertFalse(is_night(moment))
        self.assertEqual(get_time_context(now=moment, force_local_time=None).context, "DAY")

    def test_day_just_before_night_start(self) -> None:
        moment = datetime(2026, 8, 10, 21, 59, 59)
        self.assertFalse(is_night(moment))

    def test_night_evening(self) -> None:
        moment = datetime(2026, 8, 10, 22, 0, 0)
        self.assertTrue(is_night(moment))
        result = get_time_context(now=moment, force_local_time=None)
        self.assertEqual(result.context, "NIGHT")
        self.assertTrue(result.night_context_active)

    def test_night_before_midnight(self) -> None:
        moment = datetime(2026, 8, 10, 23, 15, 32)
        self.assertTrue(is_night(moment))
        self.assertEqual(get_time_context(now=moment, force_local_time=None).time_text, "23:15:32")

    def test_night_after_midnight_crossing(self) -> None:
        moment = datetime(2026, 8, 11, 2, 30, 0)
        self.assertTrue(is_night(moment))
        self.assertEqual(get_time_context(now=moment, force_local_time=None).context, "NIGHT")

    def test_night_just_before_morning_end(self) -> None:
        moment = datetime(2026, 8, 11, 5, 59, 59)
        self.assertTrue(is_night(moment))

    def test_custom_same_day_night_window(self) -> None:
        # Non-crossing window still supported.
        self.assertTrue(is_night(time(2, 0, 0), night_start=time(1, 0), night_end=time(5, 0)))
        self.assertFalse(is_night(time(6, 0, 0), night_start=time(1, 0), night_end=time(5, 0)))

    def test_force_local_time_override(self) -> None:
        result = get_time_context(force_local_time="23:15:32")
        self.assertTrue(result.using_forced_time)
        self.assertEqual(result.context, "NIGHT")
        self.assertEqual(result.time_text, "23:15:32")


if __name__ == "__main__":
    unittest.main()