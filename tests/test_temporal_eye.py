"""Unit tests for Stage 4 temporal blink / prolonged-closure logic."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from temporal_eye import (  # noqa: E402
    MAX_BLINK_DURATION_S,
    MIN_BLINK_DURATION_S,
    PROLONGED_CLOSURE_S,
    EyeClosureTracker,
)


class EyeClosureTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = EyeClosureTracker(
            close_threshold=0.21,
            open_threshold=0.23,
            min_blink_s=MIN_BLINK_DURATION_S,
            max_blink_s=MAX_BLINK_DURATION_S,
            prolonged_s=PROLONGED_CLOSURE_S,
        )

    def _run_blink(self, start: float = 1.0, duration: float = 0.30) -> None:
        # Open baseline
        self.tracker.update(0.30, start - 0.05)
        # Closed frames during the blink
        t = start
        while t < start + duration:
            state = self.tracker.update(0.10, t)
            self.assertIn(state.eye, ("CLOSING", "CLOSED"))
            t += 0.05
        # Reopen
        state = self.tracker.update(0.30, start + duration)
        self.assertEqual(state.eye, "OPEN")

    def test_one_blink_increments_count_by_one(self) -> None:
        self._run_blink(duration=0.30)
        self.assertEqual(self.tracker.blink_count, 1)

    def test_many_closed_frames_still_one_blink(self) -> None:
        # Dense frames inside one short closure must still count once.
        self.tracker.update(0.30, 0.0)
        for i in range(12):
            self.tracker.update(0.10, 0.10 + i * 0.02)  # 0.10 .. 0.32 s closed
        self.tracker.update(0.30, 0.40)
        self.assertEqual(self.tracker.blink_count, 1)

    def test_two_separate_blinks(self) -> None:
        self._run_blink(start=1.0, duration=0.25)
        self._run_blink(start=2.0, duration=0.25)
        self.assertEqual(self.tracker.blink_count, 2)

    def test_too_short_noise_is_not_a_blink(self) -> None:
        self.tracker.update(0.30, 0.0)
        self.tracker.update(0.10, 0.10)
        self.tracker.update(0.30, 0.10 + 0.05)  # 50 ms < min blink
        self.assertEqual(self.tracker.blink_count, 0)

    def test_prolonged_closure_sets_flag_without_blink(self) -> None:
        self.tracker.update(0.30, 0.0)
        state = None
        t = 0.10
        while t <= 0.10 + 2.0:
            state = self.tracker.update(0.10, t)
            t += 0.10
        assert state is not None
        self.assertTrue(state.prolonged_closure)
        self.assertGreaterEqual(state.current_closure_s, PROLONGED_CLOSURE_S)
        # Reopen after prolonged closure: no blink
        state = self.tracker.update(0.30, 2.20)
        self.assertEqual(state.eye, "OPEN")
        self.assertFalse(state.prolonged_closure)
        self.assertEqual(self.tracker.blink_count, 0)

    def test_missing_face_aborts_closure_without_blink(self) -> None:
        self.tracker.update(0.30, 0.0)
        self.tracker.update(0.10, 0.10)
        state = self.tracker.update(None, 0.20)
        self.assertEqual(state.eye, "UNKNOWN")
        self.assertEqual(state.current_closure_s, 0.0)
        self.assertFalse(state.prolonged_closure)
        # Later open should not suddenly count the aborted event
        self.tracker.update(0.30, 0.30)
        self.assertEqual(self.tracker.blink_count, 0)

    def test_hysteresis_avoids_immediate_reopen_in_band(self) -> None:
        self.tracker.update(0.30, 0.0)
        self.tracker.update(0.10, 0.10)  # close
        # EAR in the hysteresis band (0.21..0.23) should stay closed
        state = self.tracker.update(0.22, 0.20)
        self.assertIn(state.eye, ("CLOSING", "CLOSED"))
        # Clear open above open_threshold
        state = self.tracker.update(0.30, 0.40)
        self.assertEqual(state.eye, "OPEN")
        self.assertEqual(self.tracker.blink_count, 1)


class ColdStartBlinkRateTests(unittest.TestCase):
    def _blink(self, tracker, start: float, duration: float = 0.3) -> None:
        tracker.update(0.30, start - 0.05)
        t = start
        while t < start + duration:
            tracker.update(0.10, t)
            t += 0.05
        tracker.update(0.30, start + duration)

    def test_rate_uses_elapsed_time_during_first_minute(self) -> None:
        tracker = EyeClosureTracker()
        # Two blinks within the first 20 seconds of monitoring.
        tracker.update(0.30, 0.0)  # marks monitoring start
        self._blink(tracker, start=5.0)
        self._blink(tracker, start=12.0)
        state = tracker.update(0.30, 20.0)
        # (2 blinks / 20s) * 60 = 6.0, not (2 / 60) * 60 = 2.0.
        self.assertAlmostEqual(state.blink_rate, 6.0, places=4)

    def test_rate_uses_full_window_after_minute(self) -> None:
        tracker = EyeClosureTracker()
        tracker.update(0.30, 0.0)
        self._blink(tracker, start=5.0)
        self._blink(tracker, start=12.0)
        state = tracker.update(0.30, 73.0)
        # Both blinks are older than the 60s window now -> zero in-window events.
        self.assertEqual(state.blink_rate, 0.0)


if __name__ == "__main__":
    unittest.main()