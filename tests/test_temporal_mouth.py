"""Unit tests for Stage 5 temporal yawn logic."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mediapipe.tasks.python.vision import FaceLandmarksConnections  # noqa: E402
from mouth_features import (  # noqa: E402
    MAR_LEFT_CORNER,
    MAR_RIGHT_CORNER,
    MAR_VERTICAL_PAIRS,
    MAR_YAWN_THRESHOLD,
    _ALL_MAR_INDICES,
)
from temporal_mouth import YAWN_MIN_DURATION_S, YawnTracker  # noqa: E402


class MouthLandmarkIndexTests(unittest.TestCase):
    def test_mar_indices_are_in_mediapipe_lips(self) -> None:
        lips = {c.start for c in FaceLandmarksConnections.FACE_LANDMARKS_LIPS} | {
            c.end for c in FaceLandmarksConnections.FACE_LANDMARKS_LIPS
        }
        self.assertTrue(set(_ALL_MAR_INDICES).issubset(lips))
        self.assertIn(MAR_LEFT_CORNER, lips)
        self.assertIn(MAR_RIGHT_CORNER, lips)
        for pair in MAR_VERTICAL_PAIRS:
            self.assertTrue(set(pair).issubset(lips))


class YawnTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = YawnTracker(
            open_threshold=0.60,
            close_threshold=0.52,
            yawn_min_s=YAWN_MIN_DURATION_S,
        )

    def test_brief_open_is_not_a_yawn(self) -> None:
        self.tracker.update(0.20, 0.0)
        self.tracker.update(0.70, 0.10)
        self.tracker.update(0.70, 0.40)  # 0.30 s open < min duration
        state = self.tracker.update(0.20, 0.50)
        self.assertEqual(state.mouth, "CLOSED")
        self.assertFalse(state.yawn)
        self.assertEqual(self.tracker.yawn_count, 0)

    def test_sustained_open_counts_exactly_one_yawn(self) -> None:
        self.tracker.update(0.20, 0.0)
        state = None
        t = 0.10
        while t <= 0.10 + 2.0:
            state = self.tracker.update(0.75, t)
            t += 0.10
        assert state is not None
        self.assertTrue(state.yawn)
        self.assertEqual(self.tracker.yawn_count, 1)
        # Still open longer: must not increment again
        state = self.tracker.update(0.80, 2.50)
        self.assertEqual(self.tracker.yawn_count, 1)
        self.assertTrue(state.yawn)

    def test_continuous_open_never_double_counts(self) -> None:
        self.tracker.update(0.20, 0.0)
        for i in range(40):
            self.tracker.update(0.90, 0.05 * i)
        self.assertEqual(self.tracker.yawn_count, 1)

    def test_mouth_close_resets_active_yawn_keeps_count(self) -> None:
        self.tracker.update(0.20, 0.0)
        self.tracker.update(0.80, 0.10)
        self.tracker.update(0.80, 0.10 + YAWN_MIN_DURATION_S + 0.05)
        self.assertEqual(self.tracker.yawn_count, 1)
        state = self.tracker.update(0.20, 2.0)
        self.assertEqual(state.mouth, "CLOSED")
        self.assertFalse(state.yawn)
        self.assertEqual(state.current_mouth_open_s, 0.0)
        self.assertEqual(self.tracker.yawn_count, 1)

    def test_two_separate_yawns(self) -> None:
        # First yawn
        self.tracker.update(0.20, 0.0)
        self.tracker.update(0.80, 1.0)
        self.tracker.update(0.80, 1.0 + YAWN_MIN_DURATION_S)
        self.tracker.update(0.20, 3.0)
        # Second yawn
        self.tracker.update(0.80, 4.0)
        self.tracker.update(0.80, 4.0 + YAWN_MIN_DURATION_S)
        self.assertEqual(self.tracker.yawn_count, 2)

    def test_missing_face_aborts_without_extra_yawn(self) -> None:
        self.tracker.update(0.20, 0.0)
        self.tracker.update(0.80, 0.10)
        state = self.tracker.update(None, 0.50)
        self.assertEqual(state.mouth, "UNKNOWN")
        self.assertEqual(self.tracker.yawn_count, 0)
        self.tracker.update(0.20, 0.60)
        self.assertEqual(self.tracker.yawn_count, 0)

    def test_hysteresis_keeps_open_in_band(self) -> None:
        self.tracker.update(0.20, 0.0)
        self.tracker.update(0.70, 0.10)
        # Between close and open thresholds: stay open
        state = self.tracker.update(0.55, 0.30)
        self.assertEqual(state.mouth, "OPEN")


class ConfigTests(unittest.TestCase):
    def test_initial_thresholds(self) -> None:
        self.assertAlmostEqual(MAR_YAWN_THRESHOLD, 0.60)
        self.assertAlmostEqual(YAWN_MIN_DURATION_S, 1.20)


if __name__ == "__main__":
    unittest.main()