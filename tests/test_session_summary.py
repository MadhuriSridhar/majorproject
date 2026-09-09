"""Unit tests for Phase 2 session summary + PDF export."""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fatigue_engine import FatigueEngine, FatigueEvidence  # noqa: E402
from session_manager import SessionManager  # noqa: E402
from session_summary import build_summary_text, export_pdf  # noqa: E402


def _populated_stats():
    """Return a SessionStats with realistic data via a real session + engine."""
    manager = SessionManager(use_demo_duration=True)
    engine = FatigueEngine(
        drowsy_enter_hold_s=0.0,
        drowsy_exit_hold_s=0.0,
        drowsy_persistence_s=0.0,
        warning_persistence_s=0.0,
    )
    manager.start()
    # Normal frames.
    for i in range(10):
        d = engine.decide(FatigueEvidence(face_detected=True, blink_count=2, yawn_count=0))
        manager.update(d, now_s=float(i))
    # Some drowsy frames.
    for i in range(10, 15):
        d = engine.decide(
            FatigueEvidence(
                prolonged_eye_closure=True,
                current_eye_closure_seconds=2.5,
                face_detected=True,
            )
        )
        manager.update(d, now_s=float(i))
    return manager.stop()


class SummaryTextTests(unittest.TestCase):
    def test_summary_contains_key_fields(self) -> None:
        stats = _populated_stats()
        text = build_summary_text(stats)
        self.assertIn("SESSION SUMMARY", text)
        self.assertIn("Driving Time", text)
        self.assertIn("Average Fatigue Risk", text)
        self.assertIn("Maximum Fatigue Risk", text)
        self.assertIn("Total Blinks", text)
        self.assertIn("Total Yawns", text)
        self.assertIn("Final Status", text)
        self.assertIn("Risk-factor summary", text)
        self.assertIn("Eye", text)
        self.assertIn("Blink", text)
        self.assertIn("Yawn", text)
        self.assertIn("Night", text)
        self.assertIn("Driving Duration", text)

    def test_summary_with_empty_stats(self) -> None:
        manager = SessionManager()
        manager.start()
        stats = manager.stop()
        text = build_summary_text(stats)
        self.assertIn("SESSION SUMMARY", text)


class PdfExportTests(unittest.TestCase):
    def test_pdf_generated(self) -> None:
        stats = _populated_stats()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            pdf_path = export_pdf(stats, output_dir=out)
            self.assertTrue(pdf_path.is_file())
            self.assertGreater(pdf_path.stat().st_size, 0)
            # PDF magic bytes.
            data = pdf_path.read_bytes()[:5]
            self.assertEqual(data, b"%PDF-")

    def test_pdf_with_empty_session(self) -> None:
        manager = SessionManager()
        manager.start()
        stats = manager.stop()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            pdf_path = export_pdf(stats, output_dir=out)
            self.assertTrue(pdf_path.is_file())


if __name__ == "__main__":
    unittest.main()