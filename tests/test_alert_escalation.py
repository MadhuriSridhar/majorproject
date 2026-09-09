"""Unit tests for Phase 2 AlertEscalationController (4-level state machine)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alert_escalation import (  # noqa: E402
    AlertEscalationController,
    DROWSY_AUDIO_COOLDOWN,
    DROWSY_DISPLAY_SECONDS,
    LOW_FATIGUE_AUDIO_COOLDOWN,
    LOW_FATIGUE_DISPLAY_SECONDS,
    WARNING_AUDIO_COOLDOWN,
    WARNING_DISPLAY_SECONDS,
)


class AlertEntryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ctl = AlertEscalationController()

    def test_normal_no_audio(self) -> None:
        r = self.ctl.update("NORMAL", 0.0)
        self.assertFalse(r.should_play_audio)
        self.assertFalse(r.display_active)

    def test_low_fatigue_entry_triggers_audio_once(self) -> None:
        r = self.ctl.update("LOW_FATIGUE", 1.0)
        self.assertTrue(r.should_play_audio)
        self.assertEqual(r.audio_severity, "LOW_FATIGUE")
        self.assertTrue(r.display_active)

    def test_low_fatigue_continues_no_repeat(self) -> None:
        self.ctl.update("LOW_FATIGUE", 1.0)
        for t in (1.1, 1.5, 2.0, 3.0):
            r = self.ctl.update("LOW_FATIGUE", t)
            self.assertFalse(r.should_play_audio)

    def test_warning_entry_triggers_audio_once(self) -> None:
        r = self.ctl.update("WARNING", 1.0)
        self.assertTrue(r.should_play_audio)
        self.assertEqual(r.audio_severity, "WARNING")

    def test_drowsy_entry_triggers_audio_once(self) -> None:
        r = self.ctl.update("DROWSY", 1.0)
        self.assertTrue(r.should_play_audio)
        self.assertEqual(r.audio_severity, "DROWSY")

    def test_drowsy_continues_no_repeat(self) -> None:
        self.ctl.update("DROWSY", 1.0)
        for t in (1.1, 2.0, 5.0, 9.0):
            r = self.ctl.update("DROWSY", t)
            self.assertFalse(r.should_play_audio)


class CooldownTests(unittest.TestCase):
    def test_low_fatigue_cooldown_blocks_repeat(self) -> None:
        ctl = AlertEscalationController(low_fatigue_cooldown_s=10.0)
        ctl.update("LOW_FATIGUE", 0.0)
        # Recover then re-enter before cooldown.
        ctl.update("NORMAL", 1.0)
        r = ctl.update("LOW_FATIGUE", 2.0)
        self.assertFalse(r.should_play_audio)  # cooldown not elapsed

    def test_low_fatigue_cooldown_allows_after_elapsed(self) -> None:
        ctl = AlertEscalationController(low_fatigue_cooldown_s=10.0)
        ctl.update("LOW_FATIGUE", 0.0)
        ctl.update("NORMAL", 1.0)
        r = ctl.update("LOW_FATIGUE", 11.0)
        self.assertTrue(r.should_play_audio)

    def test_warning_cooldown(self) -> None:
        ctl = AlertEscalationController(warning_cooldown_s=10.0)
        ctl.update("WARNING", 0.0)
        ctl.update("NORMAL", 1.0)
        r = ctl.update("WARNING", 5.0)
        self.assertFalse(r.should_play_audio)  # cooldown not elapsed
        # Recover again, then re-enter after cooldown elapsed.
        ctl.update("NORMAL", 6.0)
        r2 = ctl.update("WARNING", 12.0)
        self.assertTrue(r2.should_play_audio)

    def test_drowsy_cooldown(self) -> None:
        ctl = AlertEscalationController(drowsy_cooldown_s=10.0)
        ctl.update("DROWSY", 0.0)
        ctl.update("NORMAL", 1.0)
        r = ctl.update("DROWSY", 5.0)
        self.assertFalse(r.should_play_audio)  # cooldown not elapsed
        # Recover again, then re-enter after cooldown elapsed.
        ctl.update("NORMAL", 6.0)
        r2 = ctl.update("DROWSY", 12.0)
        self.assertTrue(r2.should_play_audio)

    def test_default_cooldown_constants(self) -> None:
        self.assertGreater(LOW_FATIGUE_AUDIO_COOLDOWN, 0)
        self.assertGreater(WARNING_AUDIO_COOLDOWN, 0)
        self.assertGreater(DROWSY_AUDIO_COOLDOWN, 0)


class PriorityTests(unittest.TestCase):
    def test_drowsy_overrides_low_fatigue(self) -> None:
        ctl = AlertEscalationController()
        ctl.update("LOW_FATIGUE", 0.0)
        r = ctl.update("DROWSY", 1.0)
        self.assertEqual(r.active_severity, "DROWSY")
        self.assertTrue(r.should_play_audio)

    def test_warning_overrides_low_fatigue(self) -> None:
        ctl = AlertEscalationController()
        ctl.update("LOW_FATIGUE", 0.0)
        r = ctl.update("WARNING", 1.0)
        self.assertEqual(r.active_severity, "WARNING")

    def test_low_fatigue_does_not_override_warning(self) -> None:
        ctl = AlertEscalationController()
        ctl.update("WARNING", 0.0)
        r = ctl.update("LOW_FATIGUE", 1.0)
        # The controller tracks current severity; LOW_FATIGUE is lower priority.
        # It should not trigger audio for a lower severity while WARNING active.
        self.assertFalse(r.should_play_audio)


class DisplayTimerTests(unittest.TestCase):
    def test_display_countdown_decreases(self) -> None:
        ctl = AlertEscalationController(low_fatigue_display_s=5.0)
        r0 = ctl.update("LOW_FATIGUE", 0.0)
        self.assertGreater(r0.display_remaining_s, 4.0)
        r1 = ctl.update("LOW_FATIGUE", 2.0)
        self.assertLess(r1.display_remaining_s, r0.display_remaining_s)

    def test_display_expires(self) -> None:
        ctl = AlertEscalationController(low_fatigue_display_s=2.0)
        ctl.update("LOW_FATIGUE", 0.0)
        r = ctl.update("LOW_FATIGUE", 3.0)
        self.assertFalse(r.display_active)

    def test_display_duration_constants(self) -> None:
        self.assertGreater(LOW_FATIGUE_DISPLAY_SECONDS, 0)
        self.assertGreater(WARNING_DISPLAY_SECONDS, 0)
        self.assertGreater(DROWSY_DISPLAY_SECONDS, 0)


class RecoveryTests(unittest.TestCase):
    def test_recovery_resets_alert_eligibility(self) -> None:
        ctl = AlertEscalationController(drowsy_cooldown_s=0.0)
        ctl.update("DROWSY", 0.0)
        ctl.update("NORMAL", 1.0)
        r = ctl.update("DROWSY", 2.0)
        self.assertTrue(r.should_play_audio)

    def test_alert_persists_until_display_window_expires(self) -> None:
        ctl = AlertEscalationController(drowsy_display_s=5.0)
        ctl.update("DROWSY", 0.0)
        r = ctl.update("NORMAL", 1.0)
        self.assertEqual(r.active_severity, "DROWSY")
        self.assertTrue(r.display_active)
        self.assertFalse(r.should_play_audio)
        r2 = ctl.update("NORMAL", 6.0)
        self.assertFalse(r2.display_active)
        self.assertEqual(r2.active_severity, "NORMAL")

    def test_unknown_severity_normalized(self) -> None:
        ctl = AlertEscalationController()
        r = ctl.update("UNKNOWN", 0.0)
        self.assertFalse(r.should_play_audio)
        self.assertEqual(r.active_severity, "NORMAL")


if __name__ == "__main__":
    unittest.main()