"""
Regression tests for alert stability, attention severity hysteresis, global
audio gap, geofence classification, and fatigue risk recovery.

No webcam / MediaPipe / network required.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alert_escalation import AlertEscalationController  # noqa: E402
from attention_engine import AttentionEngine  # noqa: E402
from fatigue_engine import FatigueEngine, FatigueEvidence  # noqa: E402
from location_context import (  # noqa: E402
    GeofenceZone,
    LocationRiskProvider,
    classify_geofence,
)


def E(**kwargs) -> FatigueEvidence:
    return FatigueEvidence(**kwargs)


class WarningPersistenceTests(unittest.TestCase):
    def test_warning_requires_persistence(self) -> None:
        eng = AttentionEngine()
        # 60 -> HIGH band (WARNING).
        first = eng.update(fatigue_risk=60.0, distraction=None, now_s=0.0)
        self.assertNotEqual(first.alert_severity, "WARNING")
        # Sustained HIGH evidence past the one-second persistence window.
        eng.update(fatigue_risk=60.0, distraction=None, now_s=0.5)
        settled = eng.update(fatigue_risk=60.0, distraction=None, now_s=1.1)
        self.assertEqual(settled.alert_severity, "WARNING")

    def test_critical_escalates_immediately(self) -> None:
        eng = AttentionEngine()
        # 90 -> CRITICAL band (DROWSY).
        eng.update(fatigue_risk=90.0, distraction=None, now_s=0.0)
        out = eng.update(fatigue_risk=90.0, distraction=None, now_s=1.6)
        self.assertEqual(out.level, "CRITICAL")
        self.assertEqual(out.alert_severity, "DROWSY")


class AlertHysteresisTests(unittest.TestCase):
    def test_alert_enter_exit_hysteresis(self) -> None:
        eng = AttentionEngine()
        # Enter ALERT at 35.
        self.assertEqual(eng._derive_alert_severity(35.0, "MODERATE", 0.0), "NORMAL")
        self.assertEqual(eng._derive_alert_severity(35.0, "MODERATE", 0.8), "ALERT")
        # Stay ALERT down to the exit threshold (32).
        self.assertEqual(eng._derive_alert_severity(33.0, "MODERATE", 0.9), "ALERT")
        self.assertEqual(eng._derive_alert_severity(32.0, "MODERATE", 1.0), "ALERT")
        # Below exit threshold -> LOW_FATIGUE.
        self.assertEqual(eng._derive_alert_severity(31.9, "MODERATE", 2.4), "ALERT")
        self.assertEqual(eng._derive_alert_severity(31.9, "MODERATE", 3.8), "LOW_FATIGUE")
        # Re-enter requires clearing the entry threshold again.
        self.assertEqual(eng._derive_alert_severity(34.0, "MODERATE", 4.0), "LOW_FATIGUE")
        self.assertEqual(eng._derive_alert_severity(35.0, "MODERATE", 4.1), "LOW_FATIGUE")
        self.assertEqual(eng._derive_alert_severity(35.0, "MODERATE", 4.9), "ALERT")


class GlobalAudioGapTests(unittest.TestCase):
    def test_global_gap_blocks_reentry_within_gap(self) -> None:
        ctl = AlertEscalationController(
            alert_cooldown_s=0.0,
            alert_display_s=0.5,
            global_audio_gap_s=5.0,
            episode_recovery_s=0.0,
            episode_cooldown_s=0.0,
        )
        ctl.update("ALERT", 0.0)  # plays
        ctl.update("NORMAL", 1.0)  # recover
        blocked = ctl.update("ALERT", 2.0)  # within global gap, same severity
        self.assertFalse(blocked.should_play_audio)
        ctl.update("NORMAL", 3.0)
        allowed = ctl.update("ALERT", 7.0)  # gap elapsed
        self.assertTrue(allowed.should_play_audio)

    def test_higher_priority_interrupts_within_gap(self) -> None:
        ctl = AlertEscalationController(
            low_fatigue_cooldown_s=0.0,
            warning_cooldown_s=0.0,
            low_fatigue_display_s=0.5,
            global_audio_gap_s=5.0,
        )
        ctl.update("LOW_FATIGUE", 0.0)  # plays
        high = ctl.update("WARNING", 1.0)  # higher priority interrupts
        self.assertTrue(high.should_play_audio)

    def test_critical_not_blocked_by_gap(self) -> None:
        ctl = AlertEscalationController(
            low_fatigue_cooldown_s=0.0,
            drowsy_cooldown_s=0.0,
            low_fatigue_display_s=0.5,
            global_audio_gap_s=5.0,
        )
        ctl.update("LOW_FATIGUE", 0.0)
        critical = ctl.update("DROWSY", 1.0)
        self.assertTrue(critical.should_play_audio)


class AlertEpisodeSpamTests(unittest.TestCase):
    def test_threshold_oscillation_does_not_create_repeated_audio(self) -> None:
        ctl = AlertEscalationController(
            episode_recovery_s=1.0,
            episode_cooldown_s=5.0,
            warning_cooldown_s=0.0,
        )
        outputs = [
            ctl.update("WARNING", t)
            for t in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        ]
        self.assertEqual(sum(result.should_play_audio for result in outputs), 1)
        self.assertEqual(ctl.debug_state["episode_id"], 1)

    def test_escalation_is_one_episode(self) -> None:
        ctl = AlertEscalationController()
        first = ctl.update("ALERT", 0.0)
        warning = ctl.update("WARNING", 1.0)
        drowsy = ctl.update("DROWSY", 2.0)
        self.assertTrue(first.should_play_audio)
        self.assertTrue(warning.should_play_audio)
        self.assertTrue(drowsy.should_play_audio)
        self.assertEqual(ctl.debug_state["episode_id"], 1)

    def test_short_recovery_does_not_start_new_episode(self) -> None:
        ctl = AlertEscalationController(episode_recovery_s=1.5, episode_cooldown_s=5.0)
        ctl.update("WARNING", 0.0)
        ctl.update("NORMAL", 1.0)
        result = ctl.update("WARNING", 1.2)
        self.assertEqual(result.active_severity, "WARNING")
        self.assertEqual(ctl.debug_state["episode_id"], 1)

    def test_sustained_recovery_then_cooldown_allows_new_episode(self) -> None:
        ctl = AlertEscalationController(
            episode_recovery_s=1.0,
            episode_cooldown_s=3.0,
            warning_cooldown_s=0.0,
        )
        ctl.update("WARNING", 0.0)
        ctl.update("NORMAL", 1.0)
        ctl.update("NORMAL", 2.1)
        blocked = ctl.update("WARNING", 3.0)
        allowed = ctl.update("WARNING", 5.2)
        self.assertFalse(blocked.should_play_audio)
        self.assertTrue(allowed.should_play_audio)
        self.assertEqual(ctl.debug_state["episode_id"], 2)

    def test_primary_reason_remains_stable_inside_episode(self) -> None:
        ctl = AlertEscalationController()
        first = ctl.update("WARNING", 0.0, reason="Sustained attention away from the road")
        later = ctl.update("WARNING", 10.0, reason="Fatigue risk elevated")
        self.assertEqual(first.reason, "Sustained attention away from the road")
        self.assertEqual(later.reason, "Sustained attention away from the road")


class GeofenceTests(unittest.TestCase):
    def test_classify_within_radius(self) -> None:
        zones = [GeofenceZone("School", "school", 10.0, 10.0, 300.0, 70.0)]
        hit = classify_geofence(10.001, 10.0, zones)
        self.assertIsNotNone(hit)
        self.assertEqual(hit[0], "school")
        self.assertEqual(hit[2], 70.0)

    def test_classify_outside_radius(self) -> None:
        zones = [GeofenceZone("School", "school", 10.0, 10.0, 300.0, 70.0)]
        self.assertIsNone(classify_geofence(10.1, 10.0, zones))

    def test_provider_applies_geofence(self) -> None:
        provider = LocationRiskProvider(mode="gps", update_interval_s=0.0)
        provider._geofences = [
            GeofenceZone("School", "school", 12.9716, 77.5946, 1000.0, 70.0)
        ]
        ctx = provider._context_for_coordinates(12.9716, 77.5946, "gps", "GPS fix")
        self.assertEqual(ctx.zone_type, "school")
        self.assertEqual(ctx.zone_name, "School")
        self.assertEqual(ctx.risk_level, 70.0)

    def test_provider_no_geofence_is_baseline_no_zone(self) -> None:
        provider = LocationRiskProvider(mode="gps", update_interval_s=0.0)
        provider._geofences = []
        ctx = provider._context_for_coordinates(12.9716, 77.5946, "gps", "GPS fix")
        self.assertEqual(ctx.zone_type, "none")
        self.assertEqual(ctx.zone_name, "No configured high-risk zone")
        self.assertEqual(ctx.risk_level, 5.0)


class FatigueRecoveryTests(unittest.TestCase):
    def test_risk_not_reset_by_single_good_frame(self) -> None:
        eng = FatigueEngine(drowsy_enter_hold_s=0.0, drowsy_exit_hold_s=0.0)
        high = eng.update(E(prolonged_eye_closure=True), timestamp_s=0.0)
        self.assertGreaterEqual(high.smoothed_risk, 60.0)
        # A single good frame must not instantly drop the risk.
        improved = eng.update(E(), timestamp_s=0.2)
        self.assertGreaterEqual(improved.smoothed_risk, 60.0)


if __name__ == "__main__":
    unittest.main()
