"""
Focused unit tests for context-aware alertness (weather + GPS as contextual
modifiers to *existing* fatigue / distraction evidence).

Covers the 22 acceptance scenarios from the feature spec:

  1-7    context only modifies meaningful behavioural risk; it never creates it
  8-10   GPS / weather / geofence fail-safety and honesty
  11-13  zone-entry and weather are events, not per-frame alarms
  14-17  context cannot invent/erase the driver's behavioural evidence
  18     context urgency stays within safe bounds
  (19-22 existing fatigue / distraction / escalation / audio tests)

No real GPS, network, weather, or audio I/O.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alert_escalation import AlertEscalationController  # noqa: E402
from attention_engine import (  # noqa: E402
    CONTEXT_URGENCY_MIN,
    CONTEXT_URGENCY_NORMAL,
    AttentionEngine,
    apply_context_boost,
    context_urgency_from,
)
from distraction_monitor import DistractionChannel, DistractionObservation  # noqa: E402
from head_pose import HeadPoseObservation  # noqa: E402
from location_context import (  # noqa: E402
    LocationRiskProvider,
    ZoneTransitionTracker,
    classify_geofence,
    load_geofences,
)


def _dist(
    risk: float,
    primary: str = "head_turn_left",
    duration_s: float = 4.0,
) -> DistractionObservation:
    head = HeadPoseObservation(
        direction="left",
        yaw_deg=35.0,
        pitch_deg=0.0,
        looking_away=True,
        confidence=1.0,
        duration_s=duration_s,
        active=True,
    )
    ch = DistractionChannel(primary, True, duration_s, float(risk))
    empty = DistractionChannel("none", False, 0.0, 0.0)
    return DistractionObservation(
        head=head,
        head_turn_left=ch if primary == "head_turn_left" else empty,
        head_turn_right=empty,
        looking_down=empty,
        passenger_side=empty,
        distraction_risk=float(risk),
        primary=primary if risk >= 20.0 else "none",
        active=bool(risk >= 15.0),
        phone_available=False,
    )


class ContextUrgencyDerivationTests(unittest.TestCase):
    def test_normal_context_is_neutral(self) -> None:
        self.assertEqual(context_urgency_from(5.0, 5.0, "none"), CONTEXT_URGENCY_NORMAL)

    def test_heavy_rain_lowers_urgency(self) -> None:
        normal = context_urgency_from(5.0, 5.0, "none")
        rain = context_urgency_from(5.0, 80.0, "none")
        self.assertLess(rain, normal)

    def test_school_zone_lowers_urgency(self) -> None:
        normal = context_urgency_from(5.0, 5.0, "none")
        school = context_urgency_from(70.0, 5.0, "school")
        self.assertLess(school, normal)

    def test_urgency_returns_to_normal_after_leaving_zone(self) -> None:
        entering = context_urgency_from(70.0, 5.0, "school")
        self.assertLess(entering, CONTEXT_URGENCY_NORMAL)
        leaving = context_urgency_from(5.0, 5.0, "none")
        self.assertEqual(leaving, CONTEXT_URGENCY_NORMAL)

    def test_urgency_bounded_safe(self) -> None:
        cases = [
            (0.0, 0.0, "none"),
            (5.0, 5.0, "none"),
            (100.0, 100.0, "school"),
            (120.0, -5.0, "sharp_turn"),  # out-of-range inputs are clamped
        ]
        for loc, wx, zone in cases:
            urgency = context_urgency_from(loc, wx, zone)
            self.assertGreaterEqual(urgency, CONTEXT_URGENCY_MIN)
            self.assertLessEqual(urgency, CONTEXT_URGENCY_NORMAL)


class ContextCannotCreateAlertTests(unittest.TestCase):
    def test_normal_weather_normal_behaviour_no_penalty(self) -> None:
        self.assertEqual(apply_context_boost(0.0, 0.0, 5.0, 5.0, "none"), 0.0)
        eng = AttentionEngine()
        out = eng.update(fatigue_risk=0.0, distraction=None, now_s=0.0)
        self.assertEqual(out.level, "SAFE")
        self.assertEqual(out.alert_severity, "NORMAL")
        self.assertEqual(out.context_urgency, CONTEXT_URGENCY_NORMAL)

    def test_heavy_rain_normal_behaviour_no_alert(self) -> None:
        eng = AttentionEngine()
        out = eng.update(
            fatigue_risk=0.0,
            distraction=None,
            now_s=0.0,
            weather_risk=80.0,
            weather_condition="heavy_rain",
        )
        self.assertEqual(out.level, "SAFE")
        self.assertEqual(out.alert_severity, "NORMAL")

    def test_school_zone_normal_behaviour_no_alert(self) -> None:
        eng = AttentionEngine()
        out = eng.update(
            fatigue_risk=0.0,
            distraction=None,
            now_s=0.0,
            location_risk=70.0,
            location_zone="school",
        )
        self.assertEqual(out.level, "SAFE")
        self.assertEqual(out.alert_severity, "NORMAL")

    def test_context_cannot_create_fatigue(self) -> None:
        eng = AttentionEngine()
        out = eng.update(
            fatigue_risk=0.0,
            distraction=None,
            now_s=0.0,
            location_risk=70.0,
            weather_risk=80.0,
            location_zone="school",
            weather_condition="heavy_rain",
        )
        self.assertEqual(out.fatigue_risk, 0.0)
        self.assertNotIn("fatigue", out.primary_reason.lower())
        self.assertNotIn("eye closure", out.primary_reason.lower())

    def test_context_cannot_create_distraction(self) -> None:
        eng = AttentionEngine()
        out = eng.update(
            fatigue_risk=0.0,
            distraction=None,
            now_s=0.0,
            location_risk=70.0,
            weather_risk=80.0,
            location_zone="school",
            weather_condition="heavy_rain",
        )
        self.assertEqual(out.distraction_risk, 0.0)
        self.assertEqual(out.distraction_primary, "none")
        self.assertNotIn("distraction", out.primary_reason.lower())


class ContextModifiesExistingRiskTests(unittest.TestCase):
    def test_heavy_rain_raises_existing_distraction(self) -> None:
        d = 45.0
        base = apply_context_boost(d, d, 5.0, 5.0, "none")
        rain = apply_context_boost(d, d, 5.0, 80.0, "none")
        self.assertGreater(rain, base)
        self.assertLessEqual(rain, 100.0)
        self.assertLess(
            context_urgency_from(5.0, 80.0, "none"),
            context_urgency_from(5.0, 5.0, "none"),
        )

    def test_heavy_rain_raises_existing_fatigue(self) -> None:
        clear = AttentionEngine().update(
            fatigue_risk=70.0, distraction=None, now_s=0.0, weather_risk=5.0
        )
        rain = AttentionEngine().update(
            fatigue_risk=70.0,
            distraction=None,
            now_s=0.0,
            weather_risk=80.0,
            weather_condition="heavy_rain",
        )
        self.assertGreater(rain.raw_combined, clear.raw_combined)
        self.assertLess(rain.context_urgency, clear.context_urgency)

    def test_school_zone_raises_existing_distraction(self) -> None:
        d = 45.0
        base = apply_context_boost(d, d, 5.0, 5.0, "none")
        school = apply_context_boost(d, d, 70.0, 5.0, "school")
        self.assertGreater(school, base)
        self.assertLessEqual(school, 100.0)

    def test_brief_glance_not_severe_in_high_risk_zone(self) -> None:
        eng = AttentionEngine()
        out = eng.update(
            fatigue_risk=0.0,
            distraction=_dist(15.0, duration_s=0.5),
            now_s=0.0,
            location_risk=70.0,
            location_zone="school",
        )
        self.assertIn(out.level, ("SAFE", "MODERATE"))
        self.assertNotIn(out.alert_severity, ("WARNING", "DROWSY"))


class EscalationUrgencyTests(unittest.TestCase):
    def test_context_urgency_shortens_cooldown(self) -> None:
        normal = AlertEscalationController()
        urgent = AlertEscalationController()

        for controller in (normal, urgent):
            controller.update("ALERT", 0.0)
            controller.update("NORMAL", 0.1)
            controller.update("NORMAL", 2.0)

        normal_out = normal.update("ALERT", 15.0, context_urgency=1.0)
        urgent_out = urgent.update("ALERT", 15.0, context_urgency=0.5)

        # A 15s gap is below the normal ALERT cooldown (25s) but above the
        # halved cooldown of a high-risk context (12.5s), so the alert repeats.
        self.assertFalse(normal_out.should_play_audio)
        self.assertTrue(urgent_out.should_play_audio)

    def test_context_urgency_alone_never_triggers_audio(self) -> None:
        controller = AlertEscalationController()
        for t in range(0, 60):
            result = controller.update(
                "NORMAL",
                float(t),
                context_urgency=context_urgency_from(70.0, 80.0, "school"),
            )
            self.assertFalse(result.should_play_audio)


class FailSafeTests(unittest.TestCase):
    def test_no_gps_monitoring_continues(self) -> None:
        provider = LocationRiskProvider(mode="gps", update_interval_s=0.0)
        with patch.object(provider, "_try_windows_geolocation", return_value=None), patch.object(
            provider, "_try_ip_geolocation", return_value=None
        ), patch.object(provider, "_load_cache"), patch.object(provider, "_save_cache"):
            provider._last_ctx = provider._unavailable("Unavailable")
            ctx = provider.update(10.0)
        self.assertFalse(ctx.available)
        out = AttentionEngine().update(
            fatigue_risk=70.0,
            distraction=None,
            now_s=0.0,
            location_risk=0.0,
        )
        self.assertIn(out.level, ("HIGH", "CRITICAL"))
        self.assertGreaterEqual(out.combined_risk, 50.0)

    def test_no_weather_monitoring_continues(self) -> None:
        out = AttentionEngine().update(
            fatigue_risk=70.0,
            distraction=None,
            now_s=0.0,
            weather_risk=0.0,
        )
        self.assertIn(out.level, ("HIGH", "CRITICAL"))
        self.assertGreaterEqual(out.combined_risk, 50.0)

    def test_empty_geofence_no_false_zone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "geofences.json"
            path.write_text(json.dumps([]))
            zones = load_geofences(path)
        self.assertEqual(zones, [])
        self.assertIsNone(classify_geofence(12.97, 77.59, zones))

    def test_weather_refresh_never_voices_alert(self) -> None:
        eng = AttentionEngine()
        for wx in (80.0, 80.0, 90.0, 80.0):
            out = eng.update(
                fatigue_risk=0.0,
                distraction=None,
                now_s=float(wx),
                weather_risk=wx,
                weather_condition="heavy_rain",
            )
            self.assertEqual(out.alert_severity, "NORMAL")
            self.assertNotIn("weather", out.primary_reason.lower())


class HistoryPreservationTests(unittest.TestCase):
    def test_context_does_not_erase_alertness_history(self) -> None:
        eng = AttentionEngine()
        first = eng.update(fatigue_risk=60.0, distraction=None, now_s=0.0)
        self.assertGreater(first.combined_risk, 0.0)
        second = eng.update(
            fatigue_risk=0.0,
            distraction=None,
            now_s=1.0,
            location_risk=70.0,
            weather_risk=80.0,
            location_zone="school",
            weather_condition="heavy_rain",
        )
        self.assertGreater(second.combined_risk, 0.0)
        self.assertEqual(second.combined_risk, first.combined_risk)

    def test_context_does_not_override_behavioural_evidence(self) -> None:
        clear = AttentionEngine().update(
            fatigue_risk=45.0, distraction=_dist(40.0), now_s=0.0
        )
        eng = AttentionEngine()
        out = eng.update(
            fatigue_risk=45.0,
            distraction=_dist(40.0),
            now_s=0.0,
            location_risk=70.0,
            weather_risk=80.0,
            location_zone="school",
            weather_condition="heavy_rain",
        )
        self.assertEqual(out.fatigue_risk, 45.0)
        self.assertEqual(out.distraction_risk, 40.0)
        # Context only amplifies; it never reduces the behavioural signal.
        self.assertGreaterEqual(out.combined_risk, clear.combined_risk)


class ZoneEventTests(unittest.TestCase):
    def test_zone_entry_single_event(self) -> None:
        tracker = ZoneTransitionTracker()
        first = tracker.update("school", "School Zone", 0.0)
        same = tracker.update("school", "School Zone", 1.0)
        self.assertIsNotNone(first)
        self.assertEqual(first.event, "enter")
        self.assertIsNone(same)

    def test_remaining_in_same_zone_no_repeat(self) -> None:
        tracker = ZoneTransitionTracker()
        tracker.update("school", "School Zone", 0.0)
        for t in (1.0, 2.0, 3.0, 4.0):
            self.assertIsNone(tracker.update("school", "School Zone", t))


if __name__ == "__main__":
    unittest.main()



