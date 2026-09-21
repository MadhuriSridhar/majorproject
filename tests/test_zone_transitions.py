"""
Focused tests for zone entry/exit tracking and location-context honesty.

No real GPS / network / Overpass calls.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attention_engine import AttentionEngine  # noqa: E402
from distraction_monitor import DistractionChannel, DistractionObservation  # noqa: E402
from head_pose import HeadPoseObservation  # noqa: E402
from location_context import (  # noqa: E402
    BASELINE_LOCATION_RISK,
    NO_ZONE_NAME,
    GeofenceZone,
    LocationContext,
    LocationRiskProvider,
    ZoneTransitionTracker,
    classify_geofence,
    format_location_summary,
    format_zone_event_summary,
    load_geofences,
)


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
    return DistractionObservation(
        head=head,
        head_turn_left=DistractionChannel("head_turn_left", True, 4.0, risk),
        head_turn_right=DistractionChannel("none", False, 0.0, 0.0),
        looking_down=DistractionChannel("none", False, 0.0, 0.0),
        passenger_side=DistractionChannel("none", False, 0.0, 0.0),
        distraction_risk=float(risk),
        primary="head_turn_left",
        active=True,
        phone_available=False,
    )


class ZoneTransitionTrackerTests(unittest.TestCase):
    def test_enter_then_same_zone_no_repeat(self) -> None:
        tracker = ZoneTransitionTracker()
        first = tracker.update("school", "School Zone", 0.0)
        second = tracker.update("school", "School Zone", 1.0)
        self.assertEqual(first.event, "enter")
        self.assertEqual(first.zone_type, "school")
        self.assertIsNone(second)

    def test_exit_produces_one_event(self) -> None:
        tracker = ZoneTransitionTracker()
        tracker.update("none", "", 0.0)
        tracker.update("school", "School Zone", 1.0)
        exit_event = tracker.update("none", "", 2.0)
        self.assertEqual(exit_event.event, "exit")
        self.assertEqual(exit_event.zone_type, "school")
        self.assertIn("School Zone", format_zone_event_summary(exit_event))

    def test_zone_change_is_not_entry_spam(self) -> None:
        tracker = ZoneTransitionTracker()
        tracker.update("none", "", 0.0)
        tracker.update("school", "School Zone", 1.0)
        change = tracker.update("construction", "Construction Zone", 2.0)
        self.assertEqual(change.event, "change")

    def test_unknown_zone_type_classified_as_no_zone(self) -> None:
        self.assertEqual(ZoneTransitionTracker.classify("weird"), "none")
        self.assertEqual(ZoneTransitionTracker.classify(None), "none")

    def test_first_observation_of_no_zone_emits_nothing(self) -> None:
        tracker = ZoneTransitionTracker()
        self.assertIsNone(tracker.update("none", "", 0.0))
        # Entering a zone still fires exactly one enter event afterwards.
        enter = tracker.update("school", "School Zone", 1.0)
        self.assertEqual(enter.event, "enter")


class GeofenceTests(unittest.TestCase):
    def test_empty_geofence_no_fake_zone(self) -> None:
        zones = load_geofences(Path("__missing_geofences__.json"))
        self.assertEqual(zones, [])
        self.assertIsNone(classify_geofence(12.97, 77.59, []))

    def test_malformed_geofence_fails_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text("{ this is not valid json ", encoding="utf-8")
            self.assertEqual(load_geofences(bad), [])

    def test_valid_geofence_matches_correct_zone(self) -> None:
        zones = [
            GeofenceZone("School Zone", "school", 12.9700, 77.5900, 300.0, 70.0),
            GeofenceZone("Construction", "construction", 13.0000, 77.7000, 500.0, 55.0),
        ]
        hit = classify_geofence(12.9710, 77.5910, zones)
        self.assertIsNotNone(hit)
        self.assertEqual(hit[0], "school")
        self.assertEqual(hit[1], "School Zone")

    def test_no_match_beyond_radius(self) -> None:
        zones = [GeofenceZone("School Zone", "school", 12.9700, 77.5900, 300.0, 70.0)]
        self.assertIsNone(classify_geofence(13.5, 78.5, zones))


class LocationSummaryHonestyTests(unittest.TestCase):
    def test_empty_geofence_context_no_verified_safe_claim(self) -> None:
        ctx = LocationContext(
            latitude=12.97,
            longitude=77.59,
            zone_type="none",
            zone_name=NO_ZONE_NAME,
            risk_level=BASELINE_LOCATION_RISK,
            source="ip",
            place_name="Bengaluru",
            available=True,
            mode="gps",
        )
        text = format_location_summary(ctx)
        self.assertIn("No configured high-risk zone", text)
        self.assertIn("Baseline", text)
        self.assertNotIn("Normal Road", text)
        self.assertNotIn("Verified Safe", text)

    def test_school_zone_shows_high_risk(self) -> None:
        ctx = LocationContext(
            latitude=12.97,
            longitude=77.59,
            zone_type="school",
            zone_name="School Zone",
            risk_level=70.0,
            source="gps",
            place_name="",
            available=True,
            mode="gps",
        )
        text = format_location_summary(ctx)
        self.assertIn("School Zone", text)
        self.assertIn("HIGH RISK", text)


class LocationProviderFallbackTests(unittest.TestCase):
    def test_valid_coordinates_pass_through(self) -> None:
        provider = LocationRiskProvider(mode="gps", update_interval_s=0.0)
        ctx = provider._context_for_coordinates(12.97, 77.59, "gps", "Bengaluru")
        self.assertTrue(ctx.available)
        self.assertEqual(ctx.latitude, 12.97)
        self.assertEqual(ctx.longitude, 77.59)
        self.assertEqual(ctx.source, "gps")
        self.assertEqual(ctx.zone_type, "none")

    def test_ip_fallback_used_when_gps_unavailable(self) -> None:
        provider = LocationRiskProvider(mode="gps", update_interval_s=0.0)
        with patch.object(provider, "_try_windows_geolocation", return_value=None), patch.object(
            provider, "_try_ip_geolocation", return_value=(12.97, 77.59, "Bengaluru")
        ), patch.object(provider, "_save_cache"):
            ctx = provider.update(10.0)
        self.assertTrue(ctx.available)
        self.assertEqual(ctx.source, "ip")
        self.assertEqual(ctx.place_name, "Bengaluru")


class ContextReachesAttentionTests(unittest.TestCase):
    def test_location_and_weather_reach_decision(self) -> None:
        eng = AttentionEngine()
        out = eng.update(
            fatigue_risk=10.0,
            distraction=_dist(30.0),
            now_s=0.0,
            location_risk=70.0,
            weather_risk=80.0,
            location_zone="school",
            weather_condition="heavy_rain",
        )
        self.assertEqual(out.location_risk, 70.0)
        self.assertEqual(out.weather_risk, 80.0)
        self.assertEqual(out.location_zone, "school")
        self.assertEqual(out.weather_condition, "heavy_rain")

    def test_missing_context_does_not_crash(self) -> None:
        eng = AttentionEngine()
        out = eng.update(fatigue_risk=5.0, distraction=_dist(0.0), now_s=0.0)
        self.assertIn(out.level, ("SAFE", "MODERATE"))
        self.assertGreaterEqual(out.alertness_score, 0.0)
        self.assertLessEqual(out.alertness_score, 100.0)


if __name__ == "__main__":
    unittest.main()

