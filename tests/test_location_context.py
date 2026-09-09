"""
Focused unit tests for GPS / location context (demo + disabled + unavailable).

No real GPS / network calls.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from location_context import (  # noqa: E402
    DEMO_ZONE_PRESETS,
    LocationRiskProvider,
    format_location_summary,
)


class LocationModeTests(unittest.TestCase):
    def test_disabled_mode_gives_no_location_risk(self) -> None:
        provider = LocationRiskProvider(mode="disabled")
        ctx = provider.update(1.0)
        self.assertFalse(ctx.available)
        self.assertEqual(ctx.risk_level, 0.0)
        self.assertEqual(ctx.source, "Disabled")
        self.assertIn("Disabled", format_location_summary(ctx))

    def test_demo_normal_road(self) -> None:
        provider = LocationRiskProvider(mode="demo", demo_zone="normal")
        ctx = provider.update(1.0)
        self.assertTrue(ctx.available)
        self.assertEqual(ctx.source, "DEMO")
        self.assertEqual(ctx.zone_name, DEMO_ZONE_PRESETS["normal"][0])
        self.assertEqual(ctx.risk_level, DEMO_ZONE_PRESETS["normal"][2])

    def test_demo_school_zone_higher_risk(self) -> None:
        normal = LocationRiskProvider(mode="demo", demo_zone="normal").update(1.0)
        school = LocationRiskProvider(mode="demo", demo_zone="school").update(1.0)
        self.assertEqual(school.source, "DEMO")
        self.assertEqual(school.zone_type, "school")
        self.assertGreater(school.risk_level, normal.risk_level)

    def test_demo_highway(self) -> None:
        ctx = LocationRiskProvider(mode="demo", demo_zone="highway").update(1.0)
        self.assertEqual(ctx.source, "DEMO")
        self.assertEqual(ctx.zone_type, "highway")
        self.assertEqual(ctx.risk_level, DEMO_ZONE_PRESETS["highway"][2])

    def test_demo_construction(self) -> None:
        ctx = LocationRiskProvider(mode="demo", demo_zone="construction").update(1.0)
        self.assertEqual(ctx.source, "DEMO")
        self.assertEqual(ctx.zone_type, "construction")
        self.assertEqual(ctx.risk_level, DEMO_ZONE_PRESETS["construction"][2])

    def test_demo_sharp_turn(self) -> None:
        ctx = LocationRiskProvider(mode="demo", demo_zone="sharp_turn").update(1.0)
        self.assertEqual(ctx.source, "DEMO")
        self.assertEqual(ctx.zone_type, "sharp_turn")
        self.assertEqual(ctx.risk_level, DEMO_ZONE_PRESETS["sharp_turn"][2])

    def test_demo_source_clearly_demo(self) -> None:
        ctx = LocationRiskProvider(mode="demo", demo_zone="school").update(2.0)
        summary = format_location_summary(ctx)
        self.assertIn("DEMO", summary)
        self.assertEqual(ctx.source, "DEMO")

    def test_unavailable_gps_does_not_crash(self) -> None:
        provider = LocationRiskProvider(mode="gps", update_interval_s=0.0)
        with patch.object(provider, "_try_windows_geolocation", return_value=None), patch.object(
            provider, "_try_ip_geolocation", return_value=None
        ), patch.object(provider, "_load_cache"), patch.object(provider, "_save_cache"):
            provider._last_ctx = provider._unavailable("Unavailable")
            ctx = provider.update(10.0)
        self.assertFalse(ctx.available)
        self.assertEqual(ctx.risk_level, 0.0)
        self.assertEqual(ctx.source, "Unavailable")
        self.assertIn("Unavailable", format_location_summary(ctx))


if __name__ == "__main__":
    unittest.main()
