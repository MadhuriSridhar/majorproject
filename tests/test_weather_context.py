"""
Focused unit tests for weather context with mocked API responses.

No real network requests.
"""

from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from weather_context import (  # noqa: E402
    WeatherContextProvider,
    _parse_rain_expected,
    environmental_risk_from_condition,
    format_weather_summary,
    wmo_code_to_condition,
)


def _mock_urlopen(payload: dict):
    raw = json.dumps(payload).encode("utf-8")

    class _Resp:
        def read(self):
            return raw

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    return MagicMock(return_value=_Resp())


class WeatherConditionRiskTests(unittest.TestCase):
    def test_clear_weather(self) -> None:
        self.assertEqual(wmo_code_to_condition(0), "clear")
        risk = environmental_risk_from_condition("clear")
        self.assertLess(risk, 20.0)
        self.assertGreaterEqual(risk, 0.0)

    def test_rain(self) -> None:
        self.assertEqual(wmo_code_to_condition(61), "rain")
        risk = environmental_risk_from_condition("rain", precipitation=1.5)
        self.assertGreaterEqual(risk, 50.0)
        self.assertLessEqual(risk, 100.0)

    def test_heavy_rain(self) -> None:
        self.assertEqual(wmo_code_to_condition(65), "heavy_rain")
        risk = environmental_risk_from_condition("heavy_rain", precipitation=6.0)
        self.assertGreaterEqual(risk, 75.0)
        self.assertLessEqual(risk, 100.0)

    def test_poor_visibility(self) -> None:
        risk = environmental_risk_from_condition("clouds", visibility=400.0)
        self.assertGreaterEqual(risk, 80.0)
        self.assertLessEqual(risk, 100.0)

    def test_weather_risk_bounded(self) -> None:
        risk = environmental_risk_from_condition(
            "storm", precipitation=20.0, visibility=100.0, wind=80.0
        )
        self.assertGreaterEqual(risk, 0.0)
        self.assertLessEqual(risk, 100.0)


class RainForecastTests(unittest.TestCase):
    def test_forecast_rain_expected_soon(self) -> None:
        hourly = {
            "time": ["2026-01-01T12:00", "2026-01-01T13:00", "2026-01-01T14:00"],
            "precipitation_probability": [10, 70, 80],
            "precipitation": [0.0, 0.5, 1.0],
        }
        soon, text = _parse_rain_expected(hourly)
        self.assertTrue(soon)
        self.assertIn("Rain expected soon", text)

    def test_cloudy_alone_does_not_predict_rain(self) -> None:
        # Cloudy WMO codes must not invent rain-soon without precip signal.
        hourly = {
            "time": ["2026-01-01T12:00", "2026-01-01T13:00"],
            "precipitation_probability": [5, 10],
            "precipitation": [0.0, 0.0],
        }
        soon, text = _parse_rain_expected(hourly)
        self.assertFalse(soon)
        self.assertEqual(text, "")
        self.assertEqual(wmo_code_to_condition(3), "clouds")


class WeatherProviderMockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = WeatherContextProvider(enabled=True, update_interval_s=0.0)
        self.provider._last_ctx = self.provider._unavailable()
        self.provider.set_coordinates(12.97, 77.59)

    def test_api_success_clear(self) -> None:
        payload = {
            "current": {
                "weather_code": 0,
                "temperature_2m": 28.0,
                "precipitation": 0.0,
                "wind_speed_10m": 5.0,
                "visibility": 10000.0,
            },
            "hourly": {
                "time": ["2026-01-01T12:00", "2026-01-01T13:00"],
                "precipitation_probability": [0, 5],
                "precipitation": [0.0, 0.0],
            },
        }
        with patch("weather_context.urllib.request.urlopen", _mock_urlopen(payload)), patch.object(
            self.provider, "_save_cache"
        ):
            ctx = self.provider.update(1.0)
        self.assertTrue(ctx.available)
        self.assertEqual(ctx.weather_condition, "clear")
        self.assertFalse(ctx.rain_expected_soon)

    def test_api_network_failure_does_not_crash(self) -> None:
        with patch(
            "weather_context.urllib.request.urlopen",
            side_effect=OSError("network down"),
        ), patch.object(self.provider, "_save_cache"):
            ctx = self.provider.update(1.0)
        self.assertFalse(ctx.available)
        self.assertEqual(ctx.weather_risk, 0.0)
        self.assertIn("Unavailable", format_weather_summary(ctx))

    def test_stale_cache_used_safely_after_failure(self) -> None:
        # Seed a prior good reading, then fail the next fetch.
        self.provider._last_ctx = self.provider._unavailable()
        good = self.provider._unavailable()
        # Manually install a cache-like available context.
        from weather_context import EnvironmentalContext

        self.provider._last_ctx = EnvironmentalContext(
            weather_condition="rain",
            temperature=22.0,
            visibility=5000.0,
            precipitation=1.2,
            wind=10.0,
            weather_risk=55.0,
            source="api",
            available=True,
            latitude=12.97,
            longitude=77.59,
            last_update_text="upd 12:00",
            rain_expected_soon=False,
            rain_expected_text="",
        )
        with patch(
            "weather_context.urllib.request.urlopen",
            side_effect=OSError("timeout"),
        ), patch.object(self.provider, "_save_cache"):
            ctx = self.provider.update(5.0)
        self.assertTrue(ctx.available)
        self.assertEqual(ctx.source, "cache")
        self.assertEqual(ctx.weather_condition, "rain")

    def test_no_coordinates_unavailable(self) -> None:
        provider = WeatherContextProvider(enabled=True, update_interval_s=0.0)
        provider._last_ctx = provider._unavailable()
        provider.set_coordinates(None, None)
        ctx = provider.update(1.0)
        self.assertFalse(ctx.available)
        self.assertIn("Unavailable", format_weather_summary(ctx))


if __name__ == "__main__":
    unittest.main()
