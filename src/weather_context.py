"""
Real-time weather context (Open-Meteo — no API key required).

Uses current GPS/demo coordinates when available. Updates on a throttle
interval (not every camera frame). Safe fallbacks never crash the app.

Rain-expected messaging uses hourly precipitation / probability only —
never claimed from cloudy skies alone.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = PROJECT_ROOT / "config" / "weather_cache.json"

WEATHER_UPDATE_INTERVAL_S = 180.0
WEATHER_CACHE_MAX_AGE_S = 3 * 60 * 60

OPEN_METEO_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude={lat}&longitude={lon}"
    "&current=temperature_2m,precipitation,weather_code,wind_speed_10m,visibility"
    "&hourly=precipitation_probability,precipitation"
    "&forecast_days=1"
    "&timezone=auto"
)

WEATHER_RISK_MAP = {
    "clear": 5.0,
    "clouds": 10.0,
    "drizzle": 35.0,
    "rain": 55.0,
    "heavy_rain": 80.0,
    "snow": 75.0,
    "fog": 85.0,
    "storm": 90.0,
    "windy": 40.0,
    "unknown": 15.0,
}

CONDITION_LABELS = {
    "clear": "Clear",
    "clouds": "Cloudy",
    "drizzle": "Drizzle",
    "rain": "Rain",
    "heavy_rain": "Heavy rain",
    "snow": "Snow",
    "fog": "Fog / low visibility",
    "storm": "Storm",
    "windy": "Windy",
    "unknown": "Unknown",
}


@dataclass(frozen=True)
class EnvironmentalContext:
    weather_condition: str
    temperature: float | None
    visibility: float | None  # metres
    precipitation: float | None
    wind: float | None
    weather_risk: float  # 0..100
    source: str  # api | cache | Unavailable | Disabled
    available: bool
    latitude: float | None = None
    longitude: float | None = None
    last_update_text: str = ""
    rain_expected_soon: bool = False
    rain_expected_text: str = ""  # e.g. "Rain expected soon (~14:00, 70%)"


def weather_condition_label(condition: str) -> str:
    return CONDITION_LABELS.get(condition, condition.replace("_", " ").title())


def wmo_code_to_condition(code: int | None) -> str:
    if code is None:
        return "unknown"
    if code == 0:
        return "clear"
    if code in (1, 2, 3):
        return "clouds"
    if code in (45, 48):
        return "fog"
    if code in (51, 53, 55, 56, 57):
        return "drizzle"
    if code in (61, 63, 66, 80, 81):
        return "rain"
    if code in (65, 67, 82):
        return "heavy_rain"
    if code in (71, 73, 75, 77, 85, 86):
        return "snow"
    if code in (95, 96, 99):
        return "storm"
    return "unknown"


def environmental_risk_from_condition(
    condition: str,
    precipitation: float | None = None,
    visibility: float | None = None,
    wind: float | None = None,
) -> float:
    risk = float(WEATHER_RISK_MAP.get(condition, WEATHER_RISK_MAP["unknown"]))
    if precipitation is not None and precipitation >= 5.0:
        risk = max(risk, WEATHER_RISK_MAP["heavy_rain"])
    elif precipitation is not None and precipitation >= 1.0:
        risk = max(risk, WEATHER_RISK_MAP["rain"])
    if visibility is not None and visibility > 0 and visibility < 1000:
        risk = max(risk, WEATHER_RISK_MAP["fog"])
    if wind is not None and wind >= 50:
        risk = max(risk, WEATHER_RISK_MAP["windy"] + 20.0)
    elif wind is not None and wind >= 35:
        risk = max(risk, WEATHER_RISK_MAP["windy"])
    return float(min(100.0, max(0.0, risk)))


def _parse_rain_expected(hourly: dict | None, max_hours: int = 6) -> tuple[bool, str]:
    """
    Detect upcoming rain from hourly precip probability / amount only.

    Cloudy current conditions alone never trigger this.
    """
    if not isinstance(hourly, dict):
        return False, ""
    times = hourly.get("time") or []
    probs = hourly.get("precipitation_probability") or []
    precips = hourly.get("precipitation") or []
    if not times:
        return False, ""

    # Skip index 0 if it is "now"; scan the next few hours.
    limit = min(len(times), max_hours + 1)
    for i in range(1, limit):
        try:
            t = str(times[i])
            prob = probs[i] if i < len(probs) else None
            precip = precips[i] if i < len(precips) else None
            prob_f = float(prob) if prob is not None else None
            precip_f = float(precip) if precip is not None else None
        except (TypeError, ValueError, IndexError):
            continue

        # Require concrete precip signal — not clouds.
        likely = (prob_f is not None and prob_f >= 50.0) or (
            precip_f is not None and precip_f >= 0.2
        )
        if not likely:
            continue

        # Format time as HH:MM when possible.
        clock = t
        if "T" in t:
            clock = t.split("T", 1)[1][:5]
        if prob_f is not None:
            return True, f"Rain expected soon (~{clock}, {int(prob_f)}%)"
        return True, f"Rain expected soon (~{clock})"

    return False, ""


def format_weather_summary(ctx: EnvironmentalContext) -> str:
    """Compact multi-line text for the existing CONTEXT panel."""
    if not ctx.available or ctx.source in ("Unavailable", "Disabled", "disabled"):
        return "WEATHER:\nUnavailable"
    label = weather_condition_label(ctx.weather_condition)
    temp = f"{ctx.temperature:.0f}°C" if ctx.temperature is not None else "--°C"
    rain = f"{ctx.precipitation:.1f} mm" if ctx.precipitation is not None else "-- mm"
    vis = (
        f"{ctx.visibility / 1000.0:.1f} km"
        if ctx.visibility is not None and ctx.visibility > 0
        else "--"
    )
    updated = ctx.last_update_text or "--"
    lines = [
        f"WEATHER ({ctx.source}):",
        f"{label} | {temp}",
        f"Rain {rain} | Vis {vis}",
        f"Risk {int(round(ctx.weather_risk))} | {updated}",
    ]
    if ctx.rain_expected_soon and ctx.rain_expected_text:
        lines.append(ctx.rain_expected_text)
    return "\n".join(lines)


class WeatherContextProvider:
    """Cached Open-Meteo weather with safe Unavailable fallback."""

    def __init__(
        self,
        enabled: bool = True,
        update_interval_s: float = WEATHER_UPDATE_INTERVAL_S,
        timeout_s: float = 5.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.update_interval_s = float(update_interval_s)
        self.timeout_s = float(timeout_s)
        self.latitude: float | None = None
        self.longitude: float | None = None
        self._last_fetch = 0.0
        self._last_ctx = self._unavailable()
        self._load_cache()

    def configure(self, enabled: bool | None = None) -> None:
        if enabled is not None:
            self.enabled = bool(enabled)

    def set_coordinates(self, latitude: float | None, longitude: float | None) -> None:
        prev = (self.latitude, self.longitude)
        self.latitude = latitude
        self.longitude = longitude
        if latitude is None or longitude is None:
            return
        if prev[0] is None or prev[1] is None:
            self._last_fetch = 0.0
            return
        if abs(float(latitude) - float(prev[0])) > 0.01 or abs(
            float(longitude) - float(prev[1])
        ) > 0.01:
            self._last_fetch = 0.0

    def _unavailable(self, source: str = "Unavailable") -> EnvironmentalContext:
        return EnvironmentalContext(
            weather_condition="unknown",
            temperature=None,
            visibility=None,
            precipitation=None,
            wind=None,
            weather_risk=0.0,
            source=source,
            available=False,
            latitude=self.latitude,
            longitude=self.longitude,
            last_update_text="",
            rain_expected_soon=False,
            rain_expected_text="",
        )

    def _load_cache(self) -> None:
        if not CACHE_PATH.is_file():
            return
        try:
            data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            updated_at = data.get("updated_at")
            cache_fresh = False
            if updated_at is not None:
                try:
                    cache_fresh = max(0.0, time.time() - float(updated_at)) <= WEATHER_CACHE_MAX_AGE_S
                except (TypeError, ValueError):
                    cache_fresh = False
            self._last_ctx = EnvironmentalContext(
                weather_condition=str(data.get("weather_condition", "unknown")) if cache_fresh else "unknown",
                temperature=data.get("temperature"),
                visibility=data.get("visibility"),
                precipitation=data.get("precipitation"),
                wind=data.get("wind"),
                weather_risk=float(data.get("weather_risk", 0.0)) if cache_fresh else 0.0,
                source="cache",
                available=True,
                latitude=data.get("latitude"),
                longitude=data.get("longitude"),
                last_update_text=str(data.get("last_update_text", "cached")),
                rain_expected_soon=bool(data.get("rain_expected_soon", False)),
                rain_expected_text=str(data.get("rain_expected_text", "")),
            )
        except Exception:
            pass

    def _save_cache(self, ctx: EnvironmentalContext) -> None:
        try:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            CACHE_PATH.write_text(
                json.dumps(
                    {
                        "weather_condition": ctx.weather_condition,
                        "temperature": ctx.temperature,
                        "visibility": ctx.visibility,
                        "precipitation": ctx.precipitation,
                        "wind": ctx.wind,
                        "weather_risk": ctx.weather_risk,
                        "latitude": ctx.latitude,
                        "longitude": ctx.longitude,
                        "last_update_text": ctx.last_update_text,
                        "rain_expected_soon": ctx.rain_expected_soon,
                        "rain_expected_text": ctx.rain_expected_text,
                        "updated_at": time.time(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _fetch_api(self, lat: float, lon: float) -> EnvironmentalContext | None:
        url = OPEN_METEO_URL.format(lat=lat, lon=lon)
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "DriverAttentionMonitor/1.0"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            current = payload.get("current") or {}
            code = current.get("weather_code", current.get("weathercode"))
            condition = wmo_code_to_condition(int(code) if code is not None else None)
            temperature = current.get("temperature_2m", current.get("temperature"))
            precipitation = current.get("precipitation", current.get("rain"))
            wind = current.get("wind_speed_10m", current.get("windspeed"))
            visibility = current.get("visibility")
            rain_soon, rain_text = _parse_rain_expected(payload.get("hourly"))
            # Never invent rain-soon from cloudy current skies alone.
            if condition == "clouds" and not rain_soon:
                rain_text = ""
            risk = environmental_risk_from_condition(
                condition,
                precipitation=float(precipitation) if precipitation is not None else None,
                visibility=float(visibility) if visibility is not None else None,
                wind=float(wind) if wind is not None else None,
            )
            now_txt = datetime.now().strftime("%H:%M")
            return EnvironmentalContext(
                weather_condition=condition,
                temperature=float(temperature) if temperature is not None else None,
                visibility=float(visibility) if visibility is not None else None,
                precipitation=float(precipitation) if precipitation is not None else None,
                wind=float(wind) if wind is not None else None,
                weather_risk=risk,
                source="api",
                available=True,
                latitude=lat,
                longitude=lon,
                last_update_text=f"upd {now_txt}",
                rain_expected_soon=rain_soon,
                rain_expected_text=rain_text,
            )
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, TypeError, OSError) as exc:
            print(f"WEATHER: API unavailable ({exc})")
            return None
        except Exception as exc:
            print(f"WEATHER: unexpected error ({exc})")
            return None

    def update(self, now_s: float) -> EnvironmentalContext:
        if not self.enabled:
            return self._unavailable("Disabled")

        if self._last_fetch > 0 and (now_s - self._last_fetch) < self.update_interval_s:
            return self._last_ctx

        self._last_fetch = now_s
        lat = self.latitude
        lon = self.longitude
        if lat is None or lon is None:
            if self._last_ctx.available:
                return self._last_ctx
            return self._unavailable("Unavailable")

        fetched = self._fetch_api(float(lat), float(lon))
        if fetched is not None:
            self._last_ctx = fetched
            self._save_cache(fetched)
            return self._last_ctx

        if self._last_ctx.available and self._last_ctx.source in ("api", "cache"):
            self._last_ctx = EnvironmentalContext(
                weather_condition=self._last_ctx.weather_condition,
                temperature=self._last_ctx.temperature,
                visibility=self._last_ctx.visibility,
                precipitation=self._last_ctx.precipitation,
                wind=self._last_ctx.wind,
                weather_risk=self._last_ctx.weather_risk,
                source="cache",
                available=True,
                latitude=float(lat),
                longitude=float(lon),
                last_update_text=self._last_ctx.last_update_text or "cached",
                rain_expected_soon=self._last_ctx.rain_expected_soon,
                rain_expected_text=self._last_ctx.rain_expected_text,
            )
            return self._last_ctx

        self._last_ctx = self._unavailable("Unavailable")
        return self._last_ctx
