"""
GPS / location context for situational risk.

Location Mode:
  - gps / automatic: real GPS (Windows) then IP geolocation; Unavailable if none
  - demo: forced zone presets (clearly labeled DEMO)
  - disabled: no location contribution

Never crashes the app when GPS/network is unavailable.
Updates on a throttle interval (not every camera frame).
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCATION_CACHE_PATH = PROJECT_ROOT / "config" / "location_cache.json"

# Demo zone presets (explicit DEMO mode).
DEMO_ZONE_PRESETS: dict[str, tuple[str, str, float]] = {
    # key -> (display_name, zone_type, risk_level 0..100)
    "normal": ("Normal Road", "none", 5.0),
    "school": ("School Zone", "school", 70.0),
    "highway": ("Highway", "highway", 40.0),
    "construction": ("Construction Zone", "construction", 55.0),
    "sharp_turn": ("Sharp Turn / High-Risk Zone", "sharp_turn", 65.0),
}

DEMO_ZONE_LABELS = {
    "normal": "Normal Road",
    "school": "School Zone",
    "highway": "Highway",
    "construction": "Construction Zone",
    "sharp_turn": "Sharp Turn / High-Risk Zone",
}

# Default coords used only for DEMO weather lookup (Bangalore).
DEMO_LAT = 12.9716
DEMO_LON = 77.5946

LOCATION_UPDATE_INTERVAL_S = 30.0

IP_GEO_URLS = (
    "http://ip-api.com/json/?fields=status,message,lat,lon,city,regionName,country,query",
    "https://ipapi.co/json/",
)


@dataclass(frozen=True)
class LocationContext:
    latitude: float | None
    longitude: float | None
    zone_type: str  # none | school | highway | construction | sharp_turn | other
    zone_name: str
    risk_level: float  # 0..100
    source: str  # gps | ip | DEMO | cache | Unavailable | Disabled
    place_name: str = ""
    available: bool = False
    mode: str = "gps"  # gps | demo | disabled


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _http_json(url: str, timeout_s: float = 4.0) -> dict | None:
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "DriverAttentionMonitor/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError, json.JSONDecodeError):
        return None
    except Exception:
        return None


def format_location_summary(ctx: LocationContext) -> str:
    """Compact multi-line text for the existing CONTEXT panel."""
    if ctx.mode == "disabled" or ctx.source == "Disabled":
        return "LOCATION:\nDisabled"
    if not ctx.available:
        return "LOCATION:\nUnavailable"
    zone = ctx.zone_name or "Normal Road"
    src = ctx.source
    if ctx.mode == "demo" or src == "DEMO":
        src = "DEMO"
    risk = int(round(ctx.risk_level))
    place = ctx.place_name or ""
    if place:
        return f"LOCATION ({src}):\n{zone}\nRisk {risk}\n{place}"
    return f"LOCATION ({src}):\n{zone}\nRisk {risk}"


class LocationRiskProvider:
    """Throttled location provider with GPS / Demo / Disabled modes."""

    def __init__(
        self,
        mode: str = "gps",
        demo_zone: str = "normal",
        update_interval_s: float = LOCATION_UPDATE_INTERVAL_S,
        timeout_s: float = 4.0,
    ) -> None:
        self.mode = str(mode or "gps").lower()
        self.demo_zone = str(demo_zone or "normal")
        self.update_interval_s = float(update_interval_s)
        self.timeout_s = float(timeout_s)
        self._last_update = 0.0
        self._last_ctx = self._unavailable("Unavailable")
        self._load_cache()

    def configure(self, mode: str | None = None, demo_zone: str | None = None) -> None:
        if mode is not None:
            new_mode = str(mode).lower()
            if new_mode != self.mode:
                self._last_update = 0.0
            self.mode = new_mode
        if demo_zone is not None:
            if demo_zone != self.demo_zone:
                self._last_update = 0.0
            self.demo_zone = str(demo_zone)

    @staticmethod
    def _unavailable(source: str, mode: str = "gps") -> LocationContext:
        return LocationContext(
            latitude=None,
            longitude=None,
            zone_type="none",
            zone_name="",
            risk_level=0.0,
            source=source,
            available=False,
            mode=mode,
        )

    def _load_cache(self) -> None:
        if not LOCATION_CACHE_PATH.is_file():
            return
        try:
            data = json.loads(LOCATION_CACHE_PATH.read_text(encoding="utf-8"))
            lat = float(data["latitude"])
            lon = float(data["longitude"])
            self._last_ctx = LocationContext(
                latitude=lat,
                longitude=lon,
                zone_type=str(data.get("zone_type", "none")),
                zone_name=str(data.get("zone_name", "Normal Road")),
                risk_level=float(data.get("risk_level", 0.0)),
                source="cache",
                place_name=str(data.get("place_name", "")),
                available=True,
                mode="gps",
            )
        except Exception:
            pass

    def _save_cache(self, ctx: LocationContext) -> None:
        if not ctx.available or ctx.latitude is None or ctx.longitude is None:
            return
        try:
            LOCATION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            LOCATION_CACHE_PATH.write_text(
                json.dumps(
                    {
                        "latitude": ctx.latitude,
                        "longitude": ctx.longitude,
                        "place_name": ctx.place_name,
                        "zone_type": ctx.zone_type,
                        "zone_name": ctx.zone_name,
                        "risk_level": ctx.risk_level,
                        "source": ctx.source,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _demo_context(self) -> LocationContext:
        key = self.demo_zone if self.demo_zone in DEMO_ZONE_PRESETS else "normal"
        name, zone_type, risk = DEMO_ZONE_PRESETS[key]
        return LocationContext(
            latitude=DEMO_LAT,
            longitude=DEMO_LON,
            zone_type=zone_type,
            zone_name=name,
            risk_level=float(risk),
            source="DEMO",
            place_name="DEMO location",
            available=True,
            mode="demo",
        )

    def _try_windows_geolocation(self) -> tuple[float, float, float | None] | None:
        try:
            import asyncio

            from winrt.windows.devices.geolocation import (  # type: ignore
                Geolocator,
                GeolocationAccessStatus,
            )
        except Exception:
            return None

        async def _read():
            access = await Geolocator.request_access_async()
            if access != GeolocationAccessStatus.ALLOWED:
                return None
            locator = Geolocator()
            pos = await locator.get_geoposition_async()
            coord = pos.coordinate.point.position
            acc = getattr(pos.coordinate, "accuracy", None)
            return float(coord.latitude), float(coord.longitude), float(acc) if acc else None

        try:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    return None
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            result = loop.run_until_complete(_read())
            if result is None:
                return None
            return result
        except Exception as exc:
            print(f"LOCATION: Windows GPS unavailable ({exc})")
            return None

    def _try_ip_geolocation(self) -> tuple[float, float, str] | None:
        for url in IP_GEO_URLS:
            data = _http_json(url, timeout_s=self.timeout_s)
            if not isinstance(data, dict):
                continue
            try:
                if data.get("status") == "success":
                    lat = float(data["lat"])
                    lon = float(data["lon"])
                    parts = [data.get("city"), data.get("regionName"), data.get("country")]
                    place = ", ".join(p for p in parts if p)
                    return lat, lon, place
                if "latitude" in data and "longitude" in data:
                    lat = float(data["latitude"])
                    lon = float(data["longitude"])
                    parts = [data.get("city"), data.get("region"), data.get("country_name")]
                    place = ", ".join(p for p in parts if p)
                    return lat, lon, place
            except (TypeError, ValueError, KeyError):
                continue
        return None

    def update(self, now_s: float) -> LocationContext:
        mode = self.mode
        if mode in ("disabled", "off", "none"):
            self._last_ctx = self._unavailable("Disabled", mode="disabled")
            return self._last_ctx

        if mode == "demo":
            # Demo can refresh immediately when zone changes (configure resets timer).
            if self._last_update > 0 and (now_s - self._last_update) < self.update_interval_s:
                if self._last_ctx.mode == "demo" and self._last_ctx.available:
                    return self._last_ctx
            self._last_update = now_s
            self._last_ctx = self._demo_context()
            return self._last_ctx

        # GPS / Automatic
        if self._last_update > 0 and (now_s - self._last_update) < self.update_interval_s:
            return self._last_ctx

        self._last_update = now_s
        win = self._try_windows_geolocation()
        if win is not None:
            lat, lon, _acc = win
            ctx = LocationContext(
                latitude=lat,
                longitude=lon,
                zone_type="none",
                zone_name="Normal Road",
                risk_level=5.0,
                source="gps",
                place_name="GPS fix",
                available=True,
                mode="gps",
            )
            self._save_cache(ctx)
            self._last_ctx = ctx
            return ctx

        ip = self._try_ip_geolocation()
        if ip is not None:
            lat, lon, place = ip
            ctx = LocationContext(
                latitude=lat,
                longitude=lon,
                zone_type="none",
                zone_name="Normal Road",
                risk_level=5.0,
                source="ip",
                place_name=place or "Network location",
                available=True,
                mode="gps",
            )
            self._save_cache(ctx)
            self._last_ctx = ctx
            return ctx

        if self._last_ctx.available and self._last_ctx.source in ("gps", "ip", "cache"):
            self._last_ctx = LocationContext(
                latitude=self._last_ctx.latitude,
                longitude=self._last_ctx.longitude,
                zone_type=self._last_ctx.zone_type,
                zone_name=self._last_ctx.zone_name or "Normal Road",
                risk_level=self._last_ctx.risk_level,
                source="cache",
                place_name=self._last_ctx.place_name,
                available=True,
                mode="gps",
            )
            return self._last_ctx

        self._last_ctx = self._unavailable("Unavailable", mode="gps")
        return self._last_ctx
