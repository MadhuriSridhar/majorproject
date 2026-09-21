"""
GPS / location context for situational risk.

Location Mode:
  - gps / automatic: real GPS (Windows) then IP geolocation; Unavailable if none
    - carla_sim: virtual zones classified from CARLA world x/y coordinates
    - demo: forced zone presets (clearly labeled DEMO)
  - disabled: no location contribution

Never crashes the app when GPS/network is unavailable.
Updates on a throttle interval (not every camera frame).
"""

from __future__ import annotations

import json
import math
import time
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
    "accident": ("Accident / High-Risk Zone", "accident", 85.0),
    "construction": ("Construction Zone", "construction", 55.0),
    "pedestrian": ("Pedestrian Zone", "pedestrian", 65.0),
    "highway": ("Highway", "highway", 40.0),
    "sharp_turn": ("Sharp Turn / High-Risk Zone", "sharp_turn", 65.0),
    "other": ("Other Road", "other", 25.0),
}

DEMO_ZONE_LABELS = {
    "normal": "Normal Road",
    "school": "School Zone",
    "accident": "Accident / High-Risk Zone",
    "construction": "Construction Zone",
    "pedestrian": "Pedestrian Zone",
    "highway": "Highway",
    "sharp_turn": "Sharp Turn / High-Risk Zone",
    "other": "Other Road",
}

# Default coords used only for DEMO weather lookup (Bangalore).
DEMO_LAT = 12.9716
DEMO_LON = 77.5946

LOCATION_UPDATE_INTERVAL_S = 30.0
LOCATION_CACHE_MAX_AGE_S = 30 * 60

# Baseline risk used when a real location is resolved but no configured
# geofence matches. This is explicitly NOT a claim that the road is verified
# safe — it only means no high-risk zone has been configured at these
# coordinates.
BASELINE_LOCATION_RISK = 5.0
NO_ZONE_NAME = "No configured high-risk zone"

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


@dataclass(frozen=True)
class GeofenceZone:
    """A configurable real-world geofence record (lat/lon + radius)."""

    name: str
    zone_type: str  # school | highway | construction | sharp_turn | other
    latitude: float
    longitude: float
    radius_m: float
    risk_level: float  # 0..100


@dataclass(frozen=True)
class CarlaZone:
    """A virtual CARLA-world zone bounded by simulated x/y coordinates."""

    name: str
    zone_type: str
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    risk_level: float

    def contains(self, x: float, y: float) -> bool:
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y


# These are virtual zones, not real-world semantic data supplied by CARLA.
# Projects can replace the bounds for their loaded map without changing the
# classifier or its Normal Road fallback.
CARLA_SIM_ZONES: tuple[CarlaZone, ...] = (
    CarlaZone("School Zone", "school", -170.0, -35.0, -80.0, 30.0, 70.0),
    CarlaZone("Accident / High-Risk Zone", "accident", 35.0, 170.0, -105.0, -25.0, 85.0),
    CarlaZone("Construction Zone", "construction", 40.0, 160.0, 25.0, 95.0, 55.0),
    CarlaZone("Pedestrian Zone", "pedestrian", -170.0, -35.0, 25.0, 95.0, 65.0),
    CarlaZone("Highway", "highway", 50.0, 170.0, -170.0, -60.0, 40.0),
    CarlaZone("Sharp Turn / High-Risk Zone", "sharp_turn", -60.0, 30.0, -170.0, -80.0, 65.0),
)

_CANONICAL_ZONE_ALIASES: dict[str, str] = {
    "school": "school",
    "school zone": "school",
    "accident": "accident",
    "accident zone": "accident",
    "accident / high-risk zone": "accident",
    "construction": "construction",
    "construction zone": "construction",
    "pedestrian": "pedestrian",
    "pedestrian zone": "pedestrian",
    "highway": "highway",
    "sharp_turn": "sharp_turn",
    "sharp turn": "sharp_turn",
    "sharp turn / high-risk zone": "sharp_turn",
    "other": "other",
    "normal road": "none",
    "none": "none",
}


def classify_carla_zone(
    x: float | None,
    y: float | None,
    zones: tuple[CarlaZone, ...] | list[CarlaZone] | None = None,
) -> tuple[str, str, float]:
    """Classify virtual geography from the vehicle's current CARLA x/y."""

    try:
        world_x = float(x)
        world_y = float(y)
    except (TypeError, ValueError):
        return "none", "Normal Road", BASELINE_LOCATION_RISK

    # Other Road is the explicit CARLA fallback, not a competing catch-all
    # geofence. Configured semantic zones below therefore always win.

    for zone in zones if zones is not None else CARLA_SIM_ZONES:
        if zone.contains(world_x, world_y):
            return zone.zone_type, zone.name, float(zone.risk_level)
    return "other", "Other Road", 25.0


def carla_sim_location_context(
    x: float | None,
    y: float | None,
    z: float | None = None,
    zones: tuple[CarlaZone, ...] | list[CarlaZone] | None = None,
) -> LocationContext:
    """Build an available location context from simulated CARLA coordinates."""

    zone_type, zone_name, risk = classify_carla_zone(x, y, zones)
    return LocationContext(
        latitude=float(x) if x is not None else None,
        longitude=float(y) if y is not None else None,
        zone_type=zone_type,
        zone_name=zone_name,
        risk_level=risk,
        source="CARLA_SIM",
        place_name="CARLA simulated world",
        available=x is not None and y is not None,
        mode="carla_sim",
    )


# Optional real-world geofence dataset (user-configurable, kept empty by
# default so the app never pretends arbitrary coordinates are real zones).
GEOFENCES_FILE = PROJECT_ROOT / "config" / "geofences.json"


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def load_geofences(path: Path | None = None) -> list[GeofenceZone]:
    """Load the configurable geofence dataset (empty on any failure)."""
    target = Path(path) if path else GEOFENCES_FILE
    if not target.is_file():
        return []
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return []
    zones: list[GeofenceZone] = []
    for raw in data.get("zones", []) if isinstance(data, dict) else []:
        try:
            zones.append(
                GeofenceZone(
                    name=str(raw.get("name", "Zone")),
                    zone_type=str(raw.get("zone_type", "other")),
                    latitude=float(raw["latitude"]),
                    longitude=float(raw["longitude"]),
                    radius_m=float(raw.get("radius_m", 500.0)),
                    risk_level=float(raw.get("risk_level", 50.0)),
                )
            )
        except (TypeError, ValueError, KeyError):
            continue
    return zones


def classify_geofence(
    lat: float, lon: float, zones: list[GeofenceZone] | None = None
) -> tuple[str, str, float] | None:
    """Return (zone_type, zone_name, risk_level) for the first matching zone."""
    if lat is None or lon is None:
        return None
    for zone in zones or []:
        try:
            if haversine_m(float(lat), float(lon), zone.latitude, zone.longitude) <= zone.radius_m:
                return (zone.zone_type, zone.name, zone.risk_level)
        except (TypeError, ValueError):
            continue
    return None


@dataclass(frozen=True)
class ZoneTransition:
    """One geographic-context episode boundary (enter / exit / change)."""

    event: str  # enter | exit | change
    zone_type: str
    zone_name: str
    from_zone_name: str = ""
    to_zone_name: str = ""
    timestamp_s: float = 0.0


class ZoneTransitionTracker:
    """Track transitions between "no configured zone" and configured zones.

    Staying inside the same zone is treated as one geographic-context episode,
    so no repeated entry events are emitted. Entering, leaving, or switching
    between two different zones each produce exactly one transition event.
    """

    NO_ZONE = "none"
    KNOWN_ZONE_TYPES = (
        "school",
        "accident",
        "construction",
        "pedestrian",
        "highway",
        "sharp_turn",
        "other",
    )

    def __init__(self) -> None:
        self._zone_type = self.NO_ZONE
        self._zone_name = ""

    def reset(self) -> None:
        """Reset to the 'no configured zone' baseline (new driving session)."""
        self._zone_type = self.NO_ZONE
        self._zone_name = ""

    @classmethod
    def classify(cls, zone_type: str | None) -> str:
        value = str(zone_type or "none").strip().lower()
        canonical = _CANONICAL_ZONE_ALIASES.get(value, value)
        return canonical if canonical in cls.KNOWN_ZONE_TYPES else cls.NO_ZONE

    def update(self, zone_type, zone_name, now_s: float = 0.0) -> ZoneTransition | None:
        """Observe the current zone; return one ZoneTransition on a boundary."""
        zt = self.classify(zone_type)
        name = str(zone_name or "")
        if zt == self._zone_type:
            self._zone_name = name
            return None
        prev_type = self._zone_type
        prev_name = self._zone_name
        self._zone_type = zt
        self._zone_name = name
        if prev_type == self.NO_ZONE and zt != self.NO_ZONE:
            return ZoneTransition("enter", zt, name, timestamp_s=now_s)
        if prev_type != self.NO_ZONE and zt == self.NO_ZONE:
            return ZoneTransition("exit", prev_type, prev_name, timestamp_s=now_s)
        return ZoneTransition(
            "change",
            zt,
            name,
            from_zone_name=prev_name,
            to_zone_name=name,
            timestamp_s=now_s,
        )


def _zone_display_name(zone_type: str, zone_name: str) -> str:
    return zone_name or str(zone_type).replace("_", " ").title()


def format_zone_event_summary(transition: ZoneTransition | None) -> str:
    """Compact one-line event text ('' when no transition happened)."""
    if transition is None:
        return ""
    if transition.event == "enter":
        return f"Entered {_zone_display_name(transition.zone_type, transition.zone_name or transition.zone_type)}"
    if transition.event == "exit":
        return f"Exited {_zone_display_name(transition.zone_type, transition.zone_name or transition.zone_type)}"
    from_label = transition.from_zone_name or "other zone"
    return (
        f"Changed zone: {from_label} → "
        f"{_zone_display_name(transition.zone_type, transition.to_zone_name or transition.zone_name)}"
    )


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
    """Compact multi-line text for the existing CONTEXT panel.

    Honest about zones: when no configured geofence matches a real location,
    it reports "No configured high-risk zone" with a baseline risk instead of
    implying the road is verified safe.
    """
    if ctx.mode == "disabled" or ctx.source == "Disabled":
        return "REAL LOCATION:\nDisabled"
    if not ctx.available:
        return "REAL LOCATION:\nUnavailable"
    is_demo = ctx.mode == "demo" or ctx.source == "DEMO"
    src = "DEMO" if is_demo else ctx.source
    in_zone = ctx.zone_type not in ("none", "", "unknown")
    risk = int(round(ctx.risk_level))
    if in_zone:
        zone = ctx.zone_name or ctx.zone_type.replace("_", " ").title()
        marker = "HIGH RISK" if risk >= 40 else "ZONE"
        risk_line = f"Risk {risk} ({marker})"
    elif is_demo:
        zone = ctx.zone_name or "Normal Road"
        risk_line = f"Risk {risk}"
    else:
        zone = NO_ZONE_NAME
        risk_line = "Location risk: Baseline"
    place = ctx.place_name or ""
    lines = [f"REAL LOCATION ({src}):", zone, risk_line]
    if place:
        lines.append(place)
    return "\n".join(lines)


class LocationRiskProvider:
    """Throttled location provider with GPS / CARLA_SIM / Demo modes."""

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
        self._geofences = load_geofences()
        self._load_cache()

    def update_carla(
        self,
        x: float | None,
        y: float | None,
        z: float | None = None,
    ) -> LocationContext:
        """Update from CARLA world coordinates; elapsed time never changes zones."""

        self.mode = "carla_sim"
        self._last_ctx = carla_sim_location_context(x, y, z)
        return self._last_ctx

    def reload_geofences(self) -> None:
        """Re-read the configurable geofence dataset at runtime."""
        self._geofences = load_geofences()

    def _context_for_coordinates(
        self, lat: float, lon: float, source: str, place_name: str = ""
    ) -> LocationContext:
        """Build a GPS/IP context, applying any matching configured geofence."""
        hit = classify_geofence(lat, lon, self._geofences)
        if hit:
            zone_type, zone_name, risk = hit
        else:
            zone_type, zone_name, risk = "none", NO_ZONE_NAME, BASELINE_LOCATION_RISK
        return LocationContext(
            latitude=float(lat),
            longitude=float(lon),
            zone_type=zone_type,
            zone_name=zone_name,
            risk_level=float(risk),
            source=source,
            place_name=place_name,
            available=True,
            mode="gps",
        )

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
            updated_at = data.get("updated_at")
            cache_fresh = False
            if updated_at is not None:
                try:
                    cache_fresh = max(0.0, time.time() - float(updated_at)) <= LOCATION_CACHE_MAX_AGE_S
                except (TypeError, ValueError):
                    cache_fresh = False
            self._last_ctx = LocationContext(
                latitude=lat,
                longitude=lon,
                zone_type=str(data.get("zone_type", "none")) if cache_fresh else "none",
                zone_name=str(data.get("zone_name", NO_ZONE_NAME)) if cache_fresh else NO_ZONE_NAME,
                risk_level=float(data.get("risk_level", 0.0)) if cache_fresh else BASELINE_LOCATION_RISK,
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
                        "updated_at": time.time(),
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

        if mode in ("carla_sim", "carla"):
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
            ctx = self._context_for_coordinates(lat, lon, "gps", "GPS fix")
            self._save_cache(ctx)
            self._last_ctx = ctx
            return ctx

        ip = self._try_ip_geolocation()
        if ip is not None:
            lat, lon, place = ip
            ctx = self._context_for_coordinates(lat, lon, "ip", place or "Network location")
            self._save_cache(ctx)
            self._last_ctx = ctx
            return ctx

        if self._last_ctx.available and self._last_ctx.source in ("gps", "ip", "cache"):
            self._last_ctx = LocationContext(
                latitude=self._last_ctx.latitude,
                longitude=self._last_ctx.longitude,
                zone_type=self._last_ctx.zone_type,
                zone_name=self._last_ctx.zone_name or NO_ZONE_NAME,
                risk_level=self._last_ctx.risk_level,
                source="cache",
                place_name=self._last_ctx.place_name,
                available=True,
                mode="gps",
            )
            return self._last_ctx

        self._last_ctx = self._unavailable("Unavailable", mode="gps")
        return self._last_ctx
