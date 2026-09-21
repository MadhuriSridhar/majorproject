"""
Populate config/geofences.json from OpenStreetMap via the Overpass API.

This is a helper for the user to build a REAL, mapped high-risk-zone dataset.
It never fabricates zones: it only copies features already mapped in OSM for a
bounding box you provide on the command line.

Categories (all data-driven from OSM — no invented coordinates):
  school        amenity = school / kindergarten / college / university
  construction  landuse = construction / highway = construction

Important honesty limits:
  * OSM coverage varies by region. A feature being absent means it is not
    mapped, NOT that no hazard exists.
  * These are mapped geographic features, not authoritative road-safety or
    "accident-prone" data.
  * "sharp_turn" / "highway" zones are NOT generated here because OSM does not
    reliably expose per-road curve geometry as a simple point feature. Add
    those manually with coordinates you verify yourself.

Usage:
  python tools/build_geofences.py --south 12.8 --west 77.4 --north 13.1 --east 77.8
  python tools/build_geofences.py --south 12.8 --west 77.4 --north 13.1 --east 77.8 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_ROOT / "config" / "geofences.json"

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "DriverAttentionMonitor/1.0 (geofence population helper)"

# key -> (OSM tag query fragments, zone_type, default_radius_m, default_risk_level)
CATEGORIES: dict[str, tuple[str, str, float, float]] = {
    "school": (
        '"amenity"="school";"amenity"="kindergarten";"amenity"="college";"amenity"="university";',
        "school",
        300.0,
        70.0,
    ),
    "construction": (
        '"landuse"="construction";"highway"="construction";',
        "construction",
        500.0,
        55.0,
    ),
}


def _center_of(element: dict) -> tuple[float, float] | None:
    """Best-effort lat/lon for a node, or the computed center of a way/relation."""
    center = element.get("center")
    if isinstance(center, dict):
        try:
            return float(center["lat"]), float(center["lon"])
        except (KeyError, TypeError, ValueError):
            pass
    try:
        return float(element["lat"]), float(element["lon"])
    except (KeyError, TypeError, ValueError):
        return None


def _name_of(element: dict, fallback: str) -> str:
    tags = element.get("tags") or {}
    name = tags.get("name") or tags.get("ref")
    return str(name).strip() if name else fallback


def run_overpass(query: str, timeout_s: float = 90.0) -> dict | None:
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    req = urllib.request.Request(OVERPASS_URL, data=data, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_zones(south: float, west: float, north: float, east: float) -> list[dict]:
    bbox = f"{south},{west},{north},{east}"
    zones: list[dict] = []
    for category, (tags_q, zone_type, radius, risk) in CATEGORIES.items():
        query = f"[out:json][timeout:60];(nwr[{tags_q}]({bbox}););out center;"
        print(f"Querying Overpass for {category} within bbox {bbox} ...")
        try:
            payload = run_overpass(query)
        except Exception as exc:  # noqa: BLE001 - fail safe, keep going
            print(f"  !! Overpass query failed ({exc}); skipping {category}.")
            continue
        elements = (payload or {}).get("elements", [])
        count = 0
        for element in elements:
            coords = _center_of(element)
            if coords is None:
                continue
            lat, lon = coords
            name = _name_of(element, f"{category.title()} zone")
            zones.append(
                {
                    "name": name,
                    "zone_type": zone_type,
                    "latitude": round(lat, 6),
                    "longitude": round(lon, 6),
                    "radius_m": radius,
                    "risk_level": risk,
                }
            )
            count += 1
        print(f"  {category}: {count} zone(s)")
        time.sleep(2.0)  # polite spacing between category queries
    return zones


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build config/geofences.json from OSM/Overpass data."
    )
    parser.add_argument("--south", type=float, required=True, help="min latitude")
    parser.add_argument("--west", type=float, required=True, help="min longitude")
    parser.add_argument("--north", type=float, required=True, help="max latitude")
    parser.add_argument("--east", type=float, required=True, help="max longitude")
    parser.add_argument("--output", type=str, default=str(OUT_PATH))
    parser.add_argument("--dry-run", action="store_true", help="print JSON without writing")
    args = parser.parse_args(argv)

    if not (args.south < args.north and args.west < args.east):
        print("Invalid bounding box (expect south < north and west < east).", file=sys.stderr)
        return 2

    zones = build_zones(args.south, args.west, args.north, args.east)
    document = {"zones": zones}

    if args.dry_run:
        print(json.dumps(document, indent=2))
        return 0

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(zones)} zone(s) to {out}")
    print("Reminder: these are mapped OSM features, not verified safety data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
