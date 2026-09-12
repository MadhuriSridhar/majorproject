"""
Phase 2: Local settings persistence (alert / audio / dashboard prefs).

Stores settings as JSON in the project folder. No cloud, no database.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = PROJECT_ROOT / "settings.json"

DEFAULT_SETTINGS: dict = {
    "alert_volume": 0.6,
    "alarm_volume": 0.7,  # independent alarm volume (WARNING / DROWSY)
    "voice_enabled": True,
    "visual_alerts_enabled": True,
    "sound_enabled": True,
    "alert_sound": "soft_beep_2",
    "warning_sound": "warning_1",
    "drowsy_sound": "strong_alarm_1",
    # Attention / context feature toggles.
    "distraction_enabled": True,
    "location_risk_enabled": True,
    "location_mode": "gps",  # gps | demo | disabled
    "demo_zone": "normal",
    "weather_enabled": True,
}


class SettingsStore:
    """Read/write application settings to a local JSON file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else SETTINGS_FILE
        self._data: dict = dict(DEFAULT_SETTINGS)
        self._load()

    def _load(self) -> None:
        if self.path.is_file():
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    # Merge with defaults so missing keys get defaults.
                    merged = dict(DEFAULT_SETTINGS)
                    merged.update(loaded)
                    self._data = merged
            except Exception:
                # Corrupt file - fall back to defaults.
                self._data = dict(DEFAULT_SETTINGS)

    def save(self) -> None:
        """Persist current settings to JSON."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2)
        except Exception:
            # Persistence failure should never crash the app.
            pass

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def set(self, key: str, value) -> None:
        self._data[key] = value
        self.save()


class DashboardPreferences:
    """
    Stores which dashboard components are enabled/disabled.

    Persisted locally as JSON. The UI decides visibility; the backend is
    never changed.
    """

    COMPONENTS = (
        "fatigue_risk",
        "eye_monitoring",
        "blink_count",
        "blink_rate",
        "yawn_count",
        "yawn_frequency",
        "eye_closure",
        "driving_duration",
        "current_time",
        "night_context",
        "location_context",
        "weather_context",
        "camera_fps",
        "camera_status",
    )

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else PROJECT_ROOT / "dashboard_prefs.json"
        self._enabled: set[str] = set(self.COMPONENTS)
        self._load()

    def _load(self) -> None:
        if self.path.is_file():
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                enabled = data.get("enabled", [])
                if isinstance(enabled, list):
                    saved = set(enabled)
                    self._enabled = {c for c in enabled if c in self.COMPONENTS}
                    # Newly added components default ON for older prefs files.
                    for comp in ("location_context", "weather_context"):
                        if comp in self.COMPONENTS and comp not in saved:
                            self._enabled.add(comp)
            except Exception:
                self._enabled = set(self.COMPONENTS)

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump({"enabled": sorted(self._enabled)}, fh, indent=2)
        except Exception:
            pass

    def is_enabled(self, component: str) -> bool:
        return component in self._enabled

    def set_enabled(self, component: str, enabled: bool) -> None:
        if component not in self.COMPONENTS:
            return
        if enabled:
            self._enabled.add(component)
        else:
            self._enabled.discard(component)
        self.save()

    def all_enabled(self) -> list[str]:
        return [c for c in self.COMPONENTS if c in self._enabled]

    def reset_defaults(self) -> None:
        self._enabled = set(self.COMPONENTS)
        self.save()