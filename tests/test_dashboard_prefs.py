"""Unit tests for Phase 2 DashboardPreferences (persistence)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from settings_store import DashboardPreferences, SettingsStore  # noqa: E402


class DashboardPrefsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "prefs.json"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_defaults_all_enabled(self) -> None:
        prefs = DashboardPreferences(path=self.path)
        self.assertEqual(len(prefs.all_enabled()), len(prefs.COMPONENTS))

    def test_disable_and_persist(self) -> None:
        prefs = DashboardPreferences(path=self.path)
        prefs.set_enabled("blink_count", False)
        self.assertFalse(prefs.is_enabled("blink_count"))
        # Reload from disk.
        prefs2 = DashboardPreferences(path=self.path)
        self.assertFalse(prefs2.is_enabled("blink_count"))
        self.assertTrue(prefs2.is_enabled("fatigue_risk"))

    def test_reset_defaults(self) -> None:
        prefs = DashboardPreferences(path=self.path)
        prefs.set_enabled("yawn_count", False)
        prefs.reset_defaults()
        self.assertTrue(prefs.is_enabled("yawn_count"))

    def test_invalid_component_ignored(self) -> None:
        prefs = DashboardPreferences(path=self.path)
        prefs.set_enabled("not_a_component", True)
        self.assertFalse(prefs.is_enabled("not_a_component"))

    def test_corrupt_file_falls_back_to_defaults(self) -> None:
        self.path.write_text("{invalid json", encoding="utf-8")
        prefs = DashboardPreferences(path=self.path)
        self.assertEqual(len(prefs.all_enabled()), len(prefs.COMPONENTS))


class SettingsStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "settings.json"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_set_get_roundtrip(self) -> None:
        store = SettingsStore(path=self.path)
        store.set("alert_volume", 0.8)
        store2 = SettingsStore(path=self.path)
        self.assertAlmostEqual(store2.get("alert_volume"), 0.8)

    def test_defaults(self) -> None:
        store = SettingsStore(path=self.path)
        self.assertEqual(store.get("voice_enabled"), True)
        self.assertEqual(store.get("alert_sound"), "soft_beep_2")

    def test_corrupt_file_falls_back(self) -> None:
        self.path.write_text("not json", encoding="utf-8")
        store = SettingsStore(path=self.path)
        self.assertEqual(store.get("voice_enabled"), True)


if __name__ == "__main__":
    unittest.main()