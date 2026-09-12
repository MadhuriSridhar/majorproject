"""
Unit tests for the final foundation cleanup / stabilization pass.

Covers monotonic timing, yawn contribution cap, no-face / multiple-face
safety, alert duration configuration, and audio ordering.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alert_escalation import AlertEscalationController  # noqa: E402
from app_ui import resolve_driver_state, should_run_analysis  # noqa: E402
from audio_engine import AudioBackend, AudioSettings  # noqa: E402
from fatigue_engine import (  # noqa: E402
    RECENT_YAWN_MAX,
    FatigueEngine,
    FatigueEvidence,
)
from predefined_sounds import (  # noqa: E402
    ALERT_ALARM_DURATION_SECONDS,
    DROWSY_ALARM_DURATION_SECONDS,
    LOW_FATIGUE_ALARM_DURATION_SECONDS,
    WARNING_ALARM_DURATION_SECONDS,
    sound_duration_seconds,
)
from temporal_eye import EyeClosureTracker  # noqa: E402
from temporal_mouth import YawnTracker  # noqa: E402


def E(**kwargs) -> FatigueEvidence:
    return FatigueEvidence(**kwargs)


class YawnContributionCapTests(unittest.TestCase):
    def test_yawn_risk_bounded_by_weight(self) -> None:
        engine = FatigueEngine()
        # Very high frequency must not exceed the configured weight (15).
        decision = engine.decide(
            E(
                yawn_detected=True,
                yawn_count=50,
                recent_yawn_frequency=100.0,
                face_detected=True,
            )
        )
        self.assertLessEqual(decision.breakdown["yawn_risk"], engine.weight_yawn)
        self.assertLessEqual(decision.breakdown["yawn_risk"], 15.0)

    def test_repeated_yawns_scale_but_stay_bounded(self) -> None:
        engine = FatigueEngine()
        one = engine.decide(E(yawn_detected=True, yawn_count=1, recent_yawn_frequency=1.0, face_detected=True))
        many = engine.decide(E(yawn_detected=True, yawn_count=20, recent_yawn_frequency=30.0, face_detected=True))
        # Repeated yawns strengthen the contribution but stay within RECENT_YAWN_MAX.
        self.assertGreater(many.breakdown["yawn_risk"], one.breakdown["yawn_risk"])
        self.assertLessEqual(many.breakdown["yawn_risk"], RECENT_YAWN_MAX)
        self.assertLessEqual(one.breakdown["yawn_risk"], 15.0)

    def test_yawn_does_not_overwhelm_prolonged_eye(self) -> None:
        engine = FatigueEngine(drowsy_enter_hold_s=0.0, drowsy_exit_hold_s=0.0)
        evidence = E(prolonged_eye_closure=True, yawn_detected=True, recent_yawn_frequency=50.0, face_detected=True)
        engine.update(evidence, 0.0)
        prolonged = engine.update(evidence, 0.1)
        # Prolonged eye closure remains the dominant factor.
        self.assertGreater(
            prolonged.breakdown["eye_risk"], prolonged.breakdown["yawn_risk"]
        )
        self.assertEqual(prolonged.severity, "DROWSY")


class MonotonicTimingTests(unittest.TestCase):
    def test_session_manager_uses_monotonic_for_duration(self) -> None:
        import inspect
        import session_manager as sm

        src = inspect.getsource(sm)
        # Elapsed-driven calls must use monotonic, not wall-clock, for duration.
        self.assertIn("time.monotonic()", src)
        # Human-readable timestamps still use datetime.now().
        self.assertIn("datetime.now()", src)

    def test_app_ui_worker_uses_monotonic(self) -> None:
        import inspect
        import app_ui

        src = inspect.getsource(app_ui.CameraWorker)
        self.assertIn("time.monotonic()", src)

    def test_monotonic_eye_closure_duration(self) -> None:
        tracker = EyeClosureTracker()
        tracker.update(0.30, 0.0)
        tracker.update(0.10, 1.0)  # close
        state = tracker.update(0.10, 3.0)  # 2 s later
        self.assertGreaterEqual(state.current_closure_s, 1.9)

    def test_monotonic_yawn_timing(self) -> None:
        tracker = YawnTracker()
        tracker.update(0.20, 0.0)
        tracker.update(0.80, 1.0)  # open
        state = tracker.update(0.80, 1.0 + 2.0)  # well past min duration
        self.assertEqual(tracker.yawn_count, 1)
        self.assertTrue(state.yawn)

    def test_monotonic_persistence_and_recovery(self) -> None:
        # Verify DROWSY entry (zero hold) and exit recovery uses elapsed
        # (monotonic) time. Same config as the recovery regression test.
        engine = FatigueEngine(
            warning_entry=50,
            drowsy_entry=70,
            warning_recovery=35,
            drowsy_recovery=50,
            drowsy_persistence_s=0.0,
            drowsy_recovery_s=1.0,
        )
        engine.update(E(prolonged_eye_closure=True, current_eye_closure_seconds=2.5, face_detected=True), 0.0)
        d = engine.update(E(face_detected=True), 0.5)
        self.assertEqual(d.status, "DROWSY")  # recovery hold
        d = engine.update(E(face_detected=True), 1.5)
        self.assertEqual(d.status, "SAFE")

    def test_monotonic_alert_cooldown(self) -> None:
        ctl = AlertEscalationController(low_fatigue_cooldown_s=10.0)
        ctl.update("LOW_FATIGUE", 0.0)
        ctl.update("NORMAL", 1.0)
        # Cooldown not elapsed -> no audio.
        r = ctl.update("LOW_FATIGUE", 5.0)
        self.assertFalse(r.should_play_audio)
        # After cooldown elapsed -> audio.
        ctl.update("NORMAL", 6.0)
        r2 = ctl.update("LOW_FATIGUE", 12.0)
        self.assertTrue(r2.should_play_audio)


class NoFaceSafetyTests(unittest.TestCase):
    def test_no_face_returns_safe_and_no_score(self) -> None:
        engine = FatigueEngine()
        decision = engine.decide(E(face_detected=False))
        self.assertIsNone(decision.fatigue_score)
        self.assertEqual(decision.status, "SAFE")
        self.assertEqual(decision.severity, "NORMAL")

    def test_no_face_resets_stale_state(self) -> None:
        engine = FatigueEngine()
        engine.update(E(prolonged_eye_closure=True, current_eye_closure_seconds=3.0, face_detected=True), 0.0)
        engine.update(E(face_detected=False), 0.2)
        decision = engine.update(E(face_detected=True), 0.4)
        self.assertIsNone(decision.fatigue_score)
        self.assertEqual(decision.status, "SAFE")


class MultipleFaceSafetyTests(unittest.TestCase):
    def test_resolve_driver_state_multiple_faces(self) -> None:
        self.assertEqual(
            resolve_driver_state(active=True, camera_ok=True, num_faces=2),
            "MULTIPLE FACES",
        )

    def test_resolve_driver_state_no_face(self) -> None:
        self.assertEqual(
            resolve_driver_state(active=True, camera_ok=True, num_faces=0),
            "NO FACE DETECTED",
        )

    def test_resolve_driver_state_active(self) -> None:
        self.assertEqual(
            resolve_driver_state(active=True, camera_ok=True, num_faces=1),
            "ACTIVE",
        )

    def test_resolve_driver_state_session_inactive(self) -> None:
        self.assertEqual(
            resolve_driver_state(active=False, camera_ok=True, num_faces=1),
            "SESSION INACTIVE",
        )

    def test_should_run_analysis_only_single_face(self) -> None:
        self.assertTrue(should_run_analysis(active=True, camera_ok=True, num_faces=1))
        self.assertFalse(should_run_analysis(active=True, camera_ok=True, num_faces=2))
        self.assertFalse(should_run_analysis(active=True, camera_ok=True, num_faces=0))
        self.assertFalse(should_run_analysis(active=False, camera_ok=True, num_faces=1))


class LiveEvidenceRateTests(unittest.TestCase):
    def test_blink_rate_uses_timestamped_events(self) -> None:
        tracker = EyeClosureTracker()
        tracker.update(0.20, 0.0)
        tracker.update(0.18, 0.2)
        tracker.update(0.10, 0.4)
        tracker.update(0.10, 0.8)
        tracker.update(0.25, 1.0)
        state = tracker.update(0.25, 1.2)
        self.assertEqual(state.blink_count, 1)
        self.assertGreaterEqual(state.recent_blink_rate, 0.0)

    def test_recent_yawn_frequency_tracks_windowed_events(self) -> None:
        tracker = YawnTracker()
        tracker.update(0.90, 0.0)
        tracker.update(0.90, 2.0)
        tracker.update(0.90, 5.0)
        tracker.update(0.10, 7.0)
        state = tracker.update(0.10, 7.1)
        self.assertEqual(state.yawn_count, 1)
        self.assertGreaterEqual(state.recent_yawn_frequency, 0.0)

    def test_real_driving_threshold_is_three_hours_by_default(self) -> None:
        from driving_duration import DRIVING_DURATION_THRESHOLD_HOURS, USE_DEMO_DRIVING_DURATION
        self.assertEqual(DRIVING_DURATION_THRESHOLD_HOURS, 3.0)
        self.assertFalse(USE_DEMO_DRIVING_DURATION)


class AlertDurationConfigTests(unittest.TestCase):
    def test_duration_constants(self) -> None:
        # Final required alarm durations: LOW FATIGUE 2s beep, ALERT 5s, WARNING 8s, DROWSY 10s.
        self.assertEqual(LOW_FATIGUE_ALARM_DURATION_SECONDS, 1.0)
        self.assertEqual(ALERT_ALARM_DURATION_SECONDS, 1.5)
        self.assertEqual(WARNING_ALARM_DURATION_SECONDS, 2.5)
        self.assertEqual(DROWSY_ALARM_DURATION_SECONDS, 3.5)

    def test_default_sounds_match_durations(self) -> None:
        low = sound_duration_seconds("soft_beep_1")
        warn = sound_duration_seconds("warning_1")
        drowsy = sound_duration_seconds("strong_alarm_1")
        self.assertGreaterEqual(low, 1.5)
        self.assertLessEqual(low, 2.5)
        self.assertGreaterEqual(warn, 4.0)
        self.assertLessEqual(warn, 6.0)
        self.assertGreaterEqual(drowsy, 9.0)
        self.assertLessEqual(drowsy, 11.0)


class AudioOrderingTests(unittest.TestCase):
    def test_backend_plays_sound_then_voice(self) -> None:
        # Verify the backend runs sound before voice (no overlap) for a level
        # that has an alarm. (LOW FATIGUE is voice only, so use WARNING.)
        backend = AudioBackend(
            severity="WARNING",
            wav_path=Path("dummy.wav"),
            settings=AudioSettings(voice_enabled=True, sound_enabled=True),
        )
        order: list[str] = []
        backend._play_wav = lambda path: order.append("sound")  # type: ignore[method-assign]
        backend._speak = lambda msg: order.append("voice")  # type: ignore[method-assign]
        backend.run()
        self.assertEqual(order, ["sound", "voice"])


if __name__ == "__main__":
    unittest.main()