"""
Phase 2: Professional Tkinter dashboard application.

Screens:
1. Application start (START DRIVING)
2. Main dashboard (camera, fatigue risk, driver metrics, risk breakdown,
   context, system status)
3. Settings (alert / audio / dashboard customization)
4. Session summary (EXPORT PDF / NEW DRIVING SESSION / EXIT)

The camera + detection pipeline runs in a background thread (CameraWorker).
The UI reads the latest state from the worker - it never computes fatigue.
"""

from __future__ import annotations

import queue
import os
import threading
import time
import tkinter as tk
from dataclasses import replace
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from alert_escalation import (
    AlertEscalationController,
    AlertEscalationResult,
)
from attention_engine import AttentionEngine
from audio_engine import AudioEngine, AudioSettings
from alert_logging import log_alert_decision
from settings_store import DashboardPreferences
from distraction_monitor import DistractionMonitor
from driving_duration import USE_DEMO_DRIVING_DURATION
from evidence_bridge import build_fatigue_evidence
from eye_features import analyze_eyes, draw_ear_eye_landmarks
from facial_landmarks import (
    ModelMissingError,
    create_face_landmarker,
    detect_landmarks,
)
from fatigue_engine import FatigueEngine
from location_context import (
    DEMO_ZONE_LABELS,
    LocationRiskProvider,
    ZoneTransitionTracker,
    format_location_summary,
    format_zone_event_summary,
)
from mouth_features import analyze_mouth, draw_mar_mouth_landmarks
from weather_context import WeatherContextProvider, format_weather_summary
from predefined_sounds import (
    ALERT_SOUNDS,
    DROWSY_SOUNDS,
    WARNING_SOUNDS,
    ensure_sounds_generated,
)
from session_manager import SessionManager
from session_summary import build_summary_text, export_pdf
from settings_store import SettingsStore
from temporal_eye import EyeClosureTracker
from temporal_mouth import YawnTracker
from time_context import get_time_context

# ---------------------------------------------------------------------------
# Theme / colors
# ---------------------------------------------------------------------------

BG = "#12141c"
BG_PANEL = "#1a1d29"
BG_CARD = "#212636"
BG_INPUT = "#2a3040"
FG = "#e8eaf0"
FG_MUTED = "#9aa3b5"
FG_BRIGHT = "#ffffff"
ACCENT = "#3b82f6"

GREEN = "#22c55e"
YELLOW = "#eab308"
AMBER = "#f59e0b"
ORANGE = "#f97316"
RED = "#ef4444"
GRAY = "#6b7280"

SEVERITY_COLORS = {
    "NORMAL": GREEN,
    "LOW_FATIGUE": YELLOW,
    "ALERT": AMBER,
    "WARNING": ORANGE,
    "DROWSY": RED,
    "NO FACE DETECTED": GRAY,
    "MULTIPLE FACES": GRAY,
    "SESSION INACTIVE": GRAY,
    "CAMERA ERROR": GRAY,
}

# Fixed factor colors for the risk breakdown trend lines and legend.
FACTOR_COLORS = {
    "eye_risk": "#3b82f6",       # Blue
    "blink_risk": "#22c55e",     # Green
    "yawn_risk": "#eab308",      # Yellow
    "night_risk": "#a855f7",     # Purple
    "duration_risk": "#f97316",  # Orange
}

FONT_TITLE = ("Segoe UI", 22, "bold")
FONT_SUBTITLE = ("Segoe UI", 12)
FONT_BODY = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)
FONT_BIG = ("Segoe UI", 42, "bold")

SESSION_WARMUP_MIN_S = 1.5
SESSION_WARMUP_MIN_FRAMES = 5


def resolve_driver_state(active: bool, camera_ok: bool, num_faces: int) -> str:
    """
    Determine the system / driver state from session + camera + face count.

    Pure function so the no-face / multiple-face safety behaviour is testable
    without a webcam. Priority: SESSION INACTIVE > CAMERA ERROR >
    MULTIPLE FACES > NO FACE DETECTED > ACTIVE.
    """
    if not active:
        return "SESSION INACTIVE"
    if not camera_ok:
        return "CAMERA ERROR"
    if num_faces > 1:
        return "MULTIPLE FACES"
    if num_faces < 1:
        return "NO FACE DETECTED"
    return "ACTIVE"


def should_run_analysis(active: bool, camera_ok: bool, num_faces: int) -> bool:
    """
    Return True only when there is exactly one valid face to analyse.

    No face or multiple faces pause driver-fatigue analysis (no arbitrary face
    selection, no stale measurements, no false DROWSY from multiple faces).
    """
    return active and camera_ok and num_faces == 1


def format_carla_context(context: dict | None) -> dict[str, str]:
    """Format the existing CARLA snapshot for compact dashboard display."""
    snapshot = context or {}
    if not snapshot.get("carla_connected"):
        return {
            "vehicle": "UNAVAILABLE",
            "speed": "--",
            "weather": "--",
            "time": "--",
            "session": "--",
            "carla_zone": "--",
            "carla_position": "--",
            "zone_risk": "--",
        }

    weather = snapshot.get("weather") or {}
    time_context = snapshot.get("time") or {}
    location = snapshot.get("location") or {}
    position = snapshot.get("position") or {}
    vehicle = str(snapshot.get("vehicle_state") or "NO_VEHICLE").upper()
    speed = float(snapshot.get("speed_kmh") or 0.0)
    session_id = snapshot.get("driving_session_id")

    zone_name = str(location.get("zone_name") or "Normal Road")
    zone_type = str(location.get("zone_type") or "none")
    if zone_type in {"none", "", "unknown"}:
        zone_name = "Normal Road"
    risk_value = int(round(float(location.get("risk_level") or 0.0)))

    pos_x = 0.0
    pos_y = 0.0
    if isinstance(position, dict):
        try:
            pos_x = float(position.get("x") or 0.0)
            pos_y = float(position.get("y") or 0.0)
        except (TypeError, ValueError):
            pos_x = 0.0
            pos_y = 0.0
    carla_position = f"x={pos_x:.1f}, y={pos_y:.1f}"

    return {
        "vehicle": vehicle,
        "speed": f"{speed:.1f} km/h",
        "weather": str(weather.get("condition") or "UNKNOWN").upper(),
        "time": str(time_context.get("period") or "UNKNOWN").upper(),
        "session": str(session_id) if session_id else "--",
        "zone": zone_name,
        "position": carla_position,
        "zone_risk": f"{risk_value}/100",
        "carla_zone": zone_name,
        "carla_position": carla_position,
    }


# A short deterministic grace period for transient landmark dropouts (e.g. a
# head turn that briefly hides the face). Within this window the system keeps
# its last valid state and shows "temporarily unavailable" instead of treating
# the loss as monitoring shutdown or fatigue. Time-based (not frame-count based).
FACE_LOSS_GRACE_S = 1.5
MAX_RENDERED_HISTORY_POINTS = 600


def create_optional_carla_context():
    """Create CARLA context automatically when a CARLA server is available.

    The application attempts to connect to localhost:2000 by default. If a
    CARLA server is reachable, the existing CarlaContext is created and used
    for vehicle/environment context. If CARLA is unavailable, the application
    gracefully shows unavailable context and continues without CARLA.

    The user does NOT need to manually set CARLA_ENABLED, CARLA_ROOT, or
    PYTHONPATH. These remain available as overrides for advanced use cases.
    """
    # Allow explicit override for advanced users, but default to automatic
    # detection when no override is specified.
    explicit_disabled = os.environ.get("CARLA_ENABLED", "").strip().lower() in {"0", "false", "no", "off"}
    if explicit_disabled:
        return None

    host = os.environ.get("CARLA_HOST", "localhost")
    port = int(os.environ.get("CARLA_PORT", "2000"))

    # Quick connectivity check before attempting full context creation.
    # This avoids blocking on a long timeout when no CARLA server is running.
    if not _carla_host_reachable(host, port):
        return None

    try:
        from carla_context import CarlaContext

        return CarlaContext(host=host, port=port)
    except Exception as exc:
        print(f"CARLA context unavailable: {exc}")
        return None


def _carla_host_reachable(host: str, port: int, timeout_s: float = 2.0) -> bool:
    """Check whether a CARLA server is reachable on the given host/port.

    Uses a short TCP connect timeout so the check is non-blocking when no
    server is running.
    """
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def head_is_down(head) -> bool:
    """True only when head pose is actively classified as looking down.

    Downward head orientation compresses the apparent eye opening (EAR), so
    eye-closure-based fatigue must not be inferred from that period.
    """
    if head is None:
        return False
    return bool(
        getattr(head, "looking_away", False)
        and getattr(head, "direction", "forward") == "down"
    )


def head_is_turned(head) -> bool:
    """True only when head pose is actively turned left/right (not down).

    A strongly turned head hides the eyes/mouth, so the face is still PRESENT
    but its facial subfeatures are not reliably visible. This is a potential
    distraction signal, never fatigue in itself.
    """
    if head is None:
        return False
    return bool(
        getattr(head, "looking_away", False)
        and getattr(head, "direction", "forward") in ("left", "right")
    )


def classify_face_state(
    face_detected: bool, lost_for_s: float | None, grace_s: float
) -> str:
    """FACE_DETECTED | FACE_TEMPORARILY_LOST | FACE_NOT_DETECTED."""
    if face_detected:
        return "FACE_DETECTED"
    if lost_for_s is not None and lost_for_s < grace_s:
        return "FACE_TEMPORARILY_LOST"
    return "FACE_NOT_DETECTED"


class CameraWorker(threading.Thread):
    """
    Background thread: capture + MediaPipe + FatigueEngine.

    Runs the existing detection pipeline and exposes the latest frame + state
    to the UI in a thread-safe way.
    """

    def __init__(
        self,
        session_manager: SessionManager,
        alert_controller: AlertEscalationController,
        audio_engine: AudioEngine,
        on_status_change=None,
        settings_store: SettingsStore | None = None,
        face_loss_grace_s: float = FACE_LOSS_GRACE_S,
        carla_context=None,
    ) -> None:
        super().__init__(name="camera-worker", daemon=True)
        self.session_manager = session_manager
        self.alert_controller = alert_controller
        self.audio_engine = audio_engine
        self.on_status_change = on_status_change
        self.settings_store = settings_store
        self.carla_context = carla_context

        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.state: dict = {}
        self.stop_event = threading.Event()
        self.camera_ok = False
        self.camera_state = "STARTING"
        self.camera_error: str | None = None
        self._lock = threading.Lock()

        # Context (location / weather) is fetched in a dedicated background
        # thread so GPS/HTTP never blocks the real-time camera loop. The camera
        # loop only reads the latest published snapshot.
        self._context_lock = threading.Lock()
        self._location_ctx = None
        self._weather_ctx = None
        self._carla_ctx: dict = {}
        self._context_stop = threading.Event()
        self._context_thread: threading.Thread | None = None

        # Persistent detection objects.
        self.eye_tracker = EyeClosureTracker()
        self.yawn_tracker = YawnTracker()
        self.fatigue_engine = FatigueEngine()
        self.distraction_monitor = DistractionMonitor()
        self.attention_engine = AttentionEngine()
        self.location_provider = LocationRiskProvider()
        self.weather_provider = WeatherContextProvider()
        self.zone_tracker = ZoneTransitionTracker()
        self._last_zone_event_text = ""
        self.face_loss_grace_s = float(face_loss_grace_s)
        self._face_lost_since: float | None = None
        self._last_distraction_obs = None
        self.distraction_enabled = True
        self.location_risk_enabled = True
        self.landmarker = None
        self.cap = None
        self._start_ms = 0.0
        self._fps = 0.0
        self._last_fps_time = 0.0
        self._frame_count = 0
        self._last_timestamp_ms = -1
        self._valid_evidence_since: float | None = None
        self._valid_evidence_frames = 0
        self._last_diagnostic_signature: tuple | None = None
        self._last_diagnostic_log_s: float | None = None
        self._apply_context_settings()

    def reset_session_state(self) -> None:
        """Clear all driver evidence before a new session begins."""
        self.eye_tracker.reset()
        self.yawn_tracker.reset()
        self.fatigue_engine.reset(clear_history=True)
        self.distraction_monitor.reset()
        self.attention_engine.reset()
        self.alert_controller.reset()
        self._valid_evidence_since = None
        self._valid_evidence_frames = 0
        self._last_diagnostic_signature = None
        self._last_diagnostic_log_s = None
        self.zone_tracker.reset()
        self._last_zone_event_text = ""
        self._face_lost_since = None
        self._last_distraction_obs = None
        self._last_logged_severity = "NORMAL"
        with self._context_lock:
            self._location_ctx = None
            self._weather_ctx = None
            self._carla_ctx = {}

    def _evidence_ready(self, now_s: float, face_valid: bool) -> bool:
        """Require a short contiguous run of valid face evidence at startup."""
        if not face_valid:
            self._valid_evidence_since = None
            self._valid_evidence_frames = 0
            return False
        if self._valid_evidence_since is None:
            self._valid_evidence_since = now_s
            self._valid_evidence_frames = 0
        self._valid_evidence_frames += 1
        return (
            self._valid_evidence_frames >= SESSION_WARMUP_MIN_FRAMES
            and now_s - self._valid_evidence_since >= SESSION_WARMUP_MIN_S
        )

    def _apply_context_settings(self) -> None:
        """Refresh location/weather/distraction config from SettingsStore."""
        store = self.settings_store
        if store is None:
            self.distraction_enabled = True
            self.location_risk_enabled = True
            return
        mode = str(store.get("location_mode", "gps") or "gps").lower()
        demo_zone = str(store.get("demo_zone", "normal") or "normal")
        weather_on = bool(store.get("weather_enabled", True))
        self.distraction_enabled = bool(store.get("distraction_enabled", True))
        self.location_risk_enabled = bool(store.get("location_risk_enabled", True))
        # Location Mode Disabled always wins over the risk toggle.
        if not self.location_risk_enabled and mode != "disabled":
            # Keep GPS/demo acquisition for display, but risk is zeroed later.
            pass
        self.location_provider.configure(mode=mode, demo_zone=demo_zone)
        self.weather_provider.configure(enabled=weather_on)

    def _log_alert_decision(self, alert_result, attention, eye_state) -> None:
        """Emit one traceable alert-decision log line (kept off the hot path)."""
        suppressed = bool(
            not alert_result.should_play_audio
            and alert_result.active_severity != "NORMAL"
        )
        log_alert_decision(
            severity=alert_result.active_severity,
            reason=alert_result.reason or getattr(attention, "primary_reason", ""),
            alertness=getattr(attention, "alertness_score", None),
            fatigue=getattr(attention, "fatigue_score", None),
            distraction=getattr(attention, "distraction_score", None),
            context_urgency=getattr(attention, "context_urgency", None),
            head_direction=getattr(attention, "head_direction", "unknown"),
            eye_state=getattr(eye_state, "eye", "UNKNOWN"),
            face_detected=True,
            audio="PLAY" if alert_result.should_play_audio else "NONE",
            suppressed=suppressed,
            suppression_reason="same_event_cooldown" if suppressed else "",
        )

    def _log_diagnostics(
        self,
        now_s: float,
        eye_state,
        mouth_state,
        distraction,
        decision,
        attention,
        alert_result,
    ) -> dict:
        """Publish and throttle live pipeline diagnostics for startup triage."""
        head = distraction.head if distraction is not None else None
        values = {
            "ear": getattr(eye_state, "ear", None),
            "eye_state": getattr(eye_state, "eye", "UNKNOWN"),
            "eye_closure_s": getattr(eye_state, "current_closure_s", 0.0),
            "blink_count": getattr(eye_state, "blink_count", 0),
            "blink_rate": getattr(eye_state, "blink_rate", None),
            "mar": getattr(mouth_state, "mar", None),
            "yawn_count": getattr(mouth_state, "yawn_count", 0),
            "head_yaw": getattr(head, "yaw_deg", None),
            "head_pitch": getattr(head, "pitch_deg", None),
            "head_direction": getattr(head, "direction", "unknown"),
            "head_direction_duration_s": getattr(head, "duration_s", 0.0),
            "distraction_channels": {
                name: getattr(getattr(distraction, name, None), "score", 0.0)
                for name in ("head_turn_left", "head_turn_right", "looking_down", "passenger_side")
            },
            "raw_distraction_risk": getattr(distraction, "distraction_risk", 0.0),
            "fatigue_risk": getattr(decision, "smoothed_risk", None),
            "raw_fatigue_risk": getattr(decision, "fatigue_score", 0.0),
            "raw_attention_risk": getattr(attention, "raw_combined", 0.0),
            "smoothed_attention_risk": getattr(attention, "combined_risk", 0.0),
            "raw_alertness": max(0.0, 100.0 - float(getattr(attention, "raw_combined", 0.0))),
            "smoothed_alertness": getattr(attention, "alertness_score", 100.0),
            "candidate_severity": getattr(attention, "level", "SAFE"),
            "alertness": getattr(attention, "alertness_score", 100.0),
            "alert_severity": getattr(alert_result, "active_severity", "NORMAL"),
            "primary_alert_reason": getattr(attention, "primary_reason", ""),
            "alert_debug": self.alert_controller.debug_state,
        }
        signature = (
            values["eye_state"], values["head_direction"], values["alert_severity"],
            round(float(values["smoothed_attention_risk"] or 0.0), 1),
            values["primary_alert_reason"],
        )
        should_log = (
            signature != self._last_diagnostic_signature
            and (
                self._last_diagnostic_log_s is None
                or now_s - self._last_diagnostic_log_s >= 2.0
            )
        )
        if should_log:
            print(f"DIAGNOSTICS: {values}")
            self._last_diagnostic_signature = signature
            self._last_diagnostic_log_s = now_s
        return values

    def _context_worker(self) -> None:
        """Background loop: fetch location + weather off the camera thread.

        The providers throttle their own network work; this loop only drives
        them on a slow cadence and publishes the latest result so the camera
        loop can keep using the last-known context without ever waiting on HTTP.
        """
        while not self._context_stop.is_set():
            try:
                self._apply_context_settings()
                now = time.monotonic()
                location_ctx = self.location_provider.update(now)
                if location_ctx.available:
                    self.weather_provider.set_coordinates(
                        location_ctx.latitude, location_ctx.longitude
                    )
                weather_ctx = self.weather_provider.update(now)
                carla_ctx = {}
                if self.carla_context is not None:
                    try:
                        carla_ctx = self.carla_context.get_context()
                    except Exception as exc:
                        print(f"CARLA context update error: {exc}")
                with self._context_lock:
                    self._location_ctx = location_ctx
                    self._weather_ctx = weather_ctx
                    self._carla_ctx = carla_ctx
            except Exception as exc:
                print(f"Context worker error: {exc}")
            self._context_stop.wait(1.0)

    def start_camera(self) -> None:
        self.stop_event.clear()
        self.camera_state = "STARTING"
        self.camera_error = None
        self.camera_ok = False
        try:
            self.landmarker = create_face_landmarker()
        except ModelMissingError as exc:
            self.camera_error = str(exc)
            self.camera_state = "ERROR"
            self.camera_ok = False
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.camera_error = "Unable to access webcam."
            self.camera_state = "ERROR"
            self.camera_ok = False
            return
        self._start_ms = time.monotonic() * 1000.0
        self._last_fps_time = time.monotonic()
        self._last_timestamp_ms = -1
        # Only start the loop thread once.
        if not self.is_alive():
            self.start()
        # Start (or restart) the context worker so GPS/weather never blocks the
        # camera loop.
        if self._context_thread is None or not self._context_thread.is_alive():
            self._context_stop.clear()
            self._context_thread = threading.Thread(
                target=self._context_worker, name="context-worker", daemon=True
            )
            self._context_thread.start()

    def run(self) -> None:
        """Run the capture/detect loop until stopped, then finalize resources.

        The worker thread owns the camera + MediaPipe objects, so they are
        released here (in `finally`) and never from the Tk UI thread while
        the capture loop may still be using them.
        """
        try:
            self._camera_loop()
        finally:
            self._finalize_camera_resources()

    def _camera_loop(self) -> None:
        """Capture/detect loop; factored out so run() can release via finally."""
        while not self.stop_event.is_set():
            if self.cap is None:
                self.stop_event.wait(0.05)
                continue
            ok, frame = self.cap.read()
            now = time.monotonic()
            if not ok:
                self.camera_error = "Unable to read webcam frames."
                self.camera_ok = False
                self.camera_state = "ERROR"
                self.stop_event.wait(0.2)
                continue

            self.camera_ok = True
            self.camera_error = None
            self.camera_state = "CONNECTED"

            # FPS tracking (elapsed time -> monotonic).
            self._frame_count += 1
            if now - self._last_fps_time >= 1.0:
                self._fps = self._frame_count / (now - self._last_fps_time)
                self._frame_count = 0
                self._last_fps_time = now

            timestamp_ms = time.monotonic_ns() // 1_000_000
            if timestamp_ms <= self._last_timestamp_ms:
                timestamp_ms = self._last_timestamp_ms + 1
            self._last_timestamp_ms = timestamp_ms

            try:
                result = detect_landmarks(self.landmarker, frame, timestamp_ms)
                annotated = frame.copy()
                h, w = annotated.shape[:2]

                faces = result.face_landmarks
                num_faces = len(faces)
                face_detected = num_faces >= 1
                multiple_faces = num_faces > 1

                # Determine system / driver state (pure helper - testable).
                driver_state = resolve_driver_state(
                    self.session_manager.active, self.camera_ok, num_faces
                )

                session_active = self.session_manager.active
                valid_face = should_run_analysis(session_active, self.camera_ok, num_faces)

                # Track the face-loss grace window so a brief dropout (e.g. a
                # head turn) does not reset monitoring or read as fatigue.
                if face_detected:
                    self._face_lost_since = None
                elif self._face_lost_since is None:
                    self._face_lost_since = now
                lost_for = (
                    now - self._face_lost_since if self._face_lost_since is not None else None
                )
                face_state = classify_face_state(face_detected, lost_for, self.face_loss_grace_s)
                in_grace = session_active and face_state == "FACE_TEMPORARILY_LOST"

                if in_grace:
                    warmup_ready = False
                else:
                    warmup_ready = self._evidence_ready(now, valid_face)

                # Facial feature extraction (only with exactly one face).
                eye_state = None
                mouth_state = None
                distraction_obs = None
                if face_detected and not multiple_faces:
                    face = faces[0]
                    eyes = analyze_eyes(face, w, h)
                    mouth = analyze_mouth(face, w, h)
                    draw_ear_eye_landmarks(annotated, face)
                    draw_mar_mouth_landmarks(annotated, face)
                    if session_active:
                        eye_state = self.eye_tracker.update(eyes.average_ear, now)
                        mouth_state = self.yawn_tracker.update(mouth.mar, now)
                        if valid_face and not warmup_ready and eye_state.eye == "OPEN":
                            self.distraction_monitor.head_tracker.calibrate(face, w, h)
                        if warmup_ready:
                            self.distraction_monitor.head_tracker.finish_calibration()
                        if self.distraction_enabled:
                            distraction_obs = self.distraction_monitor.update(face, w, h, now)
                        else:
                            self.distraction_monitor.reset()
                    else:
                        # Preview metrics are frame-local. Do not accumulate
                        # fatigue or distraction evidence before START.
                        eye_state = eyes
                        mouth_state = mouth
                else:
                    if in_grace:
                        # Temporary dropout: retain the last distraction state
                        # instead of clearing channels or reading stale EAR.
                        distraction_obs = self._last_distraction_obs
                    elif session_active:
                        eye_state = self.eye_tracker.update(None, now)
                        mouth_state = self.yawn_tracker.update(None, now)
                        self.distraction_monitor.reset()

                if distraction_obs is not None:
                    self._last_distraction_obs = distraction_obs
                hpose = distraction_obs.head if distraction_obs is not None else None
                head_down = head_is_down(hpose)
                head_turned = head_is_turned(hpose)
                # Eyes/mouth are not reliably visible while the head is strongly
                # turned or down; treat that as a PARTIAL observation, never as
                # "eyes closed" / "mouth open" fatigue evidence.
                eyes_not_visible = bool(head_down or head_turned)
                mouth_not_visible = bool(head_down or head_turned)

                # Location / weather: published by the background context worker.
                # The camera loop never performs network I/O here, so GPS/HTTP
                # failures or delays can never stall the real-time pipeline.
                with self._context_lock:
                    location_ctx = self._location_ctx
                    weather_ctx = self._weather_ctx
                    carla_ctx = dict(self._carla_ctx)
                time_ctx = get_time_context()
                carla_period = str((carla_ctx.get("time") or {}).get("period") or "")
                if carla_ctx.get("carla_connected") and carla_period in {"DAY", "NIGHT"}:
                    time_ctx = replace(
                        time_ctx,
                        context=carla_period,
                        night_context_active=carla_period == "NIGHT",
                    )
                if location_ctx is None:
                    location_ctx = self.location_provider._unavailable("Unavailable")
                if weather_ctx is None:
                    weather_ctx = self.weather_provider._unavailable("Unavailable")

                # CARLA is the authoritative zone source when connected.
                # The regular location provider remains available for the
                # non-CARLA context panel and real-world mode.
                carla_location = carla_ctx.get("location") or {}
                active_zone_type = (
                    str(carla_location.get("zone_type") or "none")
                    if carla_ctx.get("carla_connected")
                    else str(location_ctx.zone_type or "none")
                )
                active_zone_name = (
                    str(carla_location.get("zone_name") or "Normal Road")
                    if carla_ctx.get("carla_connected")
                    else location_ctx.zone_name
                )

                # Zone entry/exit tracking (event semantics, never per frame).
                zone_transition = self.zone_tracker.update(
                    active_zone_type, active_zone_name, now
                )
                if zone_transition is not None:
                    self._last_zone_event_text = format_zone_event_summary(zone_transition)
                    if self.session_manager.active:
                        self.session_manager.record_zone_transition(zone_transition, now)

                loc_risk = (
                    float(location_ctx.risk_level or 0.0)
                    if self.location_risk_enabled and location_ctx.available
                    else 0.0
                )
                wx_risk = (
                    float(weather_ctx.weather_risk or 0.0)
                    if weather_ctx.available
                    else 0.0
                )

                # Build decision only when a session is active and a single face.
                decision = None
                attention = None
                alert_result = None
                if valid_face and warmup_ready:
                    evidence = build_fatigue_evidence(
                        eye=eye_state,
                        mouth=mouth_state,
                        time_ctx=time_ctx,
                        driving=self.session_manager.driving_result(now),
                        face_detected=True,
                        head_down=head_down,
                        head_turned=head_turned,
                        eyes_not_visible=eyes_not_visible,
                        mouth_not_visible=mouth_not_visible,
                    )
                    decision = self.fatigue_engine.update(evidence, now)
                    fatigue_risk = (
                        decision.smoothed_risk
                        if decision.smoothed_risk is not None
                        else float(decision.fatigue_score or 0)
                    )
                    attention = self.attention_engine.update(
                        fatigue_risk=fatigue_risk,
                        distraction=distraction_obs if self.distraction_enabled else None,
                        now_s=now,
                        face_ok=True,
                        location_risk=loc_risk,
                        weather_risk=wx_risk,
                        location_zone=str(location_ctx.zone_type or "none"),
                        weather_condition=str(weather_ctx.weather_condition or "unknown"),
                        carla_context=carla_ctx,
                    )
                    # Alerts follow combined attention severity, but the live
                    # reason and score remain available to the single escalation
                    # controller so the UI and audio speak the true cause.
                    alert_result = self.alert_controller.update(
                        attention.alert_severity,
                        now,
                        reason=attention.primary_reason,
                        contributing_factors=(
                            attention.distraction_primary,
                            "fatigue" if attention.fatigue_score >= 50.0 else "stable",
                        ),
                        alertness_score=attention.alertness_score,
                        fatigue_score=attention.fatigue_score,
                        distraction_score=attention.distraction_score,
                        context_urgency=attention.context_urgency,
                    )
                    if alert_result.should_play_audio:
                        self.audio_engine.play_alert(
                            alert_result.audio_severity,
                            reason=alert_result.reason,
                            contributing_factors=alert_result.contributing_factors,
                        )
                    # One event-level log per alert-decision state change (never
                    # per camera frame) so the arbitration is fully traceable.
                    log_severity = alert_result.active_severity
                    if (
                        alert_result.should_play_audio
                        or log_severity != getattr(self, "_last_logged_severity", "NORMAL")
                    ):
                        self._last_logged_severity = log_severity
                        self._log_alert_decision(alert_result, attention, eye_state)
                    self.session_manager.update(
                        decision,
                        now,
                        attention=attention,
                        alert_result=alert_result,
                    )
                elif valid_face:
                    # Warm-up evidence is allowed to build temporal trackers,
                    # but cannot become UI/alert output yet.
                    evidence = build_fatigue_evidence(
                        eye=eye_state,
                        mouth=mouth_state,
                        time_ctx=time_ctx,
                        driving=self.session_manager.driving_result(now),
                        face_detected=True,
                        head_down=head_down,
                        head_turned=head_turned,
                        eyes_not_visible=eyes_not_visible,
                        mouth_not_visible=mouth_not_visible,
                    )
                    self.fatigue_engine.update(evidence, now)
                    self.attention_engine.reset()
                    alert_result = self.alert_controller.update("NORMAL", now)
                elif in_grace:
                    # Temporary face loss within the grace window: retain the
                    # last valid state. Reset nothing, record nothing, alarm nothing.
                    pass
                else:
                    # No valid fatigue assessment - reset engine state safely.
                    self.fatigue_engine.reset()
                    self.attention_engine.reset()
                    alert_result = self.alert_controller.update("NORMAL", now)
                    # Preserve trend history when the face is temporarily not
                    # detected - record an unavailable sample rather than
                    # resetting or fabricating events.
                    if self.session_manager.active:
                        self.session_manager.record_no_face(now)

                diagnostics = self._log_diagnostics(
                    now,
                    eye_state,
                    mouth_state,
                    distraction_obs,
                    decision,
                    attention,
                    alert_result,
                )

                # Push frame + state to UI.
                try:
                    if self.frame_queue.full():
                        self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(
                        {
                            "frame": annotated,
                            "face_detected": face_detected,
                            "multiple_faces": multiple_faces,
                            "num_faces": num_faces,
                            "driver_state": driver_state,
                            "face_state": face_state,
                            "eye_state": eye_state,
                            "mouth_state": mouth_state,
                            "decision": decision,
                            "attention": attention,
                            "distraction": distraction_obs,
                            "location": location_ctx,
                            "weather": weather_ctx,
                            "carla": carla_ctx,
                            "zone_event": self._last_zone_event_text,
                            "alert_result": alert_result,
                            "diagnostics": diagnostics,
                            "time_ctx": time_ctx,
                            "fps": self._fps,
                            "camera_ok": self.camera_ok,
                        }
                    )
                except Exception as exc:
                    self.camera_error = f"UI queue update error: {exc}"
                    self.camera_state = "ERROR"
                    self.camera_ok = False
            except Exception as exc:
                self.camera_error = f"Frame processing error: {exc}"
                self.camera_ok = False
                self.camera_state = "ERROR"
                self.stop_event.wait(0.1)
                continue

    def _finalize_camera_resources(self) -> None:
        """Release camera + MediaPipe resources owned by this worker thread.

        Called from run()'s finally block so the resources are never released
        from the Tk UI thread while the capture/detect loop may still be using
        them. Safe to call even if start_camera() failed partway.
        """
        cap = self.cap
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
            self.cap = None
        landmarker = self.landmarker
        if landmarker is not None:
            try:
                landmarker.close()
            except Exception:
                pass
            self.landmarker = None

    def stop_worker(self) -> None:
        """Signal the worker to stop and wait briefly for it to finalize.

        The worker thread owns the camera / MediaPipe resources: this method
        only requests shutdown and joins (with a short timeout so the UI never
        freezes on a hung camera). The worker's finally block then releases
        the resources itself.
        """
        self.stop_event.set()
        self._context_stop.set()
        self.camera_ok = False
        self.camera_state = "DISCONNECTED"
        self.camera_error = "Camera stopped."
        try:
            self.join(timeout=0.5)
        except Exception:
            pass
        context_thread = self._context_thread
        if context_thread is not None and context_thread is not threading.current_thread():
            try:
                context_thread.join(timeout=0.5)
            except Exception:
                pass


class CustomDashboardDialog(tk.Toplevel):
    def __init__(self, master, prefs: DashboardPreferences, on_change=None) -> None:
        super().__init__(master)
        self.prefs = prefs
        self.on_change = on_change
        self.title("Customize Dashboard")
        self.configure(bg=BG)
        self.geometry("380x620")
        self.resizable(False, False)
        self._vars: dict[str, tk.BooleanVar] = {}
        self._build()

    def _build(self) -> None:
        tk.Label(
            self, text="CUSTOMIZE DASHBOARD", font=FONT_SUBTITLE, bg=BG, fg=FG_BRIGHT
        ).pack(pady=(12, 4))
        tk.Label(
            self,
            text="Select which components appear on the dashboard.",
            font=FONT_SMALL,
            bg=BG,
            fg=FG_MUTED,
        ).pack(pady=(0, 8))

        labels = {
            "fatigue_risk": "Fatigue Risk",
            "eye_monitoring": "Eye Monitoring",
            "blink_count": "Blink Count",
            "blink_rate": "Blink Rate",
            "yawn_count": "Yawn Count",
            "yawn_frequency": "Yawn Frequency",
            "eye_closure": "Eye Closure",
            "driving_duration": "Driving Duration",
            "current_time": "Current Time",
            "night_context": "Night Context",
            "location_context": "Location / GPS",
            "weather_context": "Weather",
            "camera_fps": "Camera FPS",
            "camera_status": "Camera Status",
        }

        container = tk.Frame(self, bg=BG)
        container.pack(fill="both", expand=True, padx=16)
        for comp in DashboardPreferences.COMPONENTS:
            var = tk.BooleanVar(value=self.prefs.is_enabled(comp))
            self._vars[comp] = var
            cb = tk.Checkbutton(
                container,
                text=labels.get(comp, comp),
                variable=var,
                bg=BG,
                fg=FG,
                selectcolor=BG_INPUT,
                activebackground=BG,
                activeforeground=FG,
                font=FONT_BODY,
                anchor="w",
                command=lambda c=comp: self._toggle(c),
            )
            cb.pack(fill="x", pady=2)

        btn_row = tk.Frame(self, bg=BG)
        btn_row.pack(pady=10)
        tk.Button(
            btn_row,
            text="Reset Defaults",
            command=self._reset,
            bg=BG_INPUT,
            fg=FG,
            relief="flat",
            font=FONT_BODY,
        ).pack(side="left", padx=6)
        tk.Button(
            btn_row,
            text="Close",
            command=self.destroy,
            bg=ACCENT,
            fg="#fff",
            relief="flat",
            font=FONT_BODY,
        ).pack(side="left", padx=6)

    def _toggle(self, comp: str) -> None:
        self.prefs.set_enabled(comp, self._vars[comp].get())
        if self.on_change:
            self.on_change()

    def _reset(self) -> None:
        self.prefs.reset_defaults()
        for comp, var in self._vars.items():
            var.set(self.prefs.is_enabled(comp))
        if self.on_change:
            self.on_change()


class SettingsDialog(tk.Toplevel):
    def __init__(self, master, store: SettingsStore, audio_engine: AudioEngine) -> None:
        super().__init__(master)
        self.store = store
        self.audio_engine = audio_engine
        self.title("Settings")
        self.configure(bg=BG)
        self.geometry("440x820")
        self.resizable(False, False)
        self._build()

    def _build(self) -> None:
        tk.Label(self, text="SETTINGS", font=FONT_SUBTITLE, bg=BG, fg=FG_BRIGHT).pack(pady=(12, 4))
        tk.Label(self, text="ALERT SETTINGS", font=FONT_SMALL, bg=BG, fg=ACCENT).pack(pady=(8, 2))

        # Independent volume sliders: Alert volume and Alarm volume are separate.
        alert_vol = float(self.store.get("alert_volume", 0.6))
        alarm_vol = float(self.store.get("alarm_volume", 0.7))
        row1 = tk.Frame(self, bg=BG)
        row1.pack(fill="x", padx=20, pady=4)
        tk.Label(row1, text="Alert volume", font=FONT_BODY, bg=BG, fg=FG).pack(side="left")
        self.alert_var = tk.DoubleVar(value=alert_vol)
        ttk.Scale(
            row1, from_=0.0, to=1.0, variable=self.alert_var, command=self._on_alert_volume
        ).pack(side="right", fill="x", expand=True, padx=10)

        row2 = tk.Frame(self, bg=BG)
        row2.pack(fill="x", padx=20, pady=4)
        tk.Label(row2, text="Alarm volume", font=FONT_BODY, bg=BG, fg=FG).pack(side="left")
        self.alarm_var = tk.DoubleVar(value=alarm_vol)
        ttk.Scale(
            row2, from_=0.0, to=1.0, variable=self.alarm_var, command=self._on_alarm_volume
        ).pack(side="right", fill="x", expand=True, padx=10)

        # Voice alerts toggle.
        self.voice_var = tk.BooleanVar(value=self.store.get("voice_enabled", True))
        tk.Checkbutton(
            self,
            text="Voice alerts (pyttsx3)",
            variable=self.voice_var,
            command=self._on_voice,
            bg=BG,
            fg=FG,
            selectcolor=BG_INPUT,
            activebackground=BG,
            activeforeground=FG,
            font=FONT_BODY,
        ).pack(anchor="w", padx=20, pady=4)

        # Visual alerts toggle.
        self.visual_var = tk.BooleanVar(value=self.store.get("visual_alerts_enabled", True))
        tk.Checkbutton(
            self,
            text="Visual alerts",
            variable=self.visual_var,
            command=self._on_visual,
            bg=BG,
            fg=FG,
            selectcolor=BG_INPUT,
            activebackground=BG,
            activeforeground=FG,
            font=FONT_BODY,
        ).pack(anchor="w", padx=20, pady=4)

        # Sound enabled toggle.
        self.sound_var = tk.BooleanVar(value=self.store.get("sound_enabled", True))
        tk.Checkbutton(
            self,
            text="Alarm sounds",
            variable=self.sound_var,
            command=self._on_sound,
            bg=BG,
            fg=FG,
            selectcolor=BG_INPUT,
            activebackground=BG,
            activeforeground=FG,
            font=FONT_BODY,
        ).pack(anchor="w", padx=20, pady=4)

        tk.Label(self, text="PREDEFINED SOUNDS", font=FONT_SMALL, bg=BG, fg=ACCENT).pack(pady=(10, 2))

        # Sound selectors (LOW FATIGUE is voice only - no sound selection).
        self._sound_row("Alert Sound", "alert_sound", ALERT_SOUNDS, "soft_beep_2")
        self._sound_row("Warning Sound", "warning_sound", WARNING_SOUNDS, "warning_1")
        self._sound_row("Drowsy Sound", "drowsy_sound", DROWSY_SOUNDS, "strong_alarm_1")

        tk.Label(
            self,
            text="Audio: predefined local sounds only (no internet).",
            font=("Segoe UI", 8),
            bg=BG,
            fg=FG_MUTED,
        ).pack(pady=(10, 2))

        tk.Label(self, text="LOCATION / WEATHER", font=FONT_SMALL, bg=BG, fg=ACCENT).pack(
            pady=(12, 2)
        )

        self.distraction_var = tk.BooleanVar(
            value=bool(self.store.get("distraction_enabled", True))
        )
        tk.Checkbutton(
            self,
            text="Distraction monitoring",
            variable=self.distraction_var,
            command=self._on_distraction,
            bg=BG,
            fg=FG,
            selectcolor=BG_INPUT,
            activebackground=BG,
            activeforeground=FG,
            font=FONT_BODY,
        ).pack(anchor="w", padx=20, pady=2)

        self.location_risk_var = tk.BooleanVar(
            value=bool(self.store.get("location_risk_enabled", True))
        )
        tk.Checkbutton(
            self,
            text="Location risk",
            variable=self.location_risk_var,
            command=self._on_location_risk,
            bg=BG,
            fg=FG,
            selectcolor=BG_INPUT,
            activebackground=BG,
            activeforeground=FG,
            font=FONT_BODY,
        ).pack(anchor="w", padx=20, pady=2)

        mode_row = tk.Frame(self, bg=BG)
        mode_row.pack(fill="x", padx=20, pady=4)
        tk.Label(
            mode_row, text="Location Mode", font=FONT_BODY, bg=BG, fg=FG, width=18, anchor="w"
        ).pack(side="left")
        mode_display = {
            "gps": "GPS / Automatic",
            "demo": "Demo",
            "disabled": "Disabled",
        }
        cur_mode = str(self.store.get("location_mode", "gps") or "gps").lower()
        self.location_mode_var = tk.StringVar(value=mode_display.get(cur_mode, "GPS / Automatic"))
        mode_combo = ttk.Combobox(
            mode_row,
            textvariable=self.location_mode_var,
            values=("GPS / Automatic", "Demo", "Disabled"),
            state="readonly",
            width=18,
        )
        mode_combo.pack(side="right", padx=10)
        mode_combo.bind("<<ComboboxSelected>>", self._on_location_mode)

        zone_row = tk.Frame(self, bg=BG)
        zone_row.pack(fill="x", padx=20, pady=4)
        tk.Label(
            zone_row, text="Demo Zone", font=FONT_BODY, bg=BG, fg=FG, width=18, anchor="w"
        ).pack(side="left")
        cur_zone = str(self.store.get("demo_zone", "normal") or "normal")
        self.demo_zone_var = tk.StringVar(
            value=DEMO_ZONE_LABELS.get(cur_zone, DEMO_ZONE_LABELS["normal"])
        )
        zone_combo = ttk.Combobox(
            zone_row,
            textvariable=self.demo_zone_var,
            values=tuple(DEMO_ZONE_LABELS.values()),
            state="readonly",
            width=22,
        )
        zone_combo.pack(side="right", padx=10)
        zone_combo.bind("<<ComboboxSelected>>", self._on_demo_zone)

        self.weather_var = tk.BooleanVar(value=bool(self.store.get("weather_enabled", True)))
        tk.Checkbutton(
            self,
            text="Weather monitoring",
            variable=self.weather_var,
            command=self._on_weather,
            bg=BG,
            fg=FG,
            selectcolor=BG_INPUT,
            activebackground=BG,
            activeforeground=FG,
            font=FONT_BODY,
        ).pack(anchor="w", padx=20, pady=4)

        tk.Label(
            self,
            text="Demo location is labeled DEMO. GPS uses device/network when available.",
            font=("Segoe UI", 8),
            bg=BG,
            fg=FG_MUTED,
            wraplength=400,
            justify="left",
        ).pack(pady=(4, 2), padx=20)

        tk.Button(
            self,
            text="Close",
            command=self.destroy,
            bg=ACCENT,
            fg="#fff",
            relief="flat",
            font=FONT_BODY,
        ).pack(pady=10)

    def _sound_row(self, label: str, key: str, options: tuple, default: str) -> None:
        row = tk.Frame(self, bg=BG)
        row.pack(fill="x", padx=20, pady=4)
        tk.Label(row, text=label, font=FONT_BODY, bg=BG, fg=FG, width=18, anchor="w").pack(side="left")
        var = tk.StringVar(value=self.store.get(key, default))
        combo = ttk.Combobox(
            row,
            textvariable=var,
            values=list(options),
            state="readonly",
            width=16,
        )
        combo.pack(side="right")
        combo.bind(
            "<<ComboboxSelected>>",
            lambda e, k=key, v=var: (self.store.set(k, v.get()), self._apply_audio()),
        )

    def _on_alert_volume(self, _val) -> None:
        self.store.set("alert_volume", max(0.0, min(1.0, float(self.alert_var.get()))))
        self._apply_audio()

    def _on_alarm_volume(self, _val) -> None:
        self.store.set("alarm_volume", max(0.0, min(1.0, float(self.alarm_var.get()))))
        self._apply_audio()

    def _on_voice(self) -> None:
        self.store.set("voice_enabled", self.voice_var.get())
        self._apply_audio()

    def _on_visual(self) -> None:
        self.store.set("visual_alerts_enabled", self.visual_var.get())

    def _on_sound(self) -> None:
        self.store.set("sound_enabled", self.sound_var.get())
        self._apply_audio()

    def _on_location_mode(self, _event=None) -> None:
        label = str(self.location_mode_var.get())
        mapping = {
            "GPS / Automatic": "gps",
            "Demo": "demo",
            "Disabled": "disabled",
        }
        self.store.set("location_mode", mapping.get(label, "gps"))

    def _on_demo_zone(self, _event=None) -> None:
        label = str(self.demo_zone_var.get())
        reverse = {v: k for k, v in DEMO_ZONE_LABELS.items()}
        self.store.set("demo_zone", reverse.get(label, "normal"))

    def _on_weather(self) -> None:
        self.store.set("weather_enabled", bool(self.weather_var.get()))

    def _on_distraction(self) -> None:
        self.store.set("distraction_enabled", bool(self.distraction_var.get()))

    def _on_location_risk(self) -> None:
        self.store.set("location_risk_enabled", bool(self.location_risk_var.get()))

    def _apply_audio(self) -> None:
        self.audio_engine.settings = AudioSettings(
            alert_volume=float(self.store.get("alert_volume", 0.6)),
            alarm_volume=float(self.store.get("alarm_volume", 0.7)),
            voice_enabled=bool(self.store.get("voice_enabled", True)),
            sound_enabled=bool(self.store.get("sound_enabled", True)),
            alert_sound=str(self.store.get("alert_sound", "soft_beep_2")),
            warning_sound=str(self.store.get("warning_sound", "warning_1")),
            drowsy_sound=str(self.store.get("drowsy_sound", "strong_alarm_1")),
        )


class DrowsinessApp(tk.Tk):
    """Main Tkinter application."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Neuro-Symbolic Driver Attention Monitoring")
        self.configure(bg=BG)
        self.geometry("1280x800")
        self.minsize(1024, 700)

        # Backend / state objects.
        self.settings_store = SettingsStore()
        self.prefs = DashboardPreferences()
        self.session_manager = SessionManager(use_demo_duration=USE_DEMO_DRIVING_DURATION)
        self.alert_controller = AlertEscalationController()
        self.carla_context = create_optional_carla_context()
        self.audio_engine = AudioEngine(AudioSettings(
            alert_volume=float(self.settings_store.get("alert_volume", 0.6)),
            alarm_volume=float(self.settings_store.get("alarm_volume", 0.7)),
            voice_volume=float(self.settings_store.get("voice_volume", 0.85)),
            voice_enabled=bool(self.settings_store.get("voice_enabled", True)),
            sound_enabled=bool(self.settings_store.get("sound_enabled", True)),
            alert_sound=str(self.settings_store.get("alert_sound", "soft_beep_2")),
            warning_sound=str(self.settings_store.get("warning_sound", "warning_1")),
            drowsy_sound=str(self.settings_store.get("drowsy_sound", "strong_alarm_1")),
        ))

        self.camera_worker = CameraWorker(
            session_manager=self.session_manager,
            alert_controller=self.alert_controller,
            audio_engine=self.audio_engine,
            settings_store=self.settings_store,
            carla_context=self.carla_context,
        )

        self.current_screen = "start"
        self.current_frame = None
        self._last_decision = None
        self._trend_mode = "TREND"
        self._summary_stats = None
        self._running = False
        self._last_camera_state = None
        self.trend_canvas: tk.Canvas | None = None
        self.rb_table: tk.Frame | None = None
        self._last_rendered_history_len = -1
        self._last_rendered_alert_event_count = -1
        self._last_history_render_s = 0.0
        self._last_history_render_mode: str | None = None
        self._last_camera_render_s = 0.0
        self._driver_metrics: dict[str, tk.Label] = {}

        self._build_start_screen()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Start camera on startup (for preview on start screen).
        ensure_sounds_generated()
        self._start_camera_async()

        # Poll camera state.
        self.after(50, self._poll_camera)

    # ------------------------------------------------------------------
    # Screen management
    # ------------------------------------------------------------------

    def _destroy_dashboard_ui(self) -> None:
        """Remove the packed dashboard frames so no part of a live session remains.

        The dashboard's top taskbar, main frame, and status/control bar (with
        STOP DRIVING + CUSTOMIZE) are packed directly on the root, NOT inside
        `current_frame`. So leaving a live dashboard requires explicitly
        destroying them - otherwise STOP DRIVING / CUSTOMIZE / status text /
        camera frames stay visible over the summary. Safe to call when no
        dashboard exists yet.
        """
        for name in ("top_bar", "scroll", "status_bar"):
            w = getattr(self, name, None)
            if w is not None:
                try:
                    w.destroy()
                except Exception:
                    pass
                setattr(self, name, None)
        # Any active alert popup must also disappear with the dashboard.
        self._clear_alert_overlay()

    def _clear_screen(self) -> None:
        if self.current_frame is not None:
            self.current_frame.destroy()
            self.current_frame = None

    def _build_start_screen(self) -> None:
        self._destroy_dashboard_ui()
        self._clear_screen()
        self.current_screen = "start"
        frame = tk.Frame(self, bg=BG)
        frame.pack(fill="both", expand=True)
        self.current_frame = frame

        # Camera preview area.
        preview = tk.Frame(frame, bg=BG_PANEL, width=640, height=360)
        preview.pack(pady=(40, 10))
        preview.pack_propagate(False)
        self.preview_label = tk.Label(preview, bg=BG_PANEL, text="Camera starting...", fg=FG_MUTED)
        self.preview_label.pack(fill="both", expand=True)

        tk.Label(frame, text="DRIVER ATTENTION MONITOR", font=FONT_TITLE, bg=BG, fg=FG_BRIGHT).pack(pady=(10, 4))
        self.camera_status_label = tk.Label(
            frame, text="CAMERA: STARTING", font=FONT_SUBTITLE, bg=BG, fg=FG_MUTED
        )
        self.camera_status_label.pack()
        tk.Label(frame, text="System: READY", font=FONT_SUBTITLE, bg=BG, fg=GREEN).pack(pady=(2, 16))

        self.start_button = tk.Button(
            frame,
            text="[ START DRIVING ]",
            font=("Segoe UI", 16, "bold"),
            bg=GREEN,
            fg="#0a0a0a",
            activebackground="#16a34a",
            relief="flat",
            padx=32,
            pady=12,
            command=self._start_driving,
        )
        self.start_button.pack(pady=16)

        tk.Label(
            frame,
            text="Before START DRIVING, the driving timer, fatigue assessment and\nalerts are inactive. The camera stays on for setup/testing.",
            font=FONT_SMALL,
            bg=BG,
            fg=FG_MUTED,
            justify="center",
        ).pack(pady=(8, 20))

    def _build_dashboard(self) -> None:
        self._destroy_dashboard_ui()
        self._clear_screen()
        self.current_screen = "dashboard"
        self._last_rendered_history_len = -1
        self._last_rendered_alert_event_count = -1
        self._last_history_render_s = 0.0
        self._last_rendered_history_mode = None
        self._build_top_taskbar()
        self._build_main_dashboard()

    def _build_top_taskbar(self) -> None:
        bar = tk.Frame(self, bg=BG_PANEL, height=48)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)
        self.top_bar = bar

        tk.Label(bar, text="NEURO-SYMBOLIC DRIVER ATTENTION MONITORING", font=("Segoe UI", 12, "bold"), bg=BG_PANEL, fg=FG_BRIGHT).pack(side="left", padx=20)

        self.tb_time = tk.Label(bar, text="TIME: --:--:--", font=FONT_SMALL, bg=BG_PANEL, fg=FG)
        self.tb_time.pack(side="right", padx=10)
        self.tb_camera = tk.Label(bar, text="CAMERA: --", font=FONT_SMALL, bg=BG_PANEL, fg=FG)
        self.tb_camera.pack(side="right", padx=10)
        tk.Button(
            bar,
            text="⚙",
            font=("Segoe UI", 14),
            bg=BG_PANEL,
            fg=FG,
            relief="flat",
            command=self._open_settings,
        ).pack(side="right", padx=(10, 20))

    def _build_main_dashboard(self, build_status_bar: bool = True) -> None:
        viewport = tk.Frame(self, bg=BG)
        viewport.pack(fill="both", expand=True)
        self.scroll = viewport
        content_canvas = tk.Canvas(viewport, bg=BG, highlightthickness=0)
        content_scrollbar = ttk.Scrollbar(viewport, orient="vertical", command=content_canvas.yview)
        content_canvas.configure(yscrollcommand=content_scrollbar.set)
        content_scrollbar.pack(side="right", fill="y")
        content_canvas.pack(side="left", fill="both", expand=True)
        main = tk.Frame(content_canvas, bg=BG)
        content_window = content_canvas.create_window((0, 0), window=main, anchor="nw")

        def _sync_scroll_region(_event=None) -> None:
            content_canvas.configure(scrollregion=content_canvas.bbox("all"))

        def _fit_content_width(event) -> None:
            content_canvas.itemconfigure(content_window, width=event.width)

        main.bind("<Configure>", _sync_scroll_region)
        content_canvas.bind("<Configure>", _fit_content_width)

        # Top row: camera + fatigue risk + driver view.
        top = tk.Frame(main, bg=BG)
        top_components = (
            "eye_monitoring", "fatigue_risk", "blink_count", "blink_rate",
            "yawn_count", "yawn_frequency", "eye_closure", "driving_duration",
        )
        if any(self.prefs.is_enabled(comp) for comp in top_components):
            top.pack(fill="x", padx=12, pady=8)

        # Camera panel.
        self.cam_label = self.cam_overlay = None
        if self.prefs.is_enabled("eye_monitoring"):
            cam_panel = tk.Frame(top, bg=BG_PANEL, bd=0, width=760, height=320)
            cam_panel.pack(side="left", fill="y", padx=(0, 6))
            cam_panel.pack_propagate(False)
            tk.Label(cam_panel, text="LIVE CAMERA", font=FONT_SMALL, bg=BG_PANEL, fg=FG_MUTED).pack(anchor="w", padx=10, pady=(6, 0))
            self.cam_label = tk.Label(cam_panel, bg=BG_PANEL, text="Camera starting...", fg=FG_MUTED)
            self.cam_label.pack(fill="both", expand=True, padx=6, pady=6)
            self.cam_overlay = tk.Label(cam_panel, text="", font=FONT_SMALL, bg=BG_PANEL, fg=FG)
            self.cam_overlay.pack(anchor="w", padx=10, pady=(0, 6))

        # Compact CARLA context card between the camera and right-side panels.
        self.ctx_carla_vehicle = self.ctx_carla_speed = None
        self.ctx_carla_weather = self.ctx_carla_time = self.ctx_carla_session = None
        self.ctx_carla_zone = self.ctx_carla_position = self.ctx_carla_zone_risk = None
        carla_panel = tk.Frame(
            top,
            bg=BG_CARD,
            bd=1,
            highlightthickness=1,
            highlightbackground="#2a3040",
            width=210,
            height=360,
        )
        carla_panel.pack(side="left", fill="y", padx=(0, 6))
        carla_panel.pack_propagate(False)
        tk.Label(
            carla_panel,
            text="CARLA DRIVING CONTEXT",
            font=FONT_SMALL,
            bg=BG_CARD,
            fg=FG_MUTED,
            wraplength=185,
        ).pack(anchor="w", padx=10, pady=(8, 10))
        self.ctx_carla_vehicle = tk.Label(carla_panel, text="Vehicle: UNAVAILABLE", font=FONT_SMALL, bg=BG_CARD, fg=FG, anchor="w")
        self.ctx_carla_vehicle.pack(fill="x", padx=10, pady=3)
        self.ctx_carla_speed = tk.Label(carla_panel, text="Speed: --", font=FONT_SMALL, bg=BG_CARD, fg=FG, anchor="w")
        self.ctx_carla_speed.pack(fill="x", padx=10, pady=3)
        self.ctx_carla_weather = tk.Label(carla_panel, text="Weather: --", font=FONT_SMALL, bg=BG_CARD, fg=FG, anchor="w")
        self.ctx_carla_weather.pack(fill="x", padx=10, pady=3)
        self.ctx_carla_time = tk.Label(carla_panel, text="Time: --", font=FONT_SMALL, bg=BG_CARD, fg=FG, anchor="w")
        self.ctx_carla_time.pack(fill="x", padx=10, pady=3)
        self.ctx_carla_zone = tk.Label(carla_panel, text="CARLA Zone: --", font=FONT_SMALL, bg=BG_CARD, fg=FG, anchor="w", wraplength=185, justify="left")
        self.ctx_carla_zone.pack(fill="x", padx=10, pady=3)
        self.ctx_carla_position = tk.Label(carla_panel, text="CARLA Position: --", font=FONT_SMALL, bg=BG_CARD, fg=FG, anchor="w", wraplength=185, justify="left")
        self.ctx_carla_position.pack(fill="x", padx=10, pady=3)
        self.ctx_carla_zone_risk = tk.Label(carla_panel, text="Zone Risk: --", font=FONT_SMALL, bg=BG_CARD, fg=FG, anchor="w")
        self.ctx_carla_zone_risk.pack(fill="x", padx=10, pady=3)
        self.ctx_carla_session = tk.Label(carla_panel, text="Driving Session: --", font=FONT_SMALL, bg=BG_CARD, fg=FG, anchor="w", wraplength=185, justify="left")
        self.ctx_carla_session.pack(fill="x", padx=10, pady=3)

        # Right column.
        right = tk.Frame(top, bg=BG)
        right.pack(side="right", fill="y", padx=(6, 0), expand=False)

        # Overall attention risk panel.
        self.risk_value = self.risk_status = self.risk_bar = None
        self.risk_reason = None
        if self.prefs.is_enabled("fatigue_risk"):
            risk_panel = tk.Frame(right, bg=BG_CARD, bd=1, highlightthickness=1, highlightbackground="#2a3040")
            risk_panel.pack(fill="x", pady=(0, 8))
            tk.Label(risk_panel, text="ALERTNESS SCORE", font=FONT_SMALL, bg=BG_CARD, fg=FG_MUTED).pack(pady=(8, 0))
            self.risk_value = tk.Label(risk_panel, text="--", font=FONT_BIG, bg=BG_CARD, fg=FG_BRIGHT)
            self.risk_value.pack()
            self.risk_status = tk.Label(risk_panel, text="SESSION INACTIVE", font=("Segoe UI", 12, "bold"), bg=BG_CARD, fg=GRAY)
            self.risk_status.pack(pady=(0, 2))
            self.risk_reason = tk.Label(
                risk_panel,
                text="",
                font=("Segoe UI", 9),
                bg=BG_CARD,
                fg=FG_MUTED,
                wraplength=220,
                justify="center",
            )
            self.risk_reason.pack(padx=8, pady=(0, 4))
            bar_canvas = tk.Canvas(risk_panel, bg=BG_CARD, height=14, highlightthickness=0)
            bar_canvas.pack(fill="x", padx=16, pady=(0, 10))
            self.risk_bar = bar_canvas
        self.risk_bar_rect = None

        # Driver view panel.
        self._driver_metrics = {}
        metrics = [
            ("blink_count", "blink_count", "BLINKS"),
            ("blink_rate", "blink_rate", "BLINK RATE"),
            ("yawn_count", "yawn_count", "YAWNS"),
            ("yawn_freq", "yawn_frequency", "YAWN FREQ"),
            ("eye_status", "eye_monitoring", "EYE STATUS"),
            ("eye_closure", "eye_closure", "EYE CLOSURE"),
            ("fatigue", "fatigue_risk", "FATIGUE"),
            ("distract", "fatigue_risk", "DISTRACTION"),
            ("gaze", "fatigue_risk", "HEAD / GAZE"),
            ("driving_time", "driving_duration", "DRIVING TIME"),
            ("risk", "fatigue_risk", "ALERTNESS"),
            ("status", "fatigue_risk", "CURRENT STATE"),
        ]
        visible_metrics = [item for item in metrics if self.prefs.is_enabled(item[1])]
        if visible_metrics:
            driver_panel = tk.Frame(right, bg=BG_CARD, bd=1, highlightthickness=1, highlightbackground="#2a3040")
            driver_panel.pack(fill="x")
            tk.Label(driver_panel, text="DRIVER VIEW", font=FONT_SMALL, bg=BG_CARD, fg=FG_MUTED).pack(pady=(8, 2))
            self.driver_grid = tk.Frame(driver_panel, bg=BG_CARD)
            self.driver_grid.pack(padx=12, pady=(0, 10))
        else:
            self.driver_grid = None
        for i, (key, component, label) in enumerate(visible_metrics):
            r, c = divmod(i, 3)
            cell = tk.Frame(self.driver_grid, bg=BG_INPUT, bd=0)
            cell.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")
            tk.Label(cell, text=label, font=("Segoe UI", 8), bg=BG_INPUT, fg=FG_MUTED).pack(padx=6, pady=(4, 0))
            val = tk.Label(cell, text="--", font=("Segoe UI", 12, "bold"), bg=BG_INPUT, fg=FG_BRIGHT)
            val.pack(padx=6, pady=(0, 4))
            self._driver_metrics[key] = val
        if self.driver_grid is not None:
            for c in range(3):
                self.driver_grid.columnconfigure(c, weight=1)

        # General context strip below the monitoring row.
        self.ctx_time = self.ctx_night = self.ctx_duration = None
        self.ctx_location = self.ctx_weather = None
        self.ctx_zone_event = None
        context_components = (
            "current_time",
            "night_context",
            "driving_duration",
            "location_context",
            "weather_context",
        )
        if any(self.prefs.is_enabled(comp) for comp in context_components):
            context_bar = tk.Frame(
                main,
                bg=BG_CARD,
                bd=1,
                highlightthickness=1,
                highlightbackground="#2a3040",
            )
            context_bar.pack(fill="x", padx=12, pady=(0, 2))
            tk.Label(
                context_bar,
                text="CONTEXT",
                font=FONT_SMALL,
                bg=BG_CARD,
                fg=FG_MUTED,
            ).pack(side="left", padx=(10, 14), pady=7)
            if self.prefs.is_enabled("current_time"):
                self.ctx_time = tk.Label(context_bar, text="Current Time: --:--:--", font=FONT_SMALL, bg=BG_CARD, fg=FG)
                self.ctx_time.pack(side="left", padx=8, pady=7)
            if self.prefs.is_enabled("night_context"):
                self.ctx_night = tk.Label(context_bar, text="Night: INACTIVE", font=FONT_SMALL, bg=BG_CARD, fg=FG)
                self.ctx_night.pack(side="left", padx=8, pady=7)
            if self.prefs.is_enabled("driving_duration"):
                self.ctx_duration = tk.Label(context_bar, text="Driving Duration: 00:00:00", font=FONT_SMALL, bg=BG_CARD, fg=FG)
                self.ctx_duration.pack(side="left", padx=8, pady=7)
            if self.prefs.is_enabled("location_context"):
                self.ctx_location = tk.Label(context_bar, text="Location: Unavailable", font=FONT_SMALL, bg=BG_CARD, fg=FG)
                self.ctx_location.pack(side="left", padx=8, pady=7)
                self.ctx_zone_event = tk.Label(context_bar, text="", font=FONT_SMALL, bg=BG_CARD, fg=AMBER)
            if self.prefs.is_enabled("weather_context"):
                self.ctx_weather = tk.Label(context_bar, text="Weather: Unavailable", font=FONT_SMALL, bg=BG_CARD, fg=FG)
                self.ctx_weather.pack(side="left", padx=8, pady=7)

        # Lower section: risk breakdown + context.
        lower = tk.Frame(main, bg=BG)
        lower.pack(fill="x", padx=12, pady=8)

        # Risk breakdown panel.
        rb_panel = tk.Frame(lower, bg=BG_CARD, bd=1, highlightthickness=1, highlightbackground="#2a3040", height=210)
        rb_panel.pack(side="left", fill="both", expand=True, padx=(0, 6))
        rb_panel.pack_propagate(False)
        rb_header = tk.Frame(rb_panel, bg=BG_CARD)
        rb_header.pack(fill="x", padx=10, pady=(6, 0))
        tk.Label(rb_header, text="ATTENTION TREND & ALERTS", font=FONT_SMALL, bg=BG_CARD, fg=FG_MUTED).pack(side="left")
        self.trend_btn = tk.Button(rb_header, text="TREND", font=FONT_SMALL, bg=ACCENT, fg="#fff", relief="flat", command=lambda: self._set_trend_mode("TREND"))
        self.trend_btn.pack(side="right", padx=(4, 0))
        self.table_btn = tk.Button(rb_header, text="TABLE", font=FONT_SMALL, bg=BG_INPUT, fg=FG, relief="flat", command=lambda: self._set_trend_mode("TABLE"))
        self.table_btn.pack(side="right")
        self.rb_content = tk.Frame(rb_panel, bg=BG_CARD)
        self.rb_content.pack(fill="both", expand=True, padx=10, pady=6)
        self.trend_canvas = tk.Canvas(self.rb_content, bg=BG_CARD, height=150, highlightthickness=0)
        self.trend_canvas.pack(fill="both", expand=True)
        self.trend_canvas.bind("<Configure>", self._on_history_canvas_configure)
        self.rb_table = tk.Frame(self.rb_content, bg=BG_CARD)
        # Wrap the table so wrapped rows never extend beyond the bounded panel.
        self.rb_table_container = tk.Frame(self.rb_content, bg=BG_CARD)
        self.rb_table_canvas = tk.Canvas(self.rb_table_container, bg=BG_CARD, highlightthickness=0)
        self.rb_table_scroll = ttk.Scrollbar(
            self.rb_table_container, orient="vertical", command=self.rb_table_canvas.yview
        )
        self.rb_table_canvas.configure(yscrollcommand=self.rb_table_scroll.set)
        self.rb_table_scroll.pack(side="right", fill="y")
        self.rb_table_canvas.pack(side="left", fill="both", expand=True)
        self.rb_table = tk.Frame(self.rb_table_canvas, bg=BG_CARD)
        self.rb_table_window = self.rb_table_canvas.create_window(
            (0, 0), window=self.rb_table, anchor="nw"
        )
        self.rb_table.bind(
            "<Configure>",
            lambda _e: self.rb_table_canvas.configure(scrollregion=self.rb_table_canvas.bbox("all")),
        )
        self.rb_table_canvas.bind(
            "<Configure>",
            lambda _e: self.rb_table_canvas.itemconfigure(
                self.rb_table_window, width=self.rb_table_canvas.winfo_width()
            ),
        )
        self.rb_table_container.pack(fill="both", expand=True)
        self.rb_table_container.pack_forget()

        # Alert overlay (full-screen DROWSY etc.) - built on top.
        if not hasattr(self, "alert_overlay"):
            self.alert_overlay = None

        # Fixed bottom status/control bar.
        if not build_status_bar:
            self._apply_dashboard_status_preferences()
            return
        self.status_bar = tk.Frame(self, bg=BG_PANEL, height=44)
        self.status_bar.pack(fill="x", side="bottom")
        self.status_bar.pack_propagate(False)
        self.status_fps = tk.Label(self.status_bar, text="FPS: --", font=FONT_SMALL, bg=BG_PANEL, fg=FG)
        self.status_camera = tk.Label(self.status_bar, text="CAMERA: STARTING", font=FONT_SMALL, bg=BG_PANEL, fg=FG)
        self.status_fps.pack(side="left", padx=12)
        self.status_camera.pack(side="left", padx=8)
        self.status_face = tk.Label(self.status_bar, text="FACE: --", font=FONT_SMALL, bg=BG_PANEL, fg=FG)
        self.status_face.pack(side="left", padx=8)
        self.status_system = tk.Label(self.status_bar, text="SYSTEM: RUNNING", font=FONT_SMALL, bg=BG_PANEL, fg=GREEN)
        self.status_system.pack(side="left", padx=8)
        tk.Button(
            self.status_bar,
            text="CUSTOMIZE",
            font=FONT_SMALL,
            bg=BG_INPUT,
            fg=FG,
            relief="flat",
            command=self._open_customize,
        ).pack(side="right", padx=6, pady=6)
        tk.Button(
            self.status_bar,
            text="[ STOP DRIVING ]",
            font=("Segoe UI", 11, "bold"),
            bg=RED,
            fg="#fff",
            relief="flat",
            command=self._stop_driving,
        ).pack(side="right", padx=12, pady=6)
        tk.Button(
            self.status_bar,
            text="[ EXPORT PDF ]",
            font=FONT_SMALL,
            bg=ACCENT,
            fg="#fff",
            relief="flat",
            command=self._export_pdf,
        ).pack(side="right", padx=6, pady=6)
        self._apply_dashboard_status_preferences()

    def _set_trend_mode(self, mode: str) -> None:
        self._trend_mode = mode
        if mode == "TREND":
            self.trend_btn.configure(bg=ACCENT, fg="#fff")
            self.table_btn.configure(bg=BG_INPUT, fg=FG)
        else:
            self.trend_btn.configure(bg=BG_INPUT, fg=FG)
            self.table_btn.configure(bg=ACCENT, fg="#fff")
        if self.current_screen == "dashboard":
            self._last_history_render_mode = None
            self._update_alertness_history_view()

    def _on_history_canvas_configure(self, _event=None) -> None:
        """Invalidate the cached chart when its drawable area changes."""
        if self.current_screen == "dashboard" and self._trend_mode == "TREND":
            self._last_rendered_history_mode = None
            self._update_alertness_history_view()

    def _build_summary_screen(self, stats) -> None:
        self._destroy_dashboard_ui()
        self._clear_screen()
        self.current_screen = "summary"
        frame = tk.Frame(self, bg=BG)
        frame.pack(fill="both", expand=True)
        self.current_frame = frame

        tk.Label(frame, text="SESSION SUMMARY", font=FONT_TITLE, bg=BG, fg=FG_BRIGHT).pack(pady=(30, 10))

        summary_text = build_summary_text(stats)
        text = tk.Text(frame, bg=BG_CARD, fg=FG, font=("Consolas", 11), wrap="none", bd=0, padx=20, pady=10)
        text.pack(fill="both", expand=True, padx=40, pady=10)
        text.insert("1.0", summary_text)
        text.configure(state="disabled")

        btn_row = tk.Frame(frame, bg=BG)
        btn_row.pack(pady=16)
        tk.Button(
            btn_row,
            text="[ EXPORT PDF ]",
            font=FONT_BODY,
            bg=ACCENT,
            fg="#fff",
            relief="flat",
            padx=16,
            pady=8,
            command=self._export_pdf,
        ).pack(side="left", padx=8)
        tk.Button(
            btn_row,
            text="[ NEW DRIVING SESSION ]",
            font=FONT_BODY,
            bg=GREEN,
            fg="#0a0a0a",
            relief="flat",
            padx=16,
            pady=8,
            command=self._new_session,
        ).pack(side="left", padx=8)
        tk.Button(
            btn_row,
            text="[ EXIT ]",
            font=FONT_BODY,
            bg=RED,
            fg="#fff",
            relief="flat",
            padx=16,
            pady=8,
            command=self._on_close,
        ).pack(side="left", padx=8)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _start_driving(self) -> None:
        # Start with a clean slate: no leftover alert popup / audio / stale frame.
        self._clear_alert_overlay()
        self.audio_engine.interrupt_all()
        session_id = self.session_manager.start()
        self.camera_worker.reset_session_state()
        self._running = True
        # Drop any frames pushed by the worker while the session was inactive
        # (preview phase) so the fresh dashboard starts from clean data.
        self._drain_frame_queue()
        self.alert_controller.reset()
        self._last_decision = None
        self._build_dashboard()
        self._last_camera_state = None

    def _stop_driving(self) -> None:
        # Idempotency guard: once the session is stopped/inactive (or a stop is
        # already in progress / completed), ignore extra clicks so the session
        # is never finalized twice and no duplicate summary is created.
        if not getattr(self, "_running", False) and not self.session_manager.active:
            return

        # A. Immediately mark the application as NOT RUNNING so no queued stale
        # frame or already-scheduled poll callback can repaint the live dashboard
        # (risk/counts/trend) during or after the stop sequence.
        self._running = False
        stats = None
        try:
            # C. Stop the camera worker cleanly: signal it and wait briefly.
            # The worker thread owns camera/MediaPipe resources (see
            # CameraWorker.run / _finalize_camera_resources), so nothing here
            # releases them from the Tk UI thread.
            self.camera_worker.stop_worker()
        except Exception:
            # A camera shutdown error must never prevent the rest of the stop
            # sequence (audio/alerts/finalize/summary) from running.
            pass

        try:
            # D. Stop / interrupt all active audio immediately.
            self.audio_engine.interrupt_all()
            # E. Reset the alert escalation controller and cancel/destroy any
            # active alert popup, countdown and glow.
            self.alert_controller.reset()
            self._clear_alert_overlay()
            # F. Finalize the session exactly once (SessionManager.stop is
            # idempotent) and preserve a snapshot for PDF export before the
            # live dashboard is torn down.
            stats = self.session_manager.stop()
            self._summary_stats = stats
            # Drop any frames still queued by the (now stopped) worker.
            self._drain_frame_queue()
        finally:
            # G. Always transition to the clean completion screen - even if a
            # step raised. This replaces the live dashboard, so old
            # risk/counts/trend and SYSTEM RUNNING can never reappear.
            if stats is None:
                stats = self.session_manager.stats
            self._build_summary_screen(stats)

    def _new_session(self) -> None:
        self._running = False
        self._clear_alert_overlay()
        self._build_start_screen()
        self._last_decision = None
        # A Python threading.Thread can only be started once. After STOP
        # DRIVING the previous worker's thread has finished, so create a fresh
        # CameraWorker for the new session instead of re-starting the old one.
        self.camera_worker = CameraWorker(
            session_manager=self.session_manager,
            alert_controller=self.alert_controller,
            audio_engine=self.audio_engine,
            settings_store=self.settings_store,
            carla_context=self.carla_context,
        )
        self._start_camera_async()

    def _start_camera_async(self) -> None:
        """Initialize camera resources off the Tkinter event thread."""
        threading.Thread(
            target=self.camera_worker.start_camera,
            name="camera-startup",
            daemon=True,
        ).start()

    def _export_pdf(self) -> None:
        stats = self._summary_stats or self.session_manager.stats
        if not stats.session_id:
            return
        # Run PDF generation in a background thread to avoid blocking.
        def _do():
            try:
                path = export_pdf(stats)
                self.after(0, lambda: messagebox.showinfo("PDF Export", f"PDF saved to:\n{path}"))
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("PDF Export Error", str(exc)))
        threading.Thread(target=_do, daemon=True).start()

    def _open_settings(self) -> None:
        SettingsDialog(self, self.settings_store, self.audio_engine)

    def _open_customize(self) -> None:
        CustomDashboardDialog(self, self.prefs, on_change=self._apply_dashboard_preferences)

    def _apply_dashboard_preferences(self) -> None:
        """Rebuild dashboard widgets without changing detection or session state."""
        if self.current_screen != "dashboard":
            return
        main = getattr(self, "scroll", None)
        if main is not None:
            main.destroy()
        self.scroll = None
        self._build_main_dashboard(build_status_bar=False)

    def _apply_dashboard_status_preferences(self) -> None:
        """Show or hide status items while keeping the control bar stable."""
        for name, component in (
            ("status_fps", "camera_fps"),
            ("status_camera", "camera_status"),
            ("tb_time", "current_time"),
            ("tb_camera", "camera_status"),
        ):
            widget = getattr(self, name, None)
            if widget is None:
                continue
            if self.prefs.is_enabled(component):
                if name == "tb_time":
                    widget.pack(side="right", padx=10)
                elif name == "tb_camera":
                    widget.pack(side="right", padx=10)
                else:
                    widget.pack(side="left", padx=12 if name == "status_fps" else 8)
            else:
                widget.pack_forget()

    def _on_close(self) -> None:
        self.audio_engine.shutdown()
        self.camera_worker.stop_worker()
        self.session_manager.stop()
        self._clear_alert_overlay()
        self.destroy()

    # ------------------------------------------------------------------
    # Camera polling / UI update
    # ------------------------------------------------------------------

    def _poll_camera(self) -> None:
        # Update start screen camera status & preview.
        if self.current_screen == "start":
            if self.camera_worker.camera_state == "CONNECTED":
                self.camera_status_label.configure(text="CAMERA: CONNECTED", fg=GREEN)
            elif self.camera_worker.camera_state == "ERROR":
                self.camera_status_label.configure(
                    text=f"CAMERA: ERROR - {self.camera_worker.camera_error or 'Unable to access webcam.'}",
                    fg=RED,
                )
            elif self.camera_worker.camera_state == "DISCONNECTED":
                self.camera_status_label.configure(text="CAMERA: DISCONNECTED", fg=RED)
            else:
                self.camera_status_label.configure(text="CAMERA: STARTING", fg=FG_MUTED)

        # Pull latest frame state.
        # Keep only the newest state for the UI. The worker queue is bounded,
        # but draining it here prevents the dashboard from rendering stale
        # frames when detection briefly outpaces Tkinter.
        data = None
        try:
            while True:
                data = self.camera_worker.frame_queue.get_nowait()
        except queue.Empty:
            pass

        if data is not None:
            self._last_decision = data["decision"]
            if self.current_screen == "dashboard" and getattr(self, "_running", False):
                self._update_dashboard(data)
            elif self.current_screen == "start":
                self._update_start_preview(data)

        # Update taskbar time on dashboard.
        if self.current_screen == "dashboard":
            time_ctx = get_time_context()
            if self.prefs.is_enabled("current_time"):
                self.tb_time.configure(text=f"TIME: {time_ctx.time_text}")
            if self.prefs.is_enabled("camera_status"):
                self.tb_camera.configure(
                    text=f"CAMERA: {'CONNECTED' if self.camera_worker.camera_state == 'CONNECTED' else 'ERROR' if self.camera_worker.camera_state == 'ERROR' else 'DISCONNECTED' if self.camera_worker.camera_state == 'DISCONNECTED' else 'STARTING'}",
                    fg=GREEN if self.camera_worker.camera_state == "CONNECTED" else RED,
                )

        self.after(50, self._poll_camera)

    def _update_start_preview(self, data: dict) -> None:
        frame = data["frame"]
        # Resize to preview dimensions.
        h, w = frame.shape[:2]
        target_w = 640
        target_h = 360
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=imgtk, text="")
        self.preview_label.image = imgtk

    def _update_dashboard(self, data: dict) -> None:
        # Guard against stale frames after STOP DRIVING: once the session is
        # stopped, never let old queued data restore live risk/counts/trend
        # or repaint SYSTEM RUNNING on the (fresh) start/summary screen.
        if not getattr(self, "_running", False) or not self.session_manager.active:
            return
        frame = data["frame"]
        decision = data["decision"]
        alert_result = data["alert_result"]
        eye_state = data["eye_state"]
        mouth_state = data["mouth_state"]
        time_ctx = data["time_ctx"]
        multiple = data["multiple_faces"]
        face_detected = data["face_detected"]
        face_state = data.get("face_state", "FACE_DETECTED")
        driver_state = data["driver_state"]

        # Camera display: stable 16:9 preview that preserves aspect ratio and
        # keeps the dashboard layout consistent without zooming or cropping.
        # State/metrics continue updating at the polling rate; only the
        # expensive Tk image conversion is limited to a responsive UI rate.
        now_s = time.monotonic()
        if now_s - self._last_camera_render_s >= 1.0 / 12.0:
            h, w = frame.shape[:2]
            target_w = 640
            target_h = 360
            scale = min(target_w / w, target_h / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(img)
            if self.cam_label is not None:
                self.cam_label.configure(image=imgtk, text="")
                self.cam_label.image = imgtk
            self._last_camera_render_s = now_s

        # Camera overlay.
        if not self.session_manager.active:
            overlay = "SESSION INACTIVE"
        elif multiple:
            overlay = "MULTIPLE FACES"
        elif face_state == "FACE_TEMPORARILY_LOST":
            overlay = "FACE TEMPORARILY UNAVAILABLE"
        elif not face_detected:
            overlay = "FACE NOT DETECTED"
        else:
            ear = eye_state.ear if eye_state else None
            mar = mouth_state.mar if mouth_state else None
            eye_label = eye_state.eye if eye_state else "N/A"
            ear_display = f"{ear:.3f}" if ear is not None else "N/A"
            mar_display = f"{mar:.3f}" if mar is not None else "N/A"
            overlay = (
                f"FACE DETECTED    EAR: {ear_display}    "
                f"MAR: {mar_display}    EYE: {eye_label}"
            )
        if self.cam_overlay is not None:
            self.cam_overlay.configure(text=overlay)

        # FPS / status bar.
        if self.status_fps is not None:
            self.status_fps.configure(text=f"FPS: {data['fps']:.0f}")
        if self.status_camera is not None:
            self.status_camera.configure(
                text=f"CAMERA: {'CONNECTED' if data['camera_ok'] else 'DISCONNECTED'}",
                fg=RED if not data['camera_ok'] else FG,
            )
        if face_state == "FACE_TEMPORARILY_LOST":
            status_face_text = "FACE: TEMPORARILY UNAVAILABLE"
            status_face_color = AMBER
        elif face_detected:
            status_face_text = "FACE: DETECTED"
            status_face_color = GREEN
        else:
            status_face_text = "FACE: NOT DETECTED"
            status_face_color = RED
        self.status_face.configure(text=status_face_text, fg=status_face_color)
        self.status_system.configure(
            text="SYSTEM: RUNNING" if getattr(self, "_running", False) else "SYSTEM: STOPPED",
            fg=GREEN if getattr(self, "_running", False) else GRAY,
        )

        # Overall / attention risk display.
        attention = data.get("attention")
        if attention is not None and attention.face_ok:
            risk = int(round(min(100.0, max(0.0, attention.combined_risk))))
            alertness = int(round(min(100.0, max(0.0, attention.alertness_score))))
            level = attention.level
            color = {
                "SAFE": GREEN,
                "MODERATE": YELLOW,
                "HIGH": ORANGE,
                "CRITICAL": RED,
            }.get(level, SEVERITY_COLORS.get(attention.alert_severity, GRAY))
            if self.risk_value is not None:
                self.risk_value.configure(text=str(alertness), fg=color)
            if self.risk_status is not None:
                self.risk_status.configure(text=level, fg=color)
            if self.risk_reason is not None:
                reason_txt = attention.primary_reason or ""
                if attention.safe_stop and attention.recommendation:
                    reason_txt = f"{reason_txt} — {attention.recommendation}"
                elif attention.recommendation:
                    reason_txt = f"{reason_txt} — {attention.recommendation}"
                self.risk_reason.configure(text=reason_txt)
            self._draw_risk_bar(alertness, color)
        elif decision is not None and decision.smoothed_risk is not None:
            risk = int(round(decision.smoothed_risk))
            severity = decision.severity
            color = SEVERITY_COLORS.get(severity, GRAY)
            if self.risk_value is not None:
                self.risk_value.configure(text=str(risk), fg=color)
            if self.risk_status is not None:
                self.risk_status.configure(text=severity.replace("_", " "), fg=color)
            if self.risk_reason is not None:
                self.risk_reason.configure(text="")
            self._draw_risk_bar(risk, color)
        elif multiple:
            if self.risk_value is not None:
                self.risk_value.configure(text="--", fg=GRAY)
            if self.risk_status is not None:
                self.risk_status.configure(text="MULTIPLE FACES", fg=GRAY)
            if self.risk_reason is not None:
                self.risk_reason.configure(text="")
            self._draw_risk_bar(0, GRAY)
        elif face_state == "FACE_TEMPORARILY_LOST":
            # Keep the last valid alertness value visible; only update the status.
            if self.risk_status is not None:
                self.risk_status.configure(text="FACE TEMPORARILY UNAVAILABLE", fg=GRAY)
            if self.risk_reason is not None:
                self.risk_reason.configure(text="")
        elif not face_detected:
            if self.risk_value is not None:
                self.risk_value.configure(text="--", fg=GRAY)
            if self.risk_status is not None:
                self.risk_status.configure(text="NO FACE DETECTED", fg=GRAY)
            if self.risk_reason is not None:
                self.risk_reason.configure(text="")
            self._draw_risk_bar(0, GRAY)
        else:
            if self.risk_value is not None:
                self.risk_value.configure(text="--", fg=GRAY)
            if self.risk_status is not None:
                self.risk_status.configure(text="SESSION INACTIVE", fg=GRAY)
            if self.risk_reason is not None:
                self.risk_reason.configure(text="")
            self._draw_risk_bar(0, GRAY)

        # Driver view metrics (fatigue + attention contributors).
        if decision is not None:
            severity = decision.severity
            color = SEVERITY_COLORS.get(severity, GRAY)
            evidence = decision.evidence
            if "blink_count" in self._driver_metrics:
                self._driver_metrics["blink_count"].configure(text=str(evidence.blink_count))
            br = getattr(evidence, "blink_rate", None)
            if "blink_rate" in self._driver_metrics:
                self._driver_metrics["blink_rate"].configure(
                    text=f"{br:.1f}/min" if br is not None else "N/A"
                )
            if "yawn_count" in self._driver_metrics:
                self._driver_metrics["yawn_count"].configure(text=str(evidence.yawn_count))
            yf = getattr(evidence, "recent_yawn_frequency", None)
            if "yawn_freq" in self._driver_metrics:
                self._driver_metrics["yawn_freq"].configure(
                    text=f"{yf:.1f}/min" if yf is not None else "N/A"
                )
            eye_label = eye_state.eye if eye_state else "N/A"
            if "eye_status" in self._driver_metrics:
                self._driver_metrics["eye_status"].configure(text=eye_label)
            closure = getattr(evidence, "current_eye_closure_seconds", 0.0)
            if "eye_closure" in self._driver_metrics:
                self._driver_metrics["eye_closure"].configure(text=f"{closure:.2f}s")
            if "driving_time" in self._driver_metrics:
                self._driver_metrics["driving_time"].configure(
                    text=self.session_manager.driving_time_text()
                )
            if attention is not None and attention.face_ok:
                if "fatigue" in self._driver_metrics:
                    self._driver_metrics["fatigue"].configure(
                        text=f"{int(round(attention.fatigue_risk))}"
                    )
                if "distract" in self._driver_metrics:
                    self._driver_metrics["distract"].configure(
                        text=f"{int(round(attention.distraction_risk))}"
                    )
                if "gaze" in self._driver_metrics:
                    self._driver_metrics["gaze"].configure(
                        text=str(attention.head_direction or "forward").upper()
                    )
                overall = int(round(min(100.0, max(0.0, attention.combined_risk))))
                if "risk" in self._driver_metrics:
                    self._driver_metrics["risk"].configure(text=f"{int(round(attention.alertness_score))}/100")
                if "status" in self._driver_metrics:
                    self._driver_metrics["status"].configure(text=attention.level, fg=color)
            else:
                risk = int(round(decision.smoothed_risk)) if decision.smoothed_risk is not None else 0
                if "fatigue" in self._driver_metrics:
                    self._driver_metrics["fatigue"].configure(text=f"{risk}")
                if "distract" in self._driver_metrics:
                    self._driver_metrics["distract"].configure(text="--")
                if "gaze" in self._driver_metrics:
                    self._driver_metrics["gaze"].configure(text="--")
                if "risk" in self._driver_metrics:
                    self._driver_metrics["risk"].configure(text=f"{risk}/100")
                if "status" in self._driver_metrics:
                    self._driver_metrics["status"].configure(
                        text=severity.replace("_", " "), fg=color
                    )

        # Context panel.
        if self.ctx_time is not None:
            self.ctx_time.configure(text=f"Current Time: {time_ctx.time_text}")
        night = "ACTIVE" if time_ctx.night_context_active else "INACTIVE"
        if self.ctx_night is not None:
            self.ctx_night.configure(
                text=f"Night: {night}", fg=YELLOW if time_ctx.night_context_active else FG
            )
        if self.ctx_duration is not None:
            self.ctx_duration.configure(
                text=f"Driving Duration: {self.session_manager.driving_time_text()}"
            )
        location_ctx = data.get("location")
        if self.ctx_location is not None:
            if location_ctx is not None:
                self.ctx_location.configure(
                    text=format_location_summary(location_ctx).replace("\n", " ")
                )
            else:
                self.ctx_location.configure(text="Location: Unavailable")
        weather_ctx = data.get("weather")
        if self.ctx_weather is not None:
            if weather_ctx is not None:
                self.ctx_weather.configure(
                    text=format_weather_summary(weather_ctx).replace("\n", " ")
                )
            else:
                self.ctx_weather.configure(text="Weather: Unavailable")
        carla_display = format_carla_context(data.get("carla"))
        if self.ctx_carla_vehicle is not None:
            self.ctx_carla_vehicle.configure(text=f"Vehicle: {carla_display['vehicle']}")
        if self.ctx_carla_speed is not None:
            self.ctx_carla_speed.configure(text=f"Speed: {carla_display['speed']}")
        if self.ctx_carla_weather is not None:
            self.ctx_carla_weather.configure(text=f"Weather: {carla_display['weather']}")
        if self.ctx_carla_time is not None:
            self.ctx_carla_time.configure(text=f"Time: {carla_display['time']}")
        if self.ctx_carla_zone is not None:
            self.ctx_carla_zone.configure(
                text=f"CARLA Zone: {carla_display.get('carla_zone', '--')}"
            )
        if self.ctx_carla_position is not None:
            self.ctx_carla_position.configure(
                text=f"CARLA Position: {carla_display.get('carla_position', '--')}"
            )
        if self.ctx_carla_zone_risk is not None:
            self.ctx_carla_zone_risk.configure(
                text=f"Zone Risk: {carla_display.get('zone_risk', '--')}"
            )
        if self.ctx_carla_session is not None:
            self.ctx_carla_session.configure(text=f"Driving Session: {carla_display['session']}")
        if getattr(self, "ctx_zone_event", None) is not None:
            zone_event_text = data.get("zone_event") or ""
            if zone_event_text:
                self.ctx_zone_event.configure(text=zone_event_text)
                self.ctx_zone_event.pack(anchor="w", padx=10, pady=0)
            else:
                self.ctx_zone_event.pack_forget()

        # Alertness trend / event table are driven by the recorded session model.
        self._update_alertness_history_view()

        # Alert overlay (uses combined attention risk when available).
        self._update_alert_overlay(alert_result, decision, attention)

    def _risk_color(self, score: float | None) -> str:
        """Color a fatigue score by severity band (uses the authoritative score)."""
        if score is None:
            return GRAY
        if score >= 70:
            return RED
        if score >= 50:
            return ORANGE
        if score >= 35:
            return AMBER
        if score >= 20:
            return YELLOW
        return GREEN

    def _draw_risk_bar(self, risk: int, color: str) -> None:
        if self.risk_bar is None:
            return
        self.risk_bar.delete("all")
        w = self.risk_bar.winfo_width() or 200
        h = 14
        self.risk_bar.create_rectangle(2, 2, w - 2, h - 2, fill="#2a3040", outline="")
        fill_w = max(0, int((w - 4) * min(1.0, risk / 100.0)))
        self.risk_bar.create_rectangle(2, 2, 2 + fill_w, h - 2, fill=color, outline="")

    def _update_risk_breakdown(self, decision) -> None:
        if self.trend_canvas is None or self.rb_table is None:
            return

        for child in self.rb_table.winfo_children():
            child.destroy()
        self.trend_canvas.delete("all")

        # Both TREND and TABLE use the same underlying session history.
        # Plot the complete session history (not just the last 80 samples).
        history = self.session_manager.stats.risk_history
        samples = history

        if self._trend_mode == "TABLE":
            self.trend_canvas.pack_forget()
            self.rb_table_container.pack(fill="both", expand=True)
            # Use the latest available sample from history (or current decision
            # if no history yet) so TABLE matches TREND.
            if samples:
                latest = samples[-1]
                rows = [
                    ("FACTOR", "RISK"),
                    ("Eye", f"{latest.eye_risk:.0f}" if latest.eye_risk is not None else "N/A"),
                    ("Blink", f"{latest.blink_risk:.0f}" if latest.blink_risk is not None else "N/A"),
                    ("Yawn", f"{latest.yawn_risk:.0f}" if latest.yawn_risk is not None else "N/A"),
                    ("Night", f"{latest.night_risk:.0f}" if latest.night_risk is not None else "N/A"),
                    ("Driving Duration", f"{latest.duration_risk:.0f}" if latest.duration_risk is not None else "N/A"),
                ]
            else:
                breakdown = decision.breakdown if decision is not None else {}
                rows = [
                    ("FACTOR", "RISK"),
                    ("Eye", f"{breakdown.get('eye_risk', 0):.0f}"),
                    ("Blink", f"{breakdown.get('blink_risk', 0):.0f}"),
                    ("Yawn", f"{breakdown.get('yawn_risk', 0):.0f}"),
                    ("Night", f"{breakdown.get('night_risk', 0):.0f}"),
                    ("Driving Duration", f"{breakdown.get('duration_risk', 0):.0f}"),
                ]
            for i, (label, val) in enumerate(rows):
                r_text = tk.Label(
                    self.rb_table, text=label, font=FONT_BODY, bg=BG_CARD,
                    fg=ACCENT if i == 0 else FG, anchor="w"
                )
                r_val = tk.Label(
                    self.rb_table, text=val, font=FONT_BODY, bg=BG_CARD,
                    fg=FG_BRIGHT if i == 0 else FG, anchor="e"
                )
                r_text.grid(row=i, column=0, sticky="ew", padx=(0, 10), pady=2)
                r_val.grid(row=i, column=1, sticky="e", padx=(0, 10), pady=2)
            self.rb_table.columnconfigure(0, weight=1)
            self.rb_table.columnconfigure(1, weight=0)
        else:
            self.rb_table_container.pack_forget()
            self.trend_canvas.pack(fill="both", expand=True)

            # Current factor values (from latest history sample or decision).
            if samples:
                latest = samples[-1]
                factors = [
                    ("Eye", latest.eye_risk if latest.eye_risk is not None else 0.0, "eye_risk"),
                    ("Blink", latest.blink_risk if latest.blink_risk is not None else 0.0, "blink_risk"),
                    ("Yawn", latest.yawn_risk if latest.yawn_risk is not None else 0.0, "yawn_risk"),
                    ("Night", latest.night_risk if latest.night_risk is not None else 0.0, "night_risk"),
                    ("Duration", latest.duration_risk if latest.duration_risk is not None else 0.0, "duration_risk"),
                ]
            else:
                breakdown = decision.breakdown if decision is not None else {}
                factors = [
                    ("Eye", breakdown.get("eye_risk", 0), "eye_risk"),
                    ("Blink", breakdown.get("blink_risk", 0), "blink_risk"),
                    ("Yawn", breakdown.get("yawn_risk", 0), "yawn_risk"),
                    ("Night", breakdown.get("night_risk", 0), "night_risk"),
                    ("Duration", breakdown.get("duration_risk", 0), "duration_risk"),
                ]
            factor_y = 10
            for label, val, key in factors:
                row_y = factor_y
                bar_x = 90
                bar_w = 200
                bar_color = FACTOR_COLORS.get(key, ACCENT)
                self.trend_canvas.create_text(10, row_y, anchor="nw", text=label, fill=FG_MUTED, font=FONT_SMALL)
                self.trend_canvas.create_rectangle(bar_x, row_y + 2, bar_x + bar_w, row_y + 12, fill="#2a3040", outline="")
                self.trend_canvas.create_rectangle(bar_x, row_y + 2, bar_x + int(bar_w * min(1.0, val / 100.0)), row_y + 12, fill=bar_color, outline="")
                self.trend_canvas.create_text(bar_x + bar_w + 10, row_y, anchor="nw", text=f"{val:.0f}", fill=FG, font=FONT_SMALL)
                factor_y += 18

            if len(samples) < 2:
                self.trend_canvas.create_text(10, factor_y + 6, anchor="nw", text="Collecting data...", fill=FG_MUTED, font=FONT_BODY)
                return

            # Plot each risk factor as its own historical trend line, plus the
            # overall risk. The X-axis is elapsed session time (not frame index).
            cw = max(180, self.trend_canvas.winfo_width() or 300)
            # Reserve space at the bottom for X-axis labels (~14px) and the
            # compact legend (~18px).
            ch = max(40, int(self.trend_canvas.cget("height")) - factor_y - 12 - 30)

            start_mono = self.session_manager.start_mono_s
            last_ts = samples[-1].timestamp_s
            max_elapsed = max(0.0, last_ts - start_mono)
            total_span = max(1.0, max_elapsed)
            plot_top = factor_y
            plot_bottom = factor_y + ch

            def _axis_x(ts: float) -> float:
                elapsed = max(0.0, ts - start_mono)
                return 10 + (elapsed / total_span) * cw

            def _plot_value_y(val: float, ts: float) -> float:
                return plot_bottom - (min(100.0, max(0.0, val)) / 100.0) * max(6.0, plot_bottom - plot_top - 8) - 4

            def _draw_series(series_samples, values, color, width, dot) -> None:
                """Draw contiguous segments; missing (None) values break the line."""
                seg: list[tuple[float, float]] = []
                latest_point: tuple[float, float] | None = None
                for s, val in zip(series_samples, values):
                    if val is None:
                        seg = []
                        continue
                    x = _axis_x(s.timestamp_s)
                    y = _plot_value_y(float(val), s.timestamp_s)
                    if seg:
                        self.trend_canvas.create_line(
                            seg[-1][0], seg[-1][1], x, y, fill=color, width=width
                        )
                    seg.append((x, y))
                    latest_point = (x, y)
                if latest_point is not None and dot is not None:
                    self.trend_canvas.create_oval(
                        latest_point[0] - dot, latest_point[1] - dot,
                        latest_point[0] + dot, latest_point[1] + dot,
                        fill=color, outline="",
                    )

            for key, color in FACTOR_COLORS.items():
                values = [getattr(s, key, None) for s in samples]
                _draw_series(samples, values, color, 2, 3)

            # Overall risk line (thicker, colored by severity).
            risks = [s.risk if s.risk is not None else None for s in samples]
            # Overall risk uses per-segment severity colour, so draw it
            # segment-by-segment rather than with a single colour.
            seg: list[tuple[float, float]] = []
            latest_point: tuple[float, float] | None = None
            for s, r in zip(samples, risks):
                if r is None:
                    seg = []
                    continue
                x = _axis_x(s.timestamp_s)
                y = _plot_value_y(float(r), s.timestamp_s)
                if seg:
                    color = self._risk_color(r)
                    self.trend_canvas.create_line(
                        seg[-1][0], seg[-1][1], x, y, fill=color, width=3
                    )
                seg.append((x, y))
                latest_point = (x, y)
            if latest_point is not None:
                self.trend_canvas.create_oval(
                    latest_point[0] - 3, latest_point[1] - 3,
                    latest_point[0] + 3, latest_point[1] + 3,
                    fill=self._risk_color(risks[-1] if risks[-1] is not None else 0.0),
                    outline="",
                )

            # X-axis time labels (elapsed session time, readable mm:ss / hh:mm:ss).
            def _fmt_axis(secs: float) -> str:
                secs = max(0, int(secs))
                h, rem = divmod(secs, 3600)
                m, s = divmod(rem, 60)
                if h > 0:
                    return f"{h}:{m:02d}:{s:02d}"
                return f"{m}:{s:02d}"

            if max_elapsed <= 0.0:
                tick_vals = [0.0]
            else:
                n_ticks = 4
                tick_vals = [max_elapsed * i / (n_ticks - 1) for i in range(n_ticks)]
            tick_y = plot_bottom + 8
            for t in tick_vals:
                tx = 10 + (t / total_span) * cw
                anchor = "nw" if t == 0.0 else ("ne" if t == tick_vals[-1] and len(tick_vals) > 1 else "n")
                self.trend_canvas.create_text(tx, tick_y, anchor=anchor, text=_fmt_axis(t), fill=FG_MUTED, font=("Segoe UI", 7))

            # Compact legend using the exact line colors.
            legend_items = [
                ("Eye", FACTOR_COLORS["eye_risk"]),
                ("Blink", FACTOR_COLORS["blink_risk"]),
                ("Yawn", FACTOR_COLORS["yawn_risk"]),
                ("Night", FACTOR_COLORS["night_risk"]),
                ("Driving Duration", FACTOR_COLORS["duration_risk"]),
                ("Overall", self._risk_color(risks[-1] if risks[-1] is not None else 0.0)),
            ]
            legend_y = plot_bottom + 22
            lx = 10
            for label, lcolor in legend_items:
                self.trend_canvas.create_rectangle(lx, legend_y, lx + 8, legend_y + 8, fill=lcolor, outline="")
                self.trend_canvas.create_text(lx + 12, legend_y + 4, anchor="w", text=label, fill=FG_MUTED, font=("Segoe UI", 7))
                lx += 12 + len(label) * 5 + 8

    def _update_alertness_history_view(self) -> None:
        """Render real sampled alertness data and one row per alert episode."""
        if self.trend_canvas is None or self.rb_table is None:
            return
        history = self.session_manager.stats.alertness_history
        history_len = len(history)
        event_count = len(self.session_manager.stats.alert_events)
        now_s = time.monotonic()
        mode_changed = self._last_rendered_history_mode != self._trend_mode

        if self._trend_mode == "TABLE":
            data_changed = event_count != self._last_rendered_alert_event_count
        else:
            data_changed = history_len != self._last_rendered_history_len
        if not mode_changed and not data_changed and now_s - self._last_history_render_s < 1.0:
            return

        self._last_rendered_history_len = history_len
        self._last_rendered_alert_event_count = event_count
        self._last_history_render_s = now_s
        self._last_rendered_history_mode = self._trend_mode

        if self._trend_mode == "TABLE":
            self.trend_canvas.pack_forget()
            self.rb_table_container.pack(fill="both", expand=True)
            for child in self.rb_table.winfo_children():
                child.destroy()
            headers = ("TIME", "EVENT", "SEVERITY", "PRIMARY REASON", "ALERTNESS", "ACTION")
            for col, header in enumerate(headers):
                tk.Label(self.rb_table, text=header, font=FONT_SMALL, bg=BG_CARD, fg=ACCENT, anchor="w").grid(
                    row=0, column=col, sticky="ew", padx=3, pady=2
                )
            events = self.session_manager.stats.alert_events[-8:]
            for row, event in enumerate(events, start=1):
                event_elapsed = max(0.0, event.start_time_s - self.session_manager.start_mono_s)
                event_clock = (
                    self.session_manager.stats.start_time.timestamp() + event_elapsed
                    if self.session_manager.stats.start_time is not None
                    else time.time()
                )
                stamp = time.strftime("%H:%M:%S", time.localtime(event_clock))
                values = (
                    stamp,
                    "Attention Alert",
                    event.severity,
                    event.primary_reason or "Attention risk elevated",
                    f"{event.minimum_alertness:.0f}",
                    event.action or "Stay focused",
                )
                for col, value in enumerate(values):
                    tk.Label(
                        self.rb_table,
                        text=value,
                        font=FONT_SMALL,
                        bg=BG_CARD,
                        fg=FG,
                        anchor="w",
                        justify="left",
                        wraplength=130 if col in (3, 5) else 80,
                    ).grid(row=row, column=col, sticky="ew", padx=3, pady=2)
            if not events:
                tk.Label(self.rb_table, text="No alert episodes recorded.", font=FONT_BODY, bg=BG_CARD, fg=FG_MUTED).grid(
                    row=1, column=0, columnspan=len(headers), sticky="w", padx=3, pady=8
                )
            for col in range(len(headers)):
                self.rb_table.columnconfigure(col, weight=1 if col in (3, 5) else 0)
            return

        self.trend_canvas.delete("all")
        self.rb_table_container.pack_forget()
        self.trend_canvas.pack(fill="both", expand=True)
        self.trend_canvas.create_text(10, 8, anchor="nw", text="ATTENTION TREND", fill=FG_MUTED, font=FONT_SMALL)
        if len(history) < 2:
            self.trend_canvas.create_text(10, 34, anchor="nw", text="CALIBRATING / COLLECTING DATA...", fill=FG_MUTED, font=FONT_BODY)
            return

        # Keep canvas work bounded as a session grows. The complete history
        # remains available to the session model; only the visual rendering is
        # evenly downsampled so Tk never has to recreate thousands of objects
        # on the UI thread.
        if len(history) > MAX_RENDERED_HISTORY_POINTS:
            last_index = len(history) - 1
            step = last_index / (MAX_RENDERED_HISTORY_POINTS - 1)
            render_history = [
                history[round(index * step)]
                for index in range(MAX_RENDERED_HISTORY_POINTS)
            ]
        else:
            render_history = history

        cw = max(180, self.trend_canvas.winfo_width() or 400)
        ch = max(60, int(self.trend_canvas.cget("height")) or 150)
        left = 46
        top = 30
        width = max(120, cw - left - 16)
        height = max(40, ch - top - 34)
        start = render_history[0].timestamp_s
        span = max(1.0, render_history[-1].timestamp_s - start)

        def _x(ts: float) -> float:
            return left + ((ts - start) / span) * width

        def _y(val: float) -> float:
            return top + (1.0 - min(100.0, max(0.0, float(val))) / 100.0) * height

        # Axes.
        self.trend_canvas.create_line(left, top, left, top + height, fill="#3a4254")
        self.trend_canvas.create_line(
            left, top + height, left + width, top + height, fill="#3a4254"
        )

        # Y-axis labels (0-100) — one shared scale for all metrics.
        for value in (100, 75, 50, 25, 0):
            y = _y(float(value))
            self.trend_canvas.create_text(
                left - 4, y, anchor="e", text=str(value), fill=FG_MUTED,
                font=("Segoe UI", 7),
            )

        # Legend (drawn first so series lines sit above it).
        series = [
            ("Alertness", "alertness", GREEN, 3),
            ("Fatigue", "fatigue", ORANGE, 2),
            ("Distraction", "distraction", RED, 2),
            ("Context", "environmental_risk", ACCENT, 1),
        ]
        legend_x = left
        legend_y = top + height + 16
        for label, _key, color, _w in series:
            self.trend_canvas.create_rectangle(
                legend_x, legend_y, legend_x + 8, legend_y + 8, fill=color, outline=""
            )
            self.trend_canvas.create_text(
                legend_x + 12, legend_y + 4, anchor="w", text=label,
                fill=FG_MUTED, font=("Segoe UI", 7),
            )
            legend_x += 16 + len(label) * 5 + 12

        # Draw each series as broken lines (None breaks).
        for _label, key, color, width_l in series:
            pts: list[tuple[float, float]] = []
            for sample in render_history:
                val = getattr(sample, key, None)
                if val is None:
                    pts = []
                    continue
                x = _x(sample.timestamp_s)
                y = _y(float(val))
                if pts:
                    self.trend_canvas.create_line(
                        pts[-1][0], pts[-1][1], x, y, fill=color, width=width_l
                    )
                pts.append((x, y))
            if pts:
                x, y = pts[-1]
                self.trend_canvas.create_oval(
                    x - 2, y - 2, x + 2, y + 2, fill=color, outline=""
                )

    def _severity_glow_color(self, severity: str) -> str:
        """Glow/accent color for the alert popup by severity."""
        return {
            "NORMAL": GREEN,
            "LOW_FATIGUE": "#2dd4bf",  # teal — gentle green/cyan
            "ALERT": "#fbbf24",        # amber — moderate
            "WARNING": ORANGE,         # orange — stronger
            "DROWSY": RED,             # red — strongest
        }.get(severity, GRAY)

    def _clear_alert_overlay(self) -> None:
        """Destroy any active alert popup and reset its tracking state.

        Single cleanup point used on START / STOP / NEW SESSION so no stale
        popup, countdown or glow survives onto the summary/start screen.
        """
        overlay = getattr(self, "alert_overlay", None)
        if overlay is not None:
            try:
                overlay.destroy()
            except Exception:
                pass
        self.alert_overlay = None
        self._overlay_severity = None
        self._alert_countdown = None
        self._drowsy_countdown = None

    def _drain_frame_queue(self) -> None:
        """Drop any frames still queued by the (now stopped) camera worker."""
        q = getattr(self.camera_worker, "frame_queue", None)
        if q is None:
            return
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass

    def _update_alert_overlay(
        self, alert_result: AlertEscalationResult | None, decision, attention=None
    ) -> None:
        # Never show or keep an alert popup once the driving session has
        # stopped - this also prevents a poll callback from resurrecting it.
        if not getattr(self, "_running", False):
            self._clear_alert_overlay()
            return

        if not self.settings_store.get("visual_alerts_enabled", True):
            self._clear_alert_overlay()
            return

        if not alert_result or not alert_result.display_active or alert_result.display_remaining_s <= 0:
            self._clear_alert_overlay()
            return

        severity = alert_result.active_severity
        remaining = max(0, int(alert_result.display_remaining_s))
        glow = self._severity_glow_color(severity)

        # Popup geometry per severity (fits the 1024x700 minimum window).
        geometry = {
            "DROWSY": (560, 360, 500, 300, ("Segoe UI", 20, "bold"), ("Segoe UI", 15, "bold"), ("Segoe UI", 14, "bold")),
            "WARNING": (560, 320, 500, 260, ("Segoe UI", 18, "bold"), ("Segoe UI", 13, "bold"), ("Segoe UI", 12, "bold")),
            "ALERT": (480, 280, 420, 220, ("Segoe UI", 16, "bold"), ("Segoe UI", 12, "bold"), ("Segoe UI", 12, "bold")),
            "LOW_FATIGUE": (420, 240, 360, 180, ("Segoe UI", 14, "bold"), ("Segoe UI", 11, "bold"), ("Segoe UI", 11, "bold")),
        }.get(severity, (480, 280, 420, 220, ("Segoe UI", 16, "bold"), ("Segoe UI", 12, "bold"), ("Segoe UI", 12, "bold")))
        outer_w, outer_h, inner_w, inner_h, title_font, risk_font, timer_font = geometry

        title = {
            "LOW_FATIGUE": "LOW FATIGUE — PLEASE STAY ALERT",
            "ALERT": "ALERTNESS DECREASING",
            "WARNING": "WARNING — CONSIDER A BREAK",
            "DROWSY": "CRITICAL RISK — TAKE A SAFE STOP",
        }.get(severity, severity.replace("_", " "))
        if attention is not None and getattr(attention, "safe_stop", False) and severity == "DROWSY":
            title = "SAFE STOP RECOMMENDED — CRITICAL RISK"

        if self.alert_overlay is None or getattr(self, "_overlay_severity", None) != severity:
            if self.alert_overlay is not None:
                self.alert_overlay.destroy()

            # Outer panel: dark translucent-looking base with a colored border.
            self.alert_overlay = tk.Frame(
                self, bg="#0a0c12", bd=0,
                highlightthickness=2, highlightbackground=glow, highlightcolor=glow,
            )
            self.alert_overlay.place(relx=0.5, rely=0.5, anchor="center", width=outer_w, height=outer_h)
            self._overlay_severity = severity

            # Thin colored ring simulating a soft glow around the panel.
            glow_ring = tk.Frame(self.alert_overlay, bg=glow, bd=0)
            glow_ring.place(relx=0.5, rely=0.5, anchor="center", width=inner_w + 10, height=inner_h + 10)

            # Dark inner panel (layered glass-like appearance).
            panel = tk.Frame(glow_ring, bg=BG_PANEL, bd=0)
            panel.place(relx=0.5, rely=0.5, anchor="center", width=inner_w, height=inner_h)

            # Content card with generous padding.
            card = tk.Frame(panel, bg=BG_CARD, bd=0)
            card.place(relx=0.5, rely=0.5, anchor="center", width=inner_w - 8, height=inner_h - 8)

            # Top accent bar in the severity glow color.
            accent = tk.Frame(card, bg=glow, height=4)
            accent.pack(fill="x", side="top")

            # Title — wraps naturally, never clipped.
            tk.Label(
                card, text=title, font=title_font, bg=BG_CARD, fg=FG_BRIGHT,
                wraplength=inner_w - 56, justify="center",
            ).pack(pady=(20, 6), padx=28)

            risk_text = "RISK: -- / 100"
            if attention is not None and getattr(attention, "combined_risk", None) is not None:
                risk_text = (
                    f"RISK: {int(round(min(100.0, max(0.0, attention.combined_risk))))} / 100"
                )
            elif decision is not None and decision.smoothed_risk is not None:
                risk_text = f"RISK: {int(round(decision.smoothed_risk))} / 100"
            tk.Label(
                card, text=risk_text, font=risk_font, bg=BG_CARD, fg=glow,
            ).pack(pady=(4, 2))

            reason = ""
            if attention is not None:
                reason = str(getattr(attention, "primary_reason", "") or "")
                if getattr(attention, "safe_stop", False):
                    reason = f"{reason} — Take a safe stop now.".strip(" —")
            if reason:
                tk.Label(
                    card,
                    text=reason,
                    font=("Segoe UI", 10),
                    bg=BG_CARD,
                    fg=FG_MUTED,
                    wraplength=inner_w - 56,
                    justify="center",
                ).pack(pady=(2, 2), padx=20)

            countdown_label = tk.Label(
                card, text=f"TIMER: {remaining}", font=timer_font, bg=BG_CARD, fg=FG_BRIGHT,
            )
            countdown_label.pack(pady=(6, 18))

            if severity == "DROWSY":
                self._drowsy_countdown = countdown_label
            else:
                self._alert_countdown = countdown_label

            # Resize the popup to fit its (possibly wrapped) content so the
            # title, risk, reason, and timer are never clipped (PART 14).
            try:
                card.update_idletasks()
                content_h = max(1, card.winfo_reqheight())
                fit_inner_h = max(inner_h, content_h + 16)
                fit_outer_h = max(outer_h, fit_inner_h + 60)
                card.place_configure(height=fit_inner_h)
                panel.place_configure(height=fit_inner_h)
                glow_ring.place_configure(height=fit_inner_h + 10)
                self.alert_overlay.place_configure(height=fit_outer_h)
            except Exception:
                pass
        else:
            if severity == "DROWSY" and getattr(self, "_drowsy_countdown", None) is not None:
                self._drowsy_countdown.configure(text=f"TIMER: {remaining}")
            elif getattr(self, "_alert_countdown", None) is not None:
                self._alert_countdown.configure(text=f"TIMER: {remaining}")

        # Keep the single popup on top of every other widget (canvas, status
        # bar, refreshed dashboard frames) so it is never hidden or overlapped.
        if self.alert_overlay is not None:
            self.alert_overlay.lift()


def main() -> None:
    app = DrowsinessApp()
    app.mainloop()


if __name__ == "__main__":
    main()