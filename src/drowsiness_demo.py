"""
Integrated live drowsiness monitor (webcam).

Orchestrates the detection pipeline (Stages 2-7A) and the Phase 2 alert
escalation / audio architecture. Status comes only from FatigueEngine.
Alerts are handled by AlertEscalationController + AudioEngine (sound ->
voice, per-level, once per state entry). Driving duration uses demo mode by
default (real project threshold is 3 hours).
"""

from __future__ import annotations

import time

import cv2

from alert_escalation import AlertEscalationController
from attention_engine import AttentionEngine
from audio_engine import AudioEngine, AudioSettings
from distraction_monitor import DistractionMonitor
from driving_duration import (
    DEMO_DRIVING_DURATION_THRESHOLD_SECONDS,
    DRIVING_DURATION_THRESHOLD_HOURS,
    USE_DEMO_DRIVING_DURATION,
    DrivingDurationTracker,
)
from evidence_bridge import build_fatigue_evidence
from eye_features import analyze_eyes, draw_ear_eye_landmarks
from facial_landmarks import (
    ModelMissingError,
    create_face_landmarker,
    detect_landmarks,
)
from fatigue_engine import FatigueEngine
from mouth_features import analyze_mouth, draw_mar_mouth_landmarks
from temporal_eye import EyeClosureTracker
from temporal_mouth import YawnTracker
from time_context import DEMO_FORCE_LOCAL_TIME, get_time_context


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "N/A"


def _status_color(severity: str) -> tuple[int, int, int]:
    if severity in ("DROWSY", "CRITICAL"):
        return (0, 0, 255)
    if severity in ("WARNING", "HIGH"):
        return (0, 165, 255)
    if severity == "ALERT":
        return (15, 165, 255)  # amber/orange
    if severity in ("LOW_FATIGUE", "MODERATE"):
        return (0, 255, 255)
    if severity in ("NORMAL", "SAFE"):
        return (0, 200, 0)
    return (200, 200, 200)


def _draw_hud(frame, lines: list[tuple[str, tuple[int, int, int] | None]]) -> None:
    y = 28
    for text, color in lines:
        if text == "":
            y += 12
            continue
        cv2.putText(
            frame,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color if color is not None else (235, 235, 235),
            2,
            cv2.LINE_AA,
        )
        y += 24


def main() -> None:
    try:
        landmarker = create_face_landmarker()
    except ModelMissingError as exc:
        print(f"Error: {exc}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        print("Check that the camera is connected and not used by another app.")
        landmarker.close()
        return

    # Persistent objects - do NOT recreate each frame.
    eye_tracker = EyeClosureTracker()
    yawn_tracker = YawnTracker()
    session_start = time.monotonic()
    duration_tracker = DrivingDurationTracker(start_s=session_start)
    fatigue_engine = FatigueEngine()
    distraction_monitor = DistractionMonitor()
    attention_engine = AttentionEngine()
    alert_controller = AlertEscalationController()
    audio_engine = AudioEngine(AudioSettings())

    print("Integrated drowsiness demo started (attention layer + alerts).")
    print(
        f"Driving duration: "
        f"{'DEMO MODE (' + str(int(DEMO_DRIVING_DURATION_THRESHOLD_SECONDS)) + 's)' if USE_DEMO_DRIVING_DURATION else 'REAL (' + str(DRIVING_DURATION_THRESHOLD_HOURS) + 'h)'}"
    )
    if DEMO_FORCE_LOCAL_TIME:
        print(f"DEMO TIME override active: {DEMO_FORCE_LOCAL_TIME}")
    else:
        print("Using real local system time.")
    print("Press 'q' in the video window to quit.")

    start_ms = session_start * 1000.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error: Failed to read a frame from the webcam.")
                break

            now = time.monotonic()
            elapsed_s = now - session_start
            timestamp_ms = int(now * 1000.0 - start_ms)

            result = detect_landmarks(landmarker, frame, timestamp_ms)
            annotated = frame.copy()
            h, w = annotated.shape[:2]

            faces = result.face_landmarks
            num_faces = len(faces)
            face_detected = bool(faces)
            multiple_faces = num_faces > 1

            if face_detected and not multiple_faces:
                face = faces[0]
                eyes = analyze_eyes(face, w, h)
                mouth = analyze_mouth(face, w, h)
                draw_ear_eye_landmarks(annotated, face)
                draw_mar_mouth_landmarks(annotated, face)
                eye_state = eye_tracker.update(eyes.average_ear, elapsed_s)
                mouth_state = yawn_tracker.update(mouth.mar, elapsed_s)
                distraction_obs = distraction_monitor.update(face, w, h, elapsed_s)
            else:
                # Abort in-progress eye/mouth events; do not invent closures/yawns.
                eye_state = eye_tracker.update(None, elapsed_s)
                mouth_state = yawn_tracker.update(None, elapsed_s)
                distraction_monitor.reset()
                distraction_obs = None

            time_ctx = get_time_context()
            driving = duration_tracker.update(now)

            # Only run fatigue analysis with exactly one face.
            # No face / multiple faces pause analysis safely.
            decision = None
            attention = None
            alert_result = None
            if face_detected and not multiple_faces:
                evidence = build_fatigue_evidence(
                    eye=eye_state,
                    mouth=mouth_state,
                    time_ctx=time_ctx,
                    driving=driving,
                    face_detected=True,
                )
                decision = fatigue_engine.update(evidence, elapsed_s)
                fatigue_risk = (
                    decision.smoothed_risk
                    if decision.smoothed_risk is not None
                    else float(decision.fatigue_score or 0)
                )
                attention = attention_engine.update(
                    fatigue_risk=fatigue_risk,
                    distraction=distraction_obs,
                    now_s=elapsed_s,
                    face_ok=True,
                )
                alert_result = alert_controller.update(
                    attention.alert_severity,
                    elapsed_s,
                    reason=attention.primary_reason,
                    contributing_factors=(
                        attention.distraction_primary,
                        "fatigue" if attention.fatigue_score >= 50.0 else "stable",
                    ),
                    alertness_score=attention.alertness_score,
                    fatigue_score=attention.fatigue_score,
                    distraction_score=attention.distraction_score,
                )
                if alert_result.should_play_audio:
                    audio_engine.play_alert(
                        alert_result.audio_severity,
                        reason=attention.primary_reason,
                        contributing_factors=(
                            attention.distraction_primary,
                            "fatigue" if attention.fatigue_score >= 50.0 else "stable",
                        ),
                    )
            else:
                fatigue_engine.reset()
                attention_engine.reset()
                alert_result = alert_controller.update("NORMAL", elapsed_s)

            severity = attention.alert_severity if attention else "NORMAL"
            level = attention.level if attention else "SAFE"
            status_col = _status_color(level)

            lines: list[tuple[str, tuple[int, int, int] | None]] = [
                ("DRIVER ATTENTION MONITOR", (255, 255, 255)),
            ]

            if USE_DEMO_DRIVING_DURATION:
                lines.append(
                    (
                        f"DEMO MODE (drive threshold {int(driving.threshold_s)}s; "
                        f"real={DRIVING_DURATION_THRESHOLD_HOURS:.1f}h)",
                        (0, 255, 255),
                    )
                )
            if time_ctx.using_forced_time:
                lines.append(("DEMO TIME (forced local clock)", (0, 255, 255)))
            if multiple_faces:
                lines.append(("MULTIPLE FACES - analysis paused", (0, 0, 255)))
            elif not face_detected:
                lines.append(("NO FACE DETECTED - analysis paused", (0, 0, 255)))

            lines.extend(
                [
                    ("", None),
                    (
                        f"Face: {'DETECTED' if face_detected else 'NOT DETECTED'}",
                        (0, 200, 0) if face_detected else (0, 0, 255),
                    ),
                    (f"EAR: {_fmt(eye_state.ear)}", None),
                    (f"Eyes: {eye_state.eye}", None),
                    (f"Blink Count: {eye_state.blink_count}", None),
                    ("", None),
                    (f"MAR: {_fmt(mouth_state.mar)}", None),
                    (f"Mouth: {mouth_state.mouth}", None),
                    (f"Yawn Count: {mouth_state.yawn_count}", None),
                    ("", None),
                    (f"Time: {time_ctx.time_text}", None),
                    (f"Context: {time_ctx.context}", None),
                    ("", None),
                    (f"Driving Time: {driving.driving_time_text}", None),
                    (f"Driving Risk: {driving.duration_risk}", None),
                    ("", None),
                    (f"Attention: {level}", status_col),
                    (f"Alert: {severity}", status_col),
                ]
            )

            if attention:
                lines.append(
                    (
                        f"Attention Risk: {attention.combined_risk:.0f} "
                        f"(F:{attention.fatigue_risk:.0f} D:{attention.distraction_risk:.0f})",
                        status_col,
                    )
                )
                lines.append((f"Reason: {attention.primary_reason}", status_col))
            elif decision:
                lines.append(
                    (
                        f"Fatigue Score: {decision.fatigue_score}",
                        status_col,
                    )
                )
            else:
                lines.append(("Attention Risk: N/A", (180, 180, 180)))
                lines.append(("Reason: N/A", (180, 180, 180)))

            if distraction_obs is not None and distraction_obs.head is not None:
                hpose = distraction_obs.head
                lines.append(
                    (
                        f"Head: {hpose.direction} "
                        f"yaw={hpose.yaw_deg:.0f} pitch={hpose.pitch_deg:.0f}",
                        None,
                    )
                )

            _draw_hud(annotated, lines)
            cv2.imshow("Drowsiness Monitor - Press q to quit", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        try:
            audio_engine.shutdown()
        except Exception:
            pass
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()
        print("Camera closed. Done.")


if __name__ == "__main__":
    main()