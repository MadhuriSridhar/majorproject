"""
Stage 4 demo: blink counting and prolonged eye-closure detection.

Reuses Stage 2/3 helpers. Does not replace earlier demos.
Does not implement fatigue scoring or alarms.
"""

from __future__ import annotations

import time

import cv2

from eye_features import analyze_eyes, draw_ear_eye_landmarks
from facial_landmarks import (
    ModelMissingError,
    create_face_landmarker,
    detect_landmarks,
)
from temporal_eye import (
    EAR_CLOSE_THRESHOLD,
    EAR_OPEN_THRESHOLD,
    MAX_BLINK_DURATION_S,
    MIN_BLINK_DURATION_S,
    PROLONGED_CLOSURE_S,
    EyeClosureTracker,
)


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "N/A"


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

    tracker = EyeClosureTracker()

    print("Blink / prolonged-closure demo started.")
    print(
        f"EAR close/open = {EAR_CLOSE_THRESHOLD:.2f}/{EAR_OPEN_THRESHOLD:.2f} | "
        f"blink = {MIN_BLINK_DURATION_S:.2f}-{MAX_BLINK_DURATION_S:.2f}s | "
        f"prolonged >= {PROLONGED_CLOSURE_S:.2f}s"
    )
    print("Press 'q' in the video window to quit.")

    start_s = time.monotonic()
    # MediaPipe VIDEO mode still needs an integer millisecond stamp.
    start_ms = start_s * 1000.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error: Failed to read a frame from the webcam.")
                break

            now_s = time.monotonic()
            timestamp_ms = int(now_s * 1000.0 - start_ms)
            result = detect_landmarks(landmarker, frame, timestamp_ms)

            annotated = frame.copy()
            h, w = annotated.shape[:2]

            if result.face_landmarks:
                face = result.face_landmarks[0]
                eyes = analyze_eyes(face, w, h)
                draw_ear_eye_landmarks(annotated, face)
                state = tracker.update(eyes.average_ear, now_s - start_s)
            else:
                state = tracker.update(None, now_s - start_s)

            prolonged_text = "YES" if state.prolonged_closure else "NO"
            lines = [
                f"EAR: {_fmt(state.ear)}",
                f"Eye: {state.eye}",
                f"Blink Count: {state.blink_count}",
                f"Current Closure: {state.current_closure_s:.2f} s",
                f"Prolonged Closure: {prolonged_text}",
            ]

            if state.prolonged_closure:
                status_color = (0, 0, 255)
            elif state.eye in ("CLOSING", "CLOSED"):
                status_color = (0, 165, 255)
            elif state.eye == "OPEN":
                status_color = (0, 200, 0)
            else:
                status_color = (180, 180, 180)

            for i, text in enumerate(lines):
                emphasize = text.startswith("Eye:") or text.startswith("Prolonged")
                color = status_color if emphasize else (240, 240, 240)
                cv2.putText(
                    annotated,
                    text,
                    (16, 30 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Blink Demo - Press q to quit", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()
        print("Camera closed. Done.")


if __name__ == "__main__":
    main()