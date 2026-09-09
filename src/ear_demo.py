"""
Stage 3 demo: real-time Eye Aspect Ratio (EAR) and OPEN/CLOSED status.

Reuses Stage 2 FaceLandmarker helpers. Does not replace Stage 1 or Stage 2 demos.
Does not implement blink counting or drowsiness alerts.
"""

from __future__ import annotations

import time

import cv2

from eye_features import (
    EAR_CLOSED_THRESHOLD,
    analyze_eyes,
    draw_ear_eye_landmarks,
)
from facial_landmarks import (
    ModelMissingError,
    create_face_landmarker,
    detect_landmarks,
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

    print("EAR demo started.")
    print(f"Initial EAR_CLOSED_THRESHOLD = {EAR_CLOSED_THRESHOLD} (not yet tuned).")
    print("Press 'q' in the video window to quit.")

    start_ms = time.monotonic() * 1000.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error: Failed to read a frame from the webcam.")
                break

            timestamp_ms = int(time.monotonic() * 1000.0 - start_ms)
            result = detect_landmarks(landmarker, frame, timestamp_ms)

            annotated = frame.copy()
            h, w = annotated.shape[:2]

            if result.face_landmarks:
                face = result.face_landmarks[0]
                eyes = analyze_eyes(face, w, h)
                draw_ear_eye_landmarks(annotated, face)

                lines = [
                    f"Left EAR:  {_fmt(eyes.left_ear)}",
                    f"Right EAR: {_fmt(eyes.right_ear)}",
                    f"Avg EAR:   {_fmt(eyes.average_ear)}",
                    f"Eyes: {eyes.eye_status}",
                    f"Threshold: {EAR_CLOSED_THRESHOLD:.2f} (initial)",
                ]
                status_color = {
                    "OPEN": (0, 200, 0),
                    "CLOSED": (0, 0, 255),
                    "UNKNOWN": (0, 165, 255),
                }.get(eyes.eye_status, (200, 200, 200))
            else:
                lines = [
                    "Left EAR:  N/A",
                    "Right EAR: N/A",
                    "Avg EAR:   N/A",
                    "Eyes: UNKNOWN (no face)",
                    f"Threshold: {EAR_CLOSED_THRESHOLD:.2f} (initial)",
                ]
                status_color = (0, 165, 255)

            y = 30
            for i, text in enumerate(lines):
                color = status_color if "Eyes:" in text else (240, 240, 240)
                cv2.putText(
                    annotated,
                    text,
                    (16, y + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("EAR Demo - Press q to quit", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()
        print("Camera closed. Done.")


if __name__ == "__main__":
    main()