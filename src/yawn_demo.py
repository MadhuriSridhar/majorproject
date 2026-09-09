"""
Stage 5 demo: real-time MAR and yawn detection.

Reuses Stage 2 FaceLandmarker helpers. Independent of Stage 4 eye tracker.
Does not replace earlier demos. Does not implement fatigue scoring.
"""

from __future__ import annotations

import time

import cv2

from facial_landmarks import (
    ModelMissingError,
    create_face_landmarker,
    detect_landmarks,
)
from mouth_features import (
    MAR_YAWN_THRESHOLD,
    analyze_mouth,
    draw_mar_mouth_landmarks,
)
from temporal_mouth import (
    MAR_CLOSE_THRESHOLD,
    MAR_OPEN_THRESHOLD,
    YAWN_MIN_DURATION_S,
    YawnTracker,
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

    tracker = YawnTracker()

    print("Yawn / MAR demo started.")
    print(
        f"MAR open/close = {MAR_OPEN_THRESHOLD:.2f}/{MAR_CLOSE_THRESHOLD:.2f} | "
        f"yawn min duration = {YAWN_MIN_DURATION_S:.2f}s | "
        f"base threshold = {MAR_YAWN_THRESHOLD:.2f}"
    )
    print("Press 'q' in the video window to quit.")

    start_s = time.monotonic()
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
                mouth = analyze_mouth(face, w, h)
                draw_mar_mouth_landmarks(annotated, face)
                state = tracker.update(mouth.mar, now_s - start_s)
            else:
                state = tracker.update(None, now_s - start_s)

            yawn_text = "YES" if state.yawn else "NO"
            lines = [
                f"MAR: {_fmt(state.mar)}",
                f"Mouth: {state.mouth}",
                f"Current Mouth Open: {state.current_mouth_open_s:.2f} s",
                f"Yawn Count: {state.yawn_count}",
                f"Yawn: {yawn_text}",
            ]

            if state.yawn:
                status_color = (0, 0, 255)
            elif state.mouth == "OPEN":
                status_color = (0, 165, 255)
            elif state.mouth == "CLOSED":
                status_color = (0, 200, 0)
            else:
                status_color = (180, 180, 180)

            for i, text in enumerate(lines):
                emphasize = text.startswith("Mouth:") or text.startswith("Yawn:")
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

            cv2.imshow("Yawn Demo - Press q to quit", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()
        print("Camera closed. Done.")


if __name__ == "__main__":
    main()