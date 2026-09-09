"""
Stage 2 demo: real-time MediaPipe facial landmark detection.

Opens the default webcam, detects a face, draws the landmark mesh,
and exits cleanly when you press 'q'.

Does not replace Stage 1 (src/webcam_test.py).
"""

from __future__ import annotations

import time

import cv2

from facial_landmarks import (
    ModelMissingError,
    create_face_landmarker,
    detect_landmarks,
    draw_landmarks,
)


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

    print("Facial landmark demo started.")
    print("Look at the camera. Press 'q' in the video window to quit.")

    start_ms = time.monotonic() * 1000.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error: Failed to read a frame from the webcam.")
                break

            # Monotonic timestamp required by MediaPipe VIDEO mode
            timestamp_ms = int(time.monotonic() * 1000.0 - start_ms)
            result = detect_landmarks(landmarker, frame, timestamp_ms)
            annotated = draw_landmarks(frame, result)

            status = "Face detected" if result.face_landmarks else "No face detected"
            cv2.putText(
                annotated,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0) if result.face_landmarks else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Facial Landmarks - Press q to quit", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()
        print("Camera closed. Done.")


if __name__ == "__main__":
    main()