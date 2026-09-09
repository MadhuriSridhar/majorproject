"""
Minimal webcam test for Stage 1.

Opens the default laptop camera with OpenCV and shows a live preview.
Press 'q' to quit and release the camera cleanly.
"""

import cv2


def main() -> None:
    # 0 = default built-in webcam on most laptops
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        print("Check that the camera is connected and not used by another app.")
        return

    print("Webcam opened. Press 'q' in the video window to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Error: Failed to read a frame from the webcam.")
            break

        cv2.imshow("Webcam Test - Press q to quit", frame)

        # waitKey(1) refreshes the window; & 0xFF keeps key codes portable
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed. Done.")


if __name__ == "__main__":
    main()
