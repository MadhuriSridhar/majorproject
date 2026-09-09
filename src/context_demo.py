"""
Stage 6 demo: night/day context + continuous driving-duration tracking.

Independent of vision Stages 1-5. No fatigue score / DROWSY classification.
"""

from __future__ import annotations

import time

import cv2

from driving_duration import (
    DEMO_DRIVING_DURATION_THRESHOLD_SECONDS,
    DRIVING_DURATION_THRESHOLD_HOURS,
    USE_DEMO_DRIVING_DURATION,
    DrivingDurationTracker,
)
from time_context import DEMO_FORCE_LOCAL_TIME, get_time_context


def main() -> None:
    # Optional: show a blank OpenCV window so 'q' works like other demos.
    # Context modules do not require the webcam.
    window = "Context Demo - Press q to quit"
    canvas = None

    session_start = time.monotonic()
    duration_tracker = DrivingDurationTracker(start_s=session_start)

    print("Stage 6 context demo started.")
    if DEMO_FORCE_LOCAL_TIME:
        print(f"Forced local time override active: {DEMO_FORCE_LOCAL_TIME}")
    else:
        print("Using real local system time.")
        print(
            "To verify NIGHT without changing the PC clock, set "
            "DEMO_FORCE_LOCAL_TIME in src/time_context.py "
            '(e.g. "23:15:32") and rerun.'
        )

    if USE_DEMO_DRIVING_DURATION:
        print(
            f"DEMO driving-duration mode ON "
            f"(threshold = {DEMO_DRIVING_DURATION_THRESHOLD_SECONDS:.0f}s). "
            "This is NOT the real 3-hour project threshold."
        )
    else:
        print(
            f"Real driving-duration mode "
            f"(threshold = {DRIVING_DURATION_THRESHOLD_HOURS:.1f} hours)."
        )
    print("Press 'q' in the window to quit.")

    try:
        while True:
            now = time.monotonic()
            time_ctx = get_time_context()
            drive = duration_tracker.update(now)

            # Stable black frame for overlay text only.
            canvas = canvas if canvas is not None else _blank_frame()
            frame = canvas.copy()

            lines = [
                f"Current Time: {time_ctx.time_text}",
                f"Time Context: {time_ctx.context}",
                f"Night Context Active: {'YES' if time_ctx.night_context_active else 'NO'}",
                "",
                f"Driving Time: {drive.driving_time_text}",
                f"Driving Duration Risk: {drive.duration_risk}",
            ]

            if time_ctx.using_forced_time:
                lines.insert(0, "TIME OVERRIDE ACTIVE (demo)")

            if drive.demo_mode:
                lines.append(
                    f"DEMO MODE: duration threshold = {drive.threshold_s:.0f}s "
                    f"(real project threshold = {DRIVING_DURATION_THRESHOLD_HOURS:.1f}h)"
                )
            else:
                lines.append(
                    f"Real mode: duration threshold = "
                    f"{DRIVING_DURATION_THRESHOLD_HOURS:.1f}h"
                )

            for i, text in enumerate(lines):
                if not text:
                    continue
                if "NIGHT" in text and "Context" in text:
                    color = (0, 165, 255)
                elif "ELEVATED" in text:
                    color = (0, 0, 255)
                elif "DEMO" in text or "OVERRIDE" in text:
                    color = (0, 255, 255)
                else:
                    color = (240, 240, 240)
                cv2.putText(
                    frame,
                    text,
                    (20, 40 + i * 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(window, frame)
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        print("Context demo closed. Done.")


def _blank_frame(width: int = 900, height: int = 360):
    import numpy as np

    return np.zeros((height, width, 3), dtype=np.uint8)


if __name__ == "__main__":
    main()