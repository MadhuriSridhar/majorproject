"""
Stage 3: Eye Aspect Ratio (EAR) and basic OPEN/CLOSED eye-state helpers.

Reuses MediaPipe FaceLandmarker landmarks from facial_landmarks.py.
Does not implement blink counting, prolonged closure, or drowsiness alerts.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Initial frame-level threshold so the demo can show OPEN vs CLOSED.
# This is NOT a tuned/optimal value for every person or camera.
# We will adjust EAR_CLOSED_THRESHOLD later using your laptop webcam.
EAR_CLOSED_THRESHOLD = 0.21

# Six-point eye contours used by the classic EAR formula
# (Soukupova & Cech). Indices match MediaPipe Face Landmarker topology
# (same mesh as the older Face Mesh 468/478-point model).
#
# Ordering for each eye: outer corner, upper, upper, inner corner, lower, lower
# Names follow the subject's anatomy (person's left / right), which matches
# MediaPipe's FACE_LANDMARKS_LEFT_EYE / FACE_LANDMARKS_RIGHT_EYE groups.
LEFT_EYE_EAR_INDICES = (362, 385, 387, 263, 373, 380)
RIGHT_EYE_EAR_INDICES = (33, 160, 158, 133, 153, 144)

_ALL_EAR_INDICES = LEFT_EYE_EAR_INDICES + RIGHT_EYE_EAR_INDICES
_MIN_LANDMARK_COUNT = max(_ALL_EAR_INDICES) + 1  # need index 387 inclusive


@dataclass(frozen=True)
class EyeFeatureResult:
    """Per-frame eye metrics. Values are None when landmarks are unusable."""

    left_ear: float | None
    right_ear: float | None
    average_ear: float | None
    eye_status: str  # "OPEN", "CLOSED", or "UNKNOWN"


def _landmark_xy(
    face_landmarks,
    index: int,
    frame_w: int,
    frame_h: int,
) -> tuple[float, float] | None:
    """Convert one normalized landmark to pixel (x, y)."""
    if index < 0 or index >= len(face_landmarks):
        return None
    lm = face_landmarks[index]
    return float(lm.x * frame_w), float(lm.y * frame_h)


def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def eye_aspect_ratio(
    face_landmarks,
    eye_indices: tuple[int, int, int, int, int, int],
    frame_w: int,
    frame_h: int,
) -> float | None:
    """
    Compute EAR for one eye.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Returns None if points are missing or the eye width is ~0.
    """
    if len(eye_indices) != 6:
        return None

    points: list[tuple[float, float]] = []
    for idx in eye_indices:
        xy = _landmark_xy(face_landmarks, idx, frame_w, frame_h)
        if xy is None:
            return None
        points.append(xy)

    p1, p2, p3, p4, p5, p6 = points
    vertical_1 = _euclidean(p2, p6)
    vertical_2 = _euclidean(p3, p5)
    horizontal = _euclidean(p1, p4)

    if horizontal < 1e-6:
        return None

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def classify_eye_state(average_ear: float | None, threshold: float = EAR_CLOSED_THRESHOLD) -> str:
    """
    Frame-level OPEN/CLOSED from average EAR only.

    Below threshold -> CLOSED; at/above threshold -> OPEN.
    No temporal smoothing, blink logic, or drowsiness decision.
    """
    if average_ear is None:
        return "UNKNOWN"
    if average_ear < threshold:
        return "CLOSED"
    return "OPEN"


def analyze_eyes(
    face_landmarks,
    frame_w: int,
    frame_h: int,
    threshold: float = EAR_CLOSED_THRESHOLD,
) -> EyeFeatureResult:
    """Compute left/right/average EAR and a simple OPEN/CLOSED label."""
    if face_landmarks is None or len(face_landmarks) < _MIN_LANDMARK_COUNT:
        return EyeFeatureResult(None, None, None, "UNKNOWN")

    left_ear = eye_aspect_ratio(face_landmarks, LEFT_EYE_EAR_INDICES, frame_w, frame_h)
    right_ear = eye_aspect_ratio(face_landmarks, RIGHT_EYE_EAR_INDICES, frame_w, frame_h)

    if left_ear is None and right_ear is None:
        average_ear = None
    elif left_ear is None:
        average_ear = right_ear
    elif right_ear is None:
        average_ear = left_ear
    else:
        average_ear = (left_ear + right_ear) / 2.0

    status = classify_eye_state(average_ear, threshold=threshold)
    return EyeFeatureResult(left_ear, right_ear, average_ear, status)


def draw_ear_eye_landmarks(
    frame_bgr: np.ndarray,
    face_landmarks,
    color_left: tuple[int, int, int] = (0, 255, 255),
    color_right: tuple[int, int, int] = (255, 200, 0),
) -> np.ndarray:
    """
    Highlight the 6 EAR landmarks per eye so you can verify the points.

    Left-eye EAR points: yellow; right-eye EAR points: light blue.
    Also draws the EAR contour polyline for each eye.
    """
    if face_landmarks is None or len(face_landmarks) < _MIN_LANDMARK_COUNT:
        return frame_bgr

    annotated = frame_bgr
    h, w = annotated.shape[:2]

    for indices, color in (
        (LEFT_EYE_EAR_INDICES, color_left),
        (RIGHT_EYE_EAR_INDICES, color_right),
    ):
        pixel_points: list[tuple[int, int]] = []
        for idx in indices:
            xy = _landmark_xy(face_landmarks, idx, w, h)
            if xy is None:
                pixel_points = []
                break
            pt = (int(round(xy[0])), int(round(xy[1])))
            pixel_points.append(pt)
            cv2.circle(annotated, pt, 3, color, -1, lineType=cv2.LINE_AA)

        if len(pixel_points) == 6:
            # Contour order around the eye for a simple outline
            outline = [
                pixel_points[0],  # outer
                pixel_points[1],
                pixel_points[2],
                pixel_points[3],  # inner
                pixel_points[4],
                pixel_points[5],
            ]
            cv2.polylines(
                annotated,
                [np.array(outline, dtype=np.int32)],
                isClosed=True,
                color=color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    return annotated