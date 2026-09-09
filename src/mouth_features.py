"""
Stage 5: Mouth Aspect Ratio (MAR) helpers from MediaPipe Face Landmarker.

Independent of the Stage 3/4 eye / blink modules.
Does not implement fatigue scoring or alarms.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration (starting values only — tune with your laptop camera later)
# ---------------------------------------------------------------------------

# Frame-level "mouth open enough to start yawn tracking" threshold.
# NOT claimed to be scientifically optimal for every person/camera.
MAR_YAWN_THRESHOLD = 0.60

# MediaPipe Face Landmarker lip indices used for MAR.
# Vertical pairs (upper lip -> lower lip), then mouth-corner width.
# All belong to FACE_LANDMARKS_LIPS in the current MediaPipe Tasks API.
MAR_VERTICAL_PAIRS = (
    (81, 178),
    (13, 14),
    (311, 402),
)
MAR_LEFT_CORNER = 61
MAR_RIGHT_CORNER = 291

_ALL_MAR_INDICES = (
    MAR_LEFT_CORNER,
    MAR_RIGHT_CORNER,
    *[i for pair in MAR_VERTICAL_PAIRS for i in pair],
)
_MIN_LANDMARK_COUNT = max(_ALL_MAR_INDICES) + 1


@dataclass(frozen=True)
class MouthFeatureResult:
    """Per-frame mouth metrics. Values are None when landmarks are unusable."""

    mar: float | None
    mouth_status: str  # OPEN / CLOSED / UNKNOWN (frame-level vs threshold)


def _landmark_xy(
    face_landmarks,
    index: int,
    frame_w: int,
    frame_h: int,
) -> tuple[float, float] | None:
    if index < 0 or index >= len(face_landmarks):
        return None
    lm = face_landmarks[index]
    return float(lm.x * frame_w), float(lm.y * frame_h)


def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def mouth_aspect_ratio(
    face_landmarks,
    frame_w: int,
    frame_h: int,
) -> float | None:
    """
    Compute MAR from lip landmarks.

    MAR = (v1 + v2 + v3) / (2 * mouth_width)

    where v1..v3 are vertical upper-to-lower lip distances and mouth_width
    is the distance between the outer mouth corners.

    Returns None if any required landmark is missing or width is ~0.
    """
    if face_landmarks is None or len(face_landmarks) < _MIN_LANDMARK_COUNT:
        return None

    vertical_sum = 0.0
    for top_idx, bottom_idx in MAR_VERTICAL_PAIRS:
        top = _landmark_xy(face_landmarks, top_idx, frame_w, frame_h)
        bottom = _landmark_xy(face_landmarks, bottom_idx, frame_w, frame_h)
        if top is None or bottom is None:
            return None
        vertical_sum += _euclidean(top, bottom)

    left = _landmark_xy(face_landmarks, MAR_LEFT_CORNER, frame_w, frame_h)
    right = _landmark_xy(face_landmarks, MAR_RIGHT_CORNER, frame_w, frame_h)
    if left is None or right is None:
        return None

    width = _euclidean(left, right)
    if width < 1e-6:
        return None

    return vertical_sum / (2.0 * width)


def analyze_mouth(
    face_landmarks,
    frame_w: int,
    frame_h: int,
    threshold: float = MAR_YAWN_THRESHOLD,
) -> MouthFeatureResult:
    """Compute MAR and a simple frame-level OPEN/CLOSED label."""
    mar = mouth_aspect_ratio(face_landmarks, frame_w, frame_h)
    if mar is None:
        return MouthFeatureResult(None, "UNKNOWN")
    status = "OPEN" if mar >= threshold else "CLOSED"
    return MouthFeatureResult(mar, status)


def draw_mar_mouth_landmarks(
    frame_bgr: np.ndarray,
    face_landmarks,
    color: tuple[int, int, int] = (255, 100, 255),
) -> np.ndarray:
    """Highlight the MAR landmarks so you can verify the mouth points."""
    if face_landmarks is None or len(face_landmarks) < _MIN_LANDMARK_COUNT:
        return frame_bgr

    annotated = frame_bgr
    h, w = annotated.shape[:2]

    points: dict[int, tuple[int, int]] = {}
    for idx in _ALL_MAR_INDICES:
        xy = _landmark_xy(face_landmarks, idx, w, h)
        if xy is None:
            return frame_bgr
        pt = (int(round(xy[0])), int(round(xy[1])))
        points[idx] = pt
        cv2.circle(annotated, pt, 3, color, -1, lineType=cv2.LINE_AA)

    for top_idx, bottom_idx in MAR_VERTICAL_PAIRS:
        cv2.line(annotated, points[top_idx], points[bottom_idx], color, 1, cv2.LINE_AA)

    cv2.line(
        annotated,
        points[MAR_LEFT_CORNER],
        points[MAR_RIGHT_CORNER],
        color,
        1,
        cv2.LINE_AA,
    )
    return annotated