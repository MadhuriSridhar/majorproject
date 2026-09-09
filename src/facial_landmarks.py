"""
Stage 2: MediaPipe Face Landmarker helpers.

Uses the current MediaPipe Tasks API (FaceLandmarker), not the old
mp.solutions.face_mesh API which is no longer available in MediaPipe 1.x.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    FaceLandmarksConnections,
    RunningMode,
    drawing_styles,
    drawing_utils,
)

# Project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"

# Official MediaPipe model (float16 Face Landmarker):
# https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


class ModelMissingError(FileNotFoundError):
    """Raised when the required .task model file is not on disk."""


def create_face_landmarker(
    model_path: Path | str | None = None,
    num_faces: int = 2,
) -> FaceLandmarker:
    """
    Create a FaceLandmarker for webcam / video frames (VIDEO mode).

    VIDEO mode is synchronous and beginner-friendly: call detect_for_video()
    once per frame with a rising timestamp.
    """
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    if not path.is_file():
        raise ModelMissingError(
            "Required MediaPipe model file is missing.\n"
            f"  Expected file: {path}\n"
            "  This is Google's Face Landmarker model (.task file).\n"
            "  Download it from:\n"
            f"  {MODEL_DOWNLOAD_URL}\n"
            "  Save it as models/face_landmarker.task in the project folder,\n"
            "  then run the demo again."
        )

    options = FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
        running_mode=RunningMode.VIDEO,
        num_faces=num_faces,
    )
    return FaceLandmarker.create_from_options(options)


def detect_landmarks(
    landmarker: FaceLandmarker,
    frame_bgr: np.ndarray,
    timestamp_ms: int,
):
    """
    Run face landmark detection on one BGR OpenCV frame.

    Returns a FaceLandmarkerResult. If no face is found, face_landmarks
    is an empty list (safe to check / draw over).
    """
    # MediaPipe expects RGB; OpenCV camera frames are BGR.
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    return landmarker.detect_for_video(mp_image, timestamp_ms)


def draw_landmarks(frame_bgr: np.ndarray, result) -> np.ndarray:
    """
    Draw the face mesh on a copy of the frame.

    If no face is detected, returns the original frame unchanged.
    """
    if not result.face_landmarks:
        return frame_bgr

    annotated = frame_bgr.copy()

    for face_landmarks in result.face_landmarks:
        # Light mesh
        drawing_utils.draw_landmarks(
            image=annotated,
            landmark_list=face_landmarks,
            connections=FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        # Stronger face contours (eyes, lips, oval, etc.)
        drawing_utils.draw_landmarks(
            image=annotated,
            landmark_list=face_landmarks,
            connections=FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
        )

    return annotated