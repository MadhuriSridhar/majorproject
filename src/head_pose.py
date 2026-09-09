"""
Head-pose / looking-away observation from existing MediaPipe face landmarks.

Uses the same Face Landmarker mesh already in the pipeline. No extra model.
Temporal activate/clear hysteresis ignores brief glances.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# MediaPipe Face Landmarker indices (same mesh as eye/mouth modules).
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

# Degrees away from forward before counting as looking away.
YAW_MILD_DEG = 25.0
YAW_STRONG_DEG = 35.0
PITCH_DOWN_DEG = 22.0
PITCH_UP_DEG = 28.0

# Brief glance ignored; sustained pose becomes active; clear gradually.
ACTIVATE_HOLD_S = 1.20
CLEAR_HOLD_S = 0.55


@dataclass(frozen=True)
class HeadPoseObservation:
    """Temporally filtered head-pose state for one update."""

    direction: str  # forward | left | right | down | up | unknown
    yaw_deg: float
    pitch_deg: float
    looking_away: bool
    confidence: float
    duration_s: float
    active: bool


def _landmark_xyz(landmarks, index: int, w: int, h: int) -> np.ndarray:
    lm = landmarks[index]
    return np.array([lm.x * w, lm.y * h, lm.z * w], dtype=np.float64)


def estimate_head_angles(
    landmarks, frame_w: int, frame_h: int
) -> tuple[float, float, float]:
    """
    Estimate yaw/pitch (degrees) and confidence [0,1].

    Positive yaw ≈ subject looking left. Positive pitch ≈ looking down.
    """
    try:
        nose = _landmark_xyz(landmarks, NOSE_TIP, frame_w, frame_h)
        chin = _landmark_xyz(landmarks, CHIN, frame_w, frame_h)
        left_eye = _landmark_xyz(landmarks, LEFT_EYE_OUTER, frame_w, frame_h)
        right_eye = _landmark_xyz(landmarks, RIGHT_EYE_OUTER, frame_w, frame_h)
    except Exception:
        return 0.0, 0.0, 0.0

    eye_mid = (left_eye + right_eye) / 2.0
    face_h = float(np.linalg.norm(chin[:2] - eye_mid[:2])) + 1e-6
    face_w = float(np.linalg.norm(left_eye[:2] - right_eye[:2])) + 1e-6

    yaw = float(np.clip(((nose[0] - eye_mid[0]) / face_w) * 70.0, -70.0, 70.0))
    pitch = float(np.clip(((nose[1] - eye_mid[1]) / face_h) * 55.0, -55.0, 55.0))
    conf = float(np.clip(min(face_w, face_h) / 80.0, 0.2, 1.0))
    return yaw, pitch, conf


def direction_from_angles(
    yaw_deg: float,
    pitch_deg: float,
    yaw_mild: float = YAW_MILD_DEG,
    yaw_strong: float = YAW_STRONG_DEG,
    pitch_down: float = PITCH_DOWN_DEG,
    pitch_up: float = PITCH_UP_DEG,
) -> str:
    if abs(yaw_deg) >= yaw_strong or abs(pitch_deg) >= max(pitch_down, pitch_up):
        if abs(yaw_deg) >= abs(pitch_deg):
            return "left" if yaw_deg > 0 else "right"
        return "down" if pitch_deg > 0 else "up"
    if abs(yaw_deg) >= yaw_mild:
        return "left" if yaw_deg > 0 else "right"
    if pitch_deg >= pitch_down:
        return "down"
    if pitch_deg <= -pitch_up:
        return "up"
    return "forward"


class HeadPoseTracker:
    """Temporal head-pose tracker with activate/clear hysteresis."""

    def __init__(
        self,
        activate_hold_s: float = ACTIVATE_HOLD_S,
        clear_hold_s: float = CLEAR_HOLD_S,
        yaw_mild: float = YAW_MILD_DEG,
        yaw_strong: float = YAW_STRONG_DEG,
        pitch_down: float = PITCH_DOWN_DEG,
        pitch_up: float = PITCH_UP_DEG,
    ) -> None:
        self.activate_hold_s = float(activate_hold_s)
        self.clear_hold_s = float(clear_hold_s)
        self.yaw_mild = float(yaw_mild)
        self.yaw_strong = float(yaw_strong)
        self.pitch_down = float(pitch_down)
        self.pitch_up = float(pitch_up)
        self.reset()

    def reset(self) -> None:
        self._active = False
        self._raw_away_since: float | None = None
        self._forward_since: float | None = None
        self._away_since: float | None = None
        self._last_yaw = 0.0
        self._last_pitch = 0.0
        self._last_conf = 0.0
        self._last_direction = "unknown"

    def update(
        self, landmarks, frame_w: int, frame_h: int, now_s: float
    ) -> HeadPoseObservation:
        if landmarks is None:
            self.reset()
            return HeadPoseObservation(
                direction="unknown",
                yaw_deg=0.0,
                pitch_deg=0.0,
                looking_away=False,
                confidence=0.0,
                duration_s=0.0,
                active=False,
            )

        yaw, pitch, conf = estimate_head_angles(landmarks, frame_w, frame_h)
        raw_dir = direction_from_angles(
            yaw, pitch, self.yaw_mild, self.yaw_strong, self.pitch_down, self.pitch_up
        )
        raw_away = raw_dir != "forward"

        if raw_away:
            self._forward_since = None
            if self._raw_away_since is None:
                self._raw_away_since = now_s
            if not self._active and (now_s - self._raw_away_since) >= self.activate_hold_s:
                self._active = True
                self._away_since = now_s
            self._last_direction = raw_dir
        else:
            self._raw_away_since = None
            if self._active:
                if self._forward_since is None:
                    self._forward_since = now_s
                elif (now_s - self._forward_since) >= self.clear_hold_s:
                    self._active = False
                    self._away_since = None
                    self._forward_since = None
            else:
                self._forward_since = None
            if not self._active:
                self._last_direction = "forward"

        self._last_yaw = yaw
        self._last_pitch = pitch
        self._last_conf = conf

        duration = 0.0
        if self._active and self._away_since is not None:
            duration = max(0.0, now_s - self._away_since)

        looking_away = self._active
        direction = self._last_direction if looking_away else "forward"
        return HeadPoseObservation(
            direction=direction,
            yaw_deg=float(yaw),
            pitch_deg=float(pitch),
            looking_away=looking_away,
            confidence=float(conf),
            duration_s=float(duration),
            active=looking_away,
        )
