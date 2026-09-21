"""
Stage 7B: Map live tracker/context outputs into FatigueEvidence.

No scoring here — FatigueEngine remains the only decision source.
"""

from __future__ import annotations

from driving_duration import DrivingDurationResult
from fatigue_engine import FatigueEvidence
from temporal_eye import TemporalEyeState
from temporal_mouth import TemporalMouthState
from time_context import TimeContextResult


def build_fatigue_evidence(
    eye: TemporalEyeState,
    mouth: TemporalMouthState,
    time_ctx: TimeContextResult,
    driving: DrivingDurationResult,
    face_detected: bool | None = None,
    head_down: bool = False,
    head_turned: bool = False,
    eyes_not_visible: bool = False,
    mouth_not_visible: bool = False,
) -> FatigueEvidence:
    """
    Convert current module outputs into FatigueEngine input.

    Uses current-state / current-event flags only (no new scoring).

    Face presence is decided ONLY by the detector signal when provided. Missing
    eye/mouth landmarks do NOT imply "no face": a turned head can hide the
    eyes/mouth while the face/head is still present (PARTIAL observation).
    """
    resolved_face_detected = face_detected
    if resolved_face_detected is None:
        # Do not infer "no face" from unavailable eye/mouth subfeatures.
        resolved_face_detected = True

    return FatigueEvidence(
        prolonged_eye_closure=bool(eye.prolonged_closure),
        current_eye_closure_seconds=float(eye.current_closure_s),
        blink_count=int(eye.blink_count),
        blink_rate=eye.blink_rate,
        recent_blink_rate=eye.recent_blink_rate,
        yawn_detected=bool(mouth.yawn),
        yawn_count=int(mouth.yawn_count),
        yawn_frequency=mouth.yawn_frequency,
        recent_yawn_frequency=mouth.recent_yawn_frequency,
        yawn_duration_s=mouth.yawn_duration_s,
        night_context=bool(time_ctx.night_context_active),
        driving_duration_risk=bool(driving.threshold_reached),
        driving_duration_seconds=float(driving.elapsed_s),
        face_detected=bool(resolved_face_detected),
        head_down=bool(head_down),
        head_turned=bool(head_turned),
        eyes_not_visible=bool(eyes_not_visible),
        mouth_not_visible=bool(mouth_not_visible),
    )