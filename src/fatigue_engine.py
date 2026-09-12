"""
Stage 7A: Standalone fatigue / drowsiness decision engine.

Deterministic, explainable scoring from current evidence flags.
Independent of OpenCV and MediaPipe. No alarms / UI / webcam integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Scoring weights (initial engineering values — tune during live testing)
# ---------------------------------------------------------------------------

WEIGHT_PROLONGED_EYE_CLOSURE = 70
WEIGHT_YAWN_EVENT = 15
WEIGHT_NIGHT_CONTEXT = 5
WEIGHT_DRIVING_DURATION_RISK = 10

# Blink count is tracked as input/context for later fusion, but does NOT
# add to the fatigue score (normal blinking is expected).

# ---------------------------------------------------------------------------
# Status thresholds (initial engineering values)
# ---------------------------------------------------------------------------

SAFE_SCORE_MAX = 30          # score < 30  -> SAFE
WARNING_SCORE_MAX = 60       # 30 <= score < 60 -> WARNING
# score >= WARNING_SCORE_MAX -> DROWSY

# ---------------------------------------------------------------------------
# Temporal stability (debounce / recovery)
# ---------------------------------------------------------------------------

# How long raw DROWSY evidence must persist before status becomes DROWSY,
# when prolonged eye closure is NOT already established.
DROWSY_ENTER_HOLD_S = 0.30

# How long non-DROWSY evidence must persist before leaving DROWSY.
DROWSY_EXIT_HOLD_S = 0.80

# Foundation-phase defaults for the rule-based engine.
WARNING_ENTRY = 50
WARNING_RECOVERY = 35
DROWSY_ENTRY = 70
DROWSY_RECOVERY = 50
# NOTE: WARNING persistence for *alerts* is enforced in attention_engine.py
# (WARNING_PERSISTENCE_S). The rule-based status here intentionally responds to
# score quickly; severity escalation keeps its own temporal gating.
WARNING_PERSISTENCE_SECONDS = 0.0
DROWSY_PERSISTENCE_SECONDS = 0.0
WARNING_RECOVERY_SECONDS = 0.8
DROWSY_RECOVERY_SECONDS = 0.8
LOW_FATIGUE_ENTRY = 25
ALERT_ENTRY = 35
SMOOTHING_ALPHA = 0.35

# Asymmetric risk decrease: rises normally, but brief improvement must not
# drop the displayed / severity risk immediately.
RISK_DECREASE_HOLD_S = 0.40          # hold current risk briefly after improvement starts
RISK_DECREASE_TAU_S = 1.00           # time constant for gradual decay toward raw risk

# ---------------------------------------------------------------------------
# Decaying recent-evidence window (supporting fatigue trends)
# ---------------------------------------------------------------------------

# Effective window: evidence older than ~3 minutes has negligible influence.
RECENT_EVIDENCE_WINDOW_S = 5 * 60
# Exponential decay time constant (60 seconds -> ~5% influence at 3 min).
RECENT_EVIDENCE_TAU_S = 60

# Repeated recent yawns can contribute up to 2x the single-yawn weight.
RECENT_YAWN_MAX = 2 * WEIGHT_YAWN_EVENT

# Blink anomaly thresholds (blinks per minute).
BLINK_HIGH_RATE_THRESHOLD = 18.0
BLINK_LOW_RATE_THRESHOLD = 8.0
BLINK_ANOMALY_MAX = 20.0
# Sample abnormal blink rate at most every 10 seconds.
BLINK_SAMPLE_INTERVAL_S = 10.0
# Per-sample contribution to the decaying blink evidence.
BLINK_SAMPLE_WEIGHT = 2.0


@dataclass(frozen=True)
class FatigueEvidence:
    """Current-frame / current-event evidence fed into the engine."""

    prolonged_eye_closure: bool = False
    current_eye_closure_seconds: float = 0.0
    blink_count: int = 0
    yawn_detected: bool = False
    yawn_count: int = 0
    night_context: bool = False
    driving_duration_risk: bool = False
    driving_duration_seconds: float = 0.0
    face_detected: bool = True
    head_down: bool = False
    head_turned: bool = False
    eyes_not_visible: bool = False
    mouth_not_visible: bool = False
    blink_rate: float | None = None
    recent_blink_rate: float | None = None
    recent_yawn_frequency: float | None = None
    yawn_frequency: float | None = None
    eye_closure_duration: float | None = None
    yawn_duration_s: float | None = None


@dataclass(frozen=True)
class FatigueDecision:
    """Engine output: score, smoothed status, and human-readable reasons."""

    fatigue_score: int | None
    status: str  # SAFE / WARNING / DROWSY
    reasons: tuple[str, ...]
    raw_status: str  # status from score alone (before smoothing)
    severity: str  # NORMAL / LOW_FATIGUE / ALERT / WARNING / DROWSY
    smoothed_risk: float | None
    breakdown: dict[str, float]
    evidence: FatigueEvidence = field(repr=False)


class FatigueEngine:
    """
    Stateful decision engine with light temporal smoothing.

    Score is recomputed from *current* evidence each update — it does not
    accumulate forever across frames.
    """

    def __init__(
        self,
        weight_prolonged: int = WEIGHT_PROLONGED_EYE_CLOSURE,
        weight_yawn: int = WEIGHT_YAWN_EVENT,
        weight_night: int = WEIGHT_NIGHT_CONTEXT,
        weight_driving: int = WEIGHT_DRIVING_DURATION_RISK,
        safe_score_max: int = SAFE_SCORE_MAX,
        warning_score_max: int = WARNING_SCORE_MAX,
        drowsy_enter_hold_s: float = DROWSY_ENTER_HOLD_S,
        drowsy_exit_hold_s: float = DROWSY_EXIT_HOLD_S,
        warning_entry: int = WARNING_ENTRY,
        warning_recovery: int = WARNING_RECOVERY,
        drowsy_entry: int = DROWSY_ENTRY,
        drowsy_recovery: int = DROWSY_RECOVERY,
        low_fatigue_entry: int = LOW_FATIGUE_ENTRY,
        alert_entry: int = ALERT_ENTRY,
        warning_persistence_s: float | None = None,
        drowsy_persistence_s: float | None = None,
        warning_recovery_s: float | None = None,
        drowsy_recovery_s: float | None = None,
        smoothing_alpha: float = SMOOTHING_ALPHA,
        risk_decrease_hold_s: float = RISK_DECREASE_HOLD_S,
        risk_decrease_tau_s: float = RISK_DECREASE_TAU_S,
    ) -> None:
        if not (0 <= safe_score_max <= warning_score_max):
            raise ValueError("Require 0 <= safe_score_max <= warning_score_max")

        self.weight_prolonged = weight_prolonged
        self.weight_yawn = weight_yawn
        self.weight_night = weight_night
        self.weight_driving = weight_driving
        self.safe_score_max = safe_score_max
        self.warning_score_max = warning_score_max
        self.drowsy_enter_hold_s = drowsy_enter_hold_s
        self.drowsy_exit_hold_s = drowsy_exit_hold_s
        self.warning_entry = warning_entry
        self.warning_recovery = warning_recovery
        self.drowsy_entry = drowsy_entry
        self.drowsy_recovery = drowsy_recovery
        self.low_fatigue_entry = low_fatigue_entry
        self.alert_entry = alert_entry
        self.warning_persistence_s = (
            warning_persistence_s
            if warning_persistence_s is not None
            else WARNING_PERSISTENCE_SECONDS
        )
        self.drowsy_persistence_s = (
            drowsy_persistence_s
            if drowsy_persistence_s is not None
            else max(self.drowsy_enter_hold_s, DROWSY_PERSISTENCE_SECONDS)
        )
        self.warning_recovery_s = (
            warning_recovery_s if warning_recovery_s is not None else max(self.drowsy_exit_hold_s, WARNING_RECOVERY_SECONDS)
        )
        self.drowsy_recovery_s = (
            drowsy_recovery_s if drowsy_recovery_s is not None else max(self.drowsy_exit_hold_s, DROWSY_RECOVERY_SECONDS)
        )
        self.smoothing_alpha = min(0.99, max(0.0, smoothing_alpha))
        self.risk_decrease_hold_s = max(0.0, float(risk_decrease_hold_s))
        self.risk_decrease_tau_s = max(0.05, float(risk_decrease_tau_s))

        self._severity = "NORMAL"
        self._pending_severity: str | None = None
        self._pending_since: float | None = None
        self._smoothed_risk: float | None = None
        self._risk_improve_since: float | None = None
        self._last_raw_risk: float | None = None
        self._last_smoothed_update_s: float | None = None
        self._had_face = True
        self._status = "SAFE"
        self._status_pending: str | None = None
        self._status_since: float | None = None

        # Decaying recent-evidence window (supporting fatigue trends).
        # Preserved across reset() so temporary face loss does not erase
        # session history.
        self._recent_yawn_evidence = 0.0
        self._recent_blink_evidence = 0.0
        self._last_yawn_count: int | None = None
        self._last_blink_sample_s: float | None = None
        self._last_update_s: float | None = None

    def reset(self, clear_history: bool = False) -> None:
        """Reset live state; optionally discard evidence from an old session."""
        self._severity = "NORMAL"
        self._pending_severity = None
        self._pending_since = None
        self._smoothed_risk = None
        self._risk_improve_since = None
        self._last_raw_risk = None
        self._last_smoothed_update_s = None
        self._had_face = True
        self._status = "SAFE"
        self._status_pending = None
        self._status_since = None
        if clear_history:
            self._recent_yawn_evidence = 0.0
            self._recent_blink_evidence = 0.0
            self._last_yawn_count = None
            self._last_blink_sample_s = None
            self._last_update_s = None

    def _current_blink_anomaly_magnitude(self, evidence: FatigueEvidence) -> float:
        """Magnitude of the current abnormal-blink signal (0 when normal)."""
        magnitude = 0.0
        if evidence.blink_rate is not None:
            if evidence.blink_rate > BLINK_HIGH_RATE_THRESHOLD:
                magnitude = max(
                    magnitude,
                    min(
                        BLINK_ANOMALY_MAX,
                        (evidence.blink_rate - BLINK_HIGH_RATE_THRESHOLD) * 0.8,
                    ),
                )
            elif evidence.blink_rate < BLINK_LOW_RATE_THRESHOLD:
                magnitude = max(
                    magnitude,
                    min(
                        BLINK_ANOMALY_MAX,
                        (BLINK_LOW_RATE_THRESHOLD - evidence.blink_rate) * 0.5,
                    ),
                )
        if (
            evidence.recent_blink_rate is not None
            and evidence.recent_blink_rate > BLINK_HIGH_RATE_THRESHOLD + 2.0
        ):
            magnitude = max(
                magnitude,
                min(
                    BLINK_ANOMALY_MAX,
                    (evidence.recent_blink_rate - BLINK_HIGH_RATE_THRESHOLD - 2.0) * 0.7,
                ),
            )
        return magnitude

    def _update_recent_evidence(self, evidence: FatigueEvidence, timestamp_s: float) -> None:
        """
        Update the decaying recent-evidence window.

        - Yawn events are recorded when the monotonic yawn counter increases.
        - Abnormal blink samples are recorded at most every BLINK_SAMPLE_INTERVAL_S.
        - All evidence decays exponentially with RECENT_EVIDENCE_TAU_S.
        - Face loss decays evidence (time passes) but never adds new evidence.
        """
        if self._last_update_s is None:
            self._last_update_s = timestamp_s
        dt = max(0.0, timestamp_s - self._last_update_s)
        self._last_update_s = timestamp_s

        decay = math.exp(-dt / RECENT_EVIDENCE_TAU_S)
        self._recent_yawn_evidence *= decay
        self._recent_blink_evidence *= decay

        if not evidence.face_detected:
            return

        # New yawn events (detected via the monotonic yawn counter).
        # The first frame establishes the baseline — any pre-existing
        # yawn count at engine start-up is NOT treated as recent evidence.
        if self._last_yawn_count is None:
            self._last_yawn_count = evidence.yawn_count
        elif evidence.yawn_count < self._last_yawn_count:
            self._last_yawn_count = evidence.yawn_count  # tracker was reset
        elif evidence.yawn_count > self._last_yawn_count:
            new_yawns = evidence.yawn_count - self._last_yawn_count
            self._recent_yawn_evidence += new_yawns * float(self.weight_yawn)
            self._last_yawn_count = evidence.yawn_count
        self._recent_yawn_evidence = min(RECENT_YAWN_MAX, self._recent_yawn_evidence)

        # Abnormal blink samples (sampled at most every BLINK_SAMPLE_INTERVAL_S).
        magnitude = self._current_blink_anomaly_magnitude(evidence)
        if magnitude > 0.0:
            if (
                self._last_blink_sample_s is None
                or timestamp_s - self._last_blink_sample_s >= BLINK_SAMPLE_INTERVAL_S
            ):
                self._last_blink_sample_s = timestamp_s
                sample_weight = min(BLINK_SAMPLE_WEIGHT, magnitude * 0.25)
                self._recent_blink_evidence += sample_weight
        self._recent_blink_evidence = min(
            BLINK_ANOMALY_MAX, max(0.0, self._recent_blink_evidence)
        )

    def _eyes_closed_valid(self, evidence: FatigueEvidence) -> bool:
        """True only when eye-closure evidence comes from reliably visible eyes.

        Eye-closure fatigue must come from genuine, measurable eye closure. A
        turned/down head — or eyes explicitly reported as not visible — must
        never be read as "eyes closed". These are three distinct states:
        EYES_CLOSED, EYES_NOT_VISIBLE, and FACE_NOT_DETECTED.
        """
        return bool(
            evidence.prolonged_eye_closure
            and not evidence.head_down
            and not evidence.head_turned
            and not evidence.eyes_not_visible
        )

    def compute_score(self, evidence: FatigueEvidence) -> tuple[int | None, tuple[str, ...]]:
        """Compute current-evidence score and reasons from the authoritative raw-risk breakdown."""
        if not evidence.face_detected:
            self._had_face = False
            return None, ("No face detected",)

        if not self._had_face:
            self._had_face = True
            return None, ("Face detected; resuming analysis",)

        self._had_face = True

        # Eye-closure fatigue is valid ONLY when the eyes are actually visible
        # and reliably measured. A turned/down head — or eyes explicitly marked
        # not-visible — must never be read as "eyes closed". Missing/occluded
        # eyes and a missing face are distinct from real eye closure.
        eyes_closed_valid = self._eyes_closed_valid(evidence)
        eyes_not_visible = bool(evidence.prolonged_eye_closure and not eyes_closed_valid)

        breakdown = self._risk_breakdown(evidence)
        score = int(round(min(100.0, breakdown["total_risk"])))
        reasons: list[str] = []

        if eyes_closed_valid:
            if evidence.current_eye_closure_seconds > 0:
                reasons.append(
                    f"Prolonged eye closure ({evidence.current_eye_closure_seconds:.1f}s)"
                )
            else:
                reasons.append("Prolonged eye closure")
        elif eyes_not_visible:
            # Distinguish "eyes not visible" from "eyes closed": no fatigue is
            # credited from landmarks hidden by head orientation.
            reasons.append("Eyes not visible (head orientation)")

        if evidence.blink_rate is not None and evidence.blink_rate > 0:
            reasons.append("Abnormal blink behaviour")
        elif self._recent_blink_evidence > 0:
            reasons.append("Recent abnormal blink behaviour")

        if (evidence.yawn_detected or evidence.recent_yawn_frequency) and not evidence.mouth_not_visible:
            reasons.append("Yawning detected")
        elif self._recent_yawn_evidence > 0:
            reasons.append("Recent yawning")

        if evidence.night_context:
            reasons.append("Night context")

        if evidence.driving_duration_risk:
            reasons.append("Extended driving duration")

        if not reasons:
            reasons.append("Eyes normal")

        if eyes_closed_valid and evidence.night_context and evidence.driving_duration_risk:
            reasons.append("Prolonged eye closure + night + extended driving")
        elif eyes_closed_valid and evidence.night_context:
            reasons.append("Prolonged eye closure + night context")
        elif eyes_closed_valid and evidence.driving_duration_risk:
            reasons.append("Prolonged eye closure + extended driving")

        deduped: list[str] = []
        seen: set[str] = set()
        for item in reasons:
            if item not in seen:
                deduped.append(item)
                seen.add(item)

        return score, tuple(deduped)

    def status_from_score(self, score: int | None) -> str:
        if score is None:
            return "SAFE"
        if score < self.safe_score_max:
            return "SAFE"
        if score < self.warning_score_max:
            return "WARNING"
        return "DROWSY"

    def _severity_from_risk(self, risk: float | None) -> str:
        if risk is None:
            return "NORMAL"
        if risk >= self.drowsy_entry:
            return "DROWSY"
        if risk >= self.warning_entry:
            return "WARNING"
        if risk >= self.alert_entry:
            return "ALERT"
        if risk >= self.low_fatigue_entry:
            return "LOW_FATIGUE"
        return "NORMAL"

    def _risk_breakdown(self, evidence: FatigueEvidence) -> dict[str, float]:
        if not evidence.face_detected:
            return {
                "eye_risk": 0.0,
                "blink_risk": 0.0,
                "yawn_risk": 0.0,
                "night_risk": 0.0,
                "duration_risk": 0.0,
                "total_risk": 0.0,
            }

        eye_risk = (
            float(self.weight_prolonged)
            if self._eyes_closed_valid(evidence)
            else 0.0
        )

        blink_risk = 0.0
        if evidence.blink_rate is not None and evidence.blink_rate > 0:
            blink_risk = min(20.0, max(0.0, evidence.blink_rate - 18.0) * 0.8)
        if evidence.recent_blink_rate is not None and evidence.recent_blink_rate > 0:
            blink_risk = max(blink_risk, min(20.0, max(0.0, evidence.recent_blink_rate - 20.0) * 0.7))
        # Decaying recent abnormal-blink evidence.
        blink_risk = max(blink_risk, self._recent_blink_evidence)
        blink_risk = min(BLINK_ANOMALY_MAX, max(0.0, blink_risk))

        # One genuine yawn contributes the configured weight (default 15).
        # Repeated yawns accumulate as decaying recent evidence and may grow the
        # contribution up to RECENT_YAWN_MAX, but never beyond it.
        yawn_risk = 0.0
        if evidence.yawn_detected and not evidence.mouth_not_visible:
            yawn_risk = float(self.weight_yawn)
        if (
            evidence.recent_yawn_frequency is not None
            and evidence.recent_yawn_frequency > 0
            and not evidence.mouth_not_visible
        ):
            freq_risk = min(float(self.weight_yawn), evidence.recent_yawn_frequency * 3.0)
            yawn_risk = max(yawn_risk, freq_risk)
        # Decaying recent-yawn evidence (repeated yawns strengthen support).
        yawn_risk = max(yawn_risk, self._recent_yawn_evidence)
        # A single yawn contributes weight_yawn; repeated yawns may strengthen
        # the contribution up to RECENT_YAWN_MAX, but never beyond it (and never
        # below the configured single-yawn weight).
        yawn_risk = min(max(self.weight_yawn, RECENT_YAWN_MAX), max(0.0, yawn_risk))

        night_risk = float(self.weight_night) if evidence.night_context else 0.0
        duration_risk = 0.0
        if evidence.driving_duration_risk:
            seconds = max(0.0, evidence.driving_duration_seconds)
            if seconds >= 5 * 3600:
                duration_risk = 10.0
            elif seconds >= 4 * 3600:
                duration_risk = 8.0
            elif seconds >= 3 * 3600:
                duration_risk = 4.0
            else:
                duration_risk = 0.0

        total_risk = min(100.0, eye_risk + blink_risk + yawn_risk + night_risk + duration_risk)
        return {
            "eye_risk": eye_risk,
            "blink_risk": blink_risk,
            "yawn_risk": yawn_risk,
            "night_risk": night_risk,
            "duration_risk": duration_risk,
            "total_risk": total_risk,
        }

    def _risk_from_evidence(self, evidence: FatigueEvidence) -> float | None:
        breakdown = self._risk_breakdown(evidence)
        return None if not evidence.face_detected else breakdown["total_risk"]

    def _transition_severity(
        self,
        candidate: str,
        timestamp_s: float,
        raw_risk: float | None = None,
    ) -> str:
        levels = {
            "NORMAL": 0,
            "LOW_FATIGUE": 1,
            "ALERT": 2,
            "WARNING": 3,
            "DROWSY": 4,
        }
        current_level = levels[self._severity]
        candidate_level = levels[candidate]

        if candidate == self._severity:
            self._pending_severity = None
            self._pending_since = None
            return self._severity

        if candidate_level > current_level:
            # Do not complete an escalation when live raw evidence no longer
            # supports it (held/smoothed risk alone must not promote severity
            # after a brief or sustained improvement).
            if raw_risk is not None:
                raw_level = levels[self._severity_from_risk(raw_risk)]
                if raw_level <= current_level:
                    self._pending_severity = None
                    self._pending_since = None
                    return self._severity

            if self._pending_severity != candidate:
                self._pending_severity = candidate
                self._pending_since = timestamp_s
                return self._severity

            if self._pending_since is None:
                self._pending_since = timestamp_s
            hold_s = self.drowsy_persistence_s if candidate == "DROWSY" else self.warning_persistence_s
            if timestamp_s - self._pending_since >= hold_s:
                self._severity = candidate
                self._pending_severity = None
                self._pending_since = None
            return self._severity

        if self._pending_severity != candidate:
            self._pending_severity = candidate
            self._pending_since = timestamp_s
            return self._severity

        if self._pending_since is None:
            self._pending_since = timestamp_s
        if self._severity == "DROWSY" and candidate in {"WARNING", "LOW_FATIGUE", "NORMAL"}:
            recovery_s = self.drowsy_recovery_s
        elif self._severity == "WARNING" and candidate in {"LOW_FATIGUE", "NORMAL"}:
            recovery_s = self.warning_recovery_s
        else:
            recovery_s = self.warning_recovery_s

        if timestamp_s - self._pending_since >= recovery_s:
            self._severity = candidate
            self._pending_severity = None
            self._pending_since = None
        return self._severity

    def _transition_status(self, raw_status: str, severity: str, evidence: FatigueEvidence, timestamp_s: float) -> str:
        if evidence.face_detected is False:
            self._status = "SAFE"
            self._status_pending = None
            self._status_since = None
            return "SAFE"

        if raw_status == "SAFE" and severity == "WARNING":
            if self._status == "DROWSY":
                return self._status
            if self._status == "WARNING":
                self._status_pending = None
                self._status_since = None
                return "WARNING"
            if self.warning_persistence_s <= 0.0:
                self._status = "WARNING"
                self._status_pending = None
                self._status_since = None
                return "WARNING"

            if self._status_pending != "WARNING":
                self._status_pending = "WARNING"
                self._status_since = timestamp_s
                return "SAFE"

            if self._status_since is None:
                self._status_since = timestamp_s
            if timestamp_s - self._status_since >= self.warning_persistence_s:
                self._status = "WARNING"
                self._status_pending = None
                self._status_since = None
            return self._status if self._status != "SAFE" else "SAFE"

        if raw_status == "SAFE" and severity == "DROWSY":
            # Severity may lag on held risk; raw SAFE still recovers via the
            # drowsy recovery timer instead of hard-locking forever.
            if self._status == "DROWSY":
                if self._status_pending != "SAFE":
                    self._status_pending = "SAFE"
                    self._status_since = timestamp_s
                    return "DROWSY"
                if self._status_since is None:
                    self._status_since = timestamp_s
                if timestamp_s - self._status_since >= self.drowsy_recovery_s:
                    self._status = "SAFE"
                    self._status_pending = None
                    self._status_since = None
                    return "SAFE"
                return "DROWSY"
            if self.drowsy_persistence_s <= 0.0:
                self._status = "DROWSY"
                self._status_pending = None
                self._status_since = None
                return "DROWSY"

            if self._status_pending != "DROWSY":
                self._status_pending = "DROWSY"
                self._status_since = timestamp_s
                return "SAFE"

            if self._status_since is None:
                self._status_since = timestamp_s
            if timestamp_s - self._status_since >= self.drowsy_persistence_s:
                self._status = "DROWSY"
                self._status_pending = None
                self._status_since = None
            return self._status if self._status != "SAFE" else "SAFE"

        if raw_status == "SAFE":
            if self._status == "SAFE":
                self._status_pending = None
                self._status_since = None
                return "SAFE"

            if self._status_pending != "SAFE":
                self._status_pending = "SAFE"
                self._status_since = timestamp_s
                return self._status

            if self._status_since is None:
                self._status_since = timestamp_s
            recovery_s = self.drowsy_recovery_s if self._status == "DROWSY" else self.warning_recovery_s
            if timestamp_s - self._status_since >= recovery_s:
                self._status = "SAFE"
                self._status_pending = None
                self._status_since = None
            return self._status

        if raw_status == "WARNING":
            if self._status == "DROWSY":
                return self._status
            if self._status == "WARNING":
                self._status_pending = None
                self._status_since = None
                return "WARNING"
            if self.warning_persistence_s <= 0.0:
                self._status = "WARNING"
                self._status_pending = None
                self._status_since = None
                return "WARNING"

            if self._status_pending != "WARNING":
                self._status_pending = "WARNING"
                self._status_since = timestamp_s
                return "SAFE"

            if self._status_since is None:
                self._status_since = timestamp_s
            if timestamp_s - self._status_since >= self.warning_persistence_s:
                self._status = "WARNING"
                self._status_pending = None
                self._status_since = None
            return self._status if self._status != "SAFE" else "SAFE"

        if raw_status == "DROWSY":
            if self._status == "DROWSY":
                return "DROWSY"
            if self.drowsy_persistence_s <= 0.0:
                self._status = "DROWSY"
                self._status_pending = None
                self._status_since = None
                return "DROWSY"

            if self._status_pending != "DROWSY":
                self._status_pending = "DROWSY"
                self._status_since = timestamp_s
                return "SAFE"

            if self._status_since is None:
                self._status_since = timestamp_s
            if timestamp_s - self._status_since >= self.drowsy_persistence_s:
                self._status = "DROWSY"
                self._status_pending = None
                self._status_since = None
            return self._status if self._status != "SAFE" else "SAFE"

        return "SAFE"

    def _update_smoothed_risk(self, risk: float | None, timestamp_s: float) -> float | None:
        if risk is None:
            self._smoothed_risk = None
            self._risk_improve_since = None
            self._last_raw_risk = None
            self._last_smoothed_update_s = None
            self._severity = "NORMAL"
            self._pending_severity = None
            self._pending_since = None
            return None

        raw = max(0.0, min(100.0, float(risk)))

        if self._smoothed_risk is None:
            self._smoothed_risk = raw
            self._last_raw_risk = raw
            self._last_smoothed_update_s = timestamp_s
            self._risk_improve_since = None
            return self._smoothed_risk

        current = float(self._smoothed_risk)

        # Danger rising or equal: increase normally; cancel any decrease.
        if raw >= current:
            self._smoothed_risk = current + self.smoothing_alpha * (raw - current)
            self._risk_improve_since = None
            self._last_raw_risk = raw
            self._last_smoothed_update_s = timestamp_s
            self._smoothed_risk = max(0.0, min(100.0, float(self._smoothed_risk)))
            return self._smoothed_risk

        # Raw improved (lower). If a danger signal is returning (raw rising
        # again while still below the held risk), stop decreasing immediately.
        if self._last_raw_risk is not None and raw > self._last_raw_risk:
            self._risk_improve_since = None
            self._last_raw_risk = raw
            self._last_smoothed_update_s = timestamp_s
            return current

        if self._risk_improve_since is None:
            self._risk_improve_since = timestamp_s
            self._last_raw_risk = raw
            self._last_smoothed_update_s = timestamp_s
            return current

        # Hold current risk briefly; require sustained improvement first.
        if timestamp_s - self._risk_improve_since < self.risk_decrease_hold_s:
            self._last_raw_risk = raw
            return current

        # Gradual time-based decay toward the improved raw risk.
        last_t = self._last_smoothed_update_s
        dt = max(0.0, timestamp_s - last_t) if last_t is not None else 0.0
        if dt <= 0.0:
            self._last_raw_risk = raw
            return current

        decay = math.exp(-dt / self.risk_decrease_tau_s)
        self._smoothed_risk = raw + (current - raw) * decay
        self._smoothed_risk = max(0.0, min(100.0, float(self._smoothed_risk)))
        self._last_raw_risk = raw
        self._last_smoothed_update_s = timestamp_s
        return self._smoothed_risk

    def update(self, evidence: FatigueEvidence, timestamp_s: float) -> FatigueDecision:
        """Update the engine with current evidence and a monotonic timestamp."""
        self._update_recent_evidence(evidence, timestamp_s)
        score, reasons = self.compute_score(evidence)
        raw_status = self.status_from_score(score)
        rule_risk = self._risk_from_evidence(evidence)
        smoothed_risk = self._update_smoothed_risk(rule_risk, timestamp_s)
        severity = self._transition_severity(
            self._severity_from_risk(smoothed_risk),
            timestamp_s,
            raw_risk=rule_risk,
        )
        status = self._transition_status(raw_status, severity, evidence, timestamp_s)

        breakdown = self._risk_breakdown(evidence)
        return FatigueDecision(
            fatigue_score=score,
            status=status,
            reasons=reasons,
            raw_status=raw_status,
            severity=severity,
            smoothed_risk=smoothed_risk,
            breakdown=breakdown,
            evidence=evidence,
        )

    def decide(self, evidence: FatigueEvidence, timestamp_s: float = 0.0) -> FatigueDecision:
        """Alias for update() for one-shot / stateless-style calls."""
        return self.update(evidence, timestamp_s)