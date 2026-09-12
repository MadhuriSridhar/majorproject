"""
Phase 2: Driving-session manager.

Tracks session statistics, risk history, and alert counters.
Consumes values from the existing FatigueEngine - it does NOT re-implement
detection or scoring.

The UI reads these values; the backend (FatigueEngine) remains the only
decision source.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from driving_duration import DrivingDurationTracker, format_hms

# Bounded risk-history buffer (do not store unlimited data in memory).
# One sample per second; supports sessions well beyond 30 minutes (3 hours).
MAX_RISK_HISTORY = 10800  # ~3 hours at 1 sample/sec

# Lightweight sampling interval for the risk trend history.
# History points are appended at most once per second so the bounded buffer
# represents real session time rather than camera-frame count.
RISK_SAMPLE_INTERVAL_S = 1.0


@dataclass(frozen=True)
class RiskSample:
    """One point in the risk trend history."""

    timestamp_s: float
    risk: float | None
    severity: str
    eye_risk: float | None
    blink_risk: float | None
    yawn_risk: float | None
    night_risk: float | None
    duration_risk: float | None


@dataclass(frozen=True)
class AlertnessSample:
    """Authoritative sampled alertness state for all downstream surfaces."""

    timestamp_s: float
    alertness: float | None
    attention_risk: float | None
    fatigue: float | None
    distraction: float | None
    environmental_risk: float | None
    severity: str
    primary_reason: str


@dataclass
class AlertEvent:
    """One continuous alert episode, including any later escalation."""

    start_time_s: float
    end_time_s: float | None
    severity: str
    primary_reason: str
    contributing_factors: tuple[str, ...] = ()
    alertness_at_start: float = 100.0
    minimum_alertness: float = 100.0
    recovery_time_s: float | None = None
    action: str = ""


@dataclass
class SessionStats:
    """Aggregated statistics for the current (or last) session."""

    session_id: str = ""
    start_time: datetime | None = None
    end_time: datetime | None = None
    driving_duration_s: float = 0.0
    average_fatigue_risk: float = 0.0
    maximum_fatigue_risk: float = 0.0
    blink_count: int = 0
    average_blink_rate: float = 0.0
    yawn_count: int = 0
    low_fatigue_count: int = 0
    alert_count: int = 0
    warning_count: int = 0
    drowsy_count: int = 0
    night_context_active: bool = False
    final_status: str = "NORMAL"
    risk_history: list[RiskSample] = field(default_factory=list)
    risk_factor_history: dict[str, list[float | None]] = field(default_factory=dict)
    alertness_history: list[AlertnessSample] = field(default_factory=list)
    alert_events: list[AlertEvent] = field(default_factory=list)
    average_alertness: float = 100.0
    minimum_alertness: float = 100.0
    maximum_alertness: float = 100.0
    time_safe_s: float = 0.0
    time_moderate_s: float = 0.0
    time_high_s: float = 0.0
    time_critical_s: float = 0.0
    distraction_event_count: int = 0
    fatigue_event_count: int = 0
    recovery_event_count: int = 0
    high_risk_zone_event_count: int = 0
    average_distraction: float = 0.0
    average_environmental_risk: float = 0.0


class SessionManager:
    """
    Manages a driving session lifecycle.

    - start(): creates a new session ID, resets stats, starts the timer.
    - update(): records a FatigueDecision sample (risk history, counters).
    - stop(): finalizes statistics and returns a SessionStats snapshot.
    """

    def __init__(self, use_demo_duration: bool = True) -> None:
        self.use_demo_duration = use_demo_duration
        self._active = False
        self._start_mono_s: float = 0.0
        self._duration_tracker: DrivingDurationTracker | None = None
        self._stats = SessionStats()
        self._risk_sum = 0.0
        self._risk_samples = 0
        self._blink_rate_sum = 0.0
        self._blink_rate_samples = 0
        self._last_risk_sample_s: float | None = None
        self._alertness_sum = 0.0
        self._alertness_samples = 0
        self._distraction_sum = 0.0
        self._environment_sum = 0.0
        self._last_alertness_severity = "SAFE"
        self._last_zone_event = False
        self._active_alert_event: AlertEvent | None = None

    @property
    def start_mono_s(self) -> float:
        """Monotonic start time (seconds) of the current/last session."""
        return self._start_mono_s

    @property
    def active(self) -> bool:
        return self._active

    @property
    def stats(self) -> SessionStats:
        return self._stats

    def start(self) -> str:
        """Start a new driving session. Returns the new session ID."""
        self._active = True
        # Elapsed-driven timer uses monotonic time (immune to wall-clock changes).
        self._start_mono_s = time.monotonic()
        self._duration_tracker = DrivingDurationTracker(
            start_s=self._start_mono_s,
            use_demo=self.use_demo_duration,
        )
        self._stats = SessionStats(
            session_id=uuid.uuid4().hex[:12].upper(),
            start_time=datetime.now(),
        )
        self._risk_sum = 0.0
        self._risk_samples = 0
        self._blink_rate_sum = 0.0
        self._blink_rate_samples = 0
        self._last_severity = None
        self._last_risk_sample_s = None
        self._alertness_sum = 0.0
        self._alertness_samples = 0
        self._distraction_sum = 0.0
        self._environment_sum = 0.0
        self._last_alertness_severity = "SAFE"
        self._last_zone_event = False
        self._active_alert_event = None
        return self._stats.session_id

    def stop(self) -> SessionStats:
        """Finalize the session and return a snapshot of the stats."""
        if not self._active:
            return self._stats
        self._active = False
        # Human-readable wall-clock for the end timestamp.
        self._stats.end_time = datetime.now()
        if self._duration_tracker is not None:
            result = self._duration_tracker.update(time.monotonic())
            self._stats.driving_duration_s = result.elapsed_s
        if self._risk_samples > 0:
            self._stats.average_fatigue_risk = self._risk_sum / self._risk_samples
        if self._blink_rate_samples > 0:
            self._stats.average_blink_rate = self._blink_rate_sum / self._blink_rate_samples
        if self._alertness_samples > 0:
            self._stats.average_alertness = self._alertness_sum / self._alertness_samples
            self._stats.average_distraction = self._distraction_sum / self._alertness_samples
            self._stats.average_environmental_risk = self._environment_sum / self._alertness_samples
        if self._stats.alertness_history:
            values = [s.alertness for s in self._stats.alertness_history if s.alertness is not None]
            if values:
                self._stats.minimum_alertness = min(values)
                self._stats.maximum_alertness = max(values)
            self._finalize_state_durations()
        if self._active_alert_event is not None:
            self._active_alert_event.end_time_s = self._stats.alertness_history[-1].timestamp_s if self._stats.alertness_history else None
            self._active_alert_event = None
        return self._stats

    def _finalize_state_durations(self) -> None:
        history = self._stats.alertness_history
        for current, following in zip(history, history[1:]):
            seconds = max(0.0, following.timestamp_s - current.timestamp_s)
            if current.severity in ("SAFE", "NORMAL"):
                self._stats.time_safe_s += seconds
            elif current.severity in ("MODERATE", "LOW_FATIGUE", "ALERT"):
                self._stats.time_moderate_s += seconds
            elif current.severity in ("HIGH", "WARNING"):
                self._stats.time_high_s += seconds
            elif current.severity in ("CRITICAL", "DROWSY"):
                self._stats.time_critical_s += seconds

    def driving_result(self, now_s: float | None = None):
        """Return the current DrivingDurationResult from the backend tracker."""
        if not self._active or self._duration_tracker is None:
            from driving_duration import DrivingDurationResult

            return DrivingDurationResult(
                elapsed_s=0.0,
                driving_time_text="00:00:00",
                threshold_s=0.0,
                duration_risk="NORMAL",
                threshold_reached=False,
                demo_mode=True,
            )
        return self._duration_tracker.update(now_s if now_s is not None else time.monotonic())

    def driving_time_text(self, now_s: float | None = None) -> str:
        """Return the authoritative backend session duration as HH:MM:SS."""
        if not self._active or self._duration_tracker is None:
            return "00:00:00"
        result = self._duration_tracker.update(now_s if now_s is not None else time.monotonic())
        return result.driving_time_text

    def _should_sample_history(self, now: float) -> bool:
        """Return True when a new history point should be recorded."""
        return (
            self._last_risk_sample_s is None
            or (now - self._last_risk_sample_s) >= RISK_SAMPLE_INTERVAL_S
        )

    def _should_sample_alertness(self, now: float) -> bool:
        """Sample alertness independently while keeping the same cadence."""
        return (
            not self._stats.alertness_history
            or (now - self._stats.alertness_history[-1].timestamp_s) >= RISK_SAMPLE_INTERVAL_S
        )

    def record_no_face(self, now_s: float | None = None) -> None:
        """
        Record a 'no face detected' sample without creating fake events.

        The risk and factor values are marked as unavailable (None) rather
        than zero, so the trend continues showing historical data without
        fabricating eye/blink/yawn events.

        History is sampled at the same RISK_SAMPLE_INTERVAL_S so a brief
        face loss does not flood the buffer; previous valid points are
        preserved untouched.
        """
        if not self._active:
            return
        now = now_s if now_s is not None else time.monotonic()
        if not self._should_sample_history(now):
            return
        self._last_risk_sample_s = now
        sample = RiskSample(
            timestamp_s=now,
            risk=None,
            severity="NO FACE DETECTED",
            eye_risk=None,
            blink_risk=None,
            yawn_risk=None,
            night_risk=None,
            duration_risk=None,
        )
        self._stats.risk_history.append(sample)
        if len(self._stats.risk_history) > MAX_RISK_HISTORY:
            self._stats.risk_history = self._stats.risk_history[-MAX_RISK_HISTORY:]
        for key in ("eye_risk", "blink_risk", "yawn_risk", "night_risk", "duration_risk"):
            series = self._stats.risk_factor_history.setdefault(key, [])
            series.append(None)
            if len(series) > MAX_RISK_HISTORY:
                self._stats.risk_factor_history[key] = series[-MAX_RISK_HISTORY:]

    def update(self, decision, now_s: float | None = None, attention=None, alert_result=None) -> None:
        """
        Record one FatigueDecision sample into the session.

        `decision` is a FatigueEngine.FatigueDecision (or any object with
        .severity, .smoothed_risk, .breakdown, .evidence).
        """
        if not self._active:
            return

        now = now_s if now_s is not None else time.monotonic()
        risk = decision.smoothed_risk
        severity = decision.severity

        # Risk history (bounded, sampled ~1/sec so it reflects session time).
        if self._should_sample_history(now):
            self._last_risk_sample_s = now
            sample = RiskSample(
                timestamp_s=now,
                risk=risk,
                severity=severity,
                eye_risk=decision.breakdown.get("eye_risk", 0.0),
                blink_risk=decision.breakdown.get("blink_risk", 0.0),
                yawn_risk=decision.breakdown.get("yawn_risk", 0.0),
                night_risk=decision.breakdown.get("night_risk", 0.0),
                duration_risk=decision.breakdown.get("duration_risk", 0.0),
            )
            self._stats.risk_history.append(sample)
            if len(self._stats.risk_history) > MAX_RISK_HISTORY:
                self._stats.risk_history = self._stats.risk_history[-MAX_RISK_HISTORY:]

            # Risk-factor history (bounded).
            for key in ("eye_risk", "blink_risk", "yawn_risk", "night_risk", "duration_risk"):
                series = self._stats.risk_factor_history.setdefault(key, [])
                series.append(decision.breakdown.get(key, 0.0))
                if len(series) > MAX_RISK_HISTORY:
                    self._stats.risk_factor_history[key] = series[-MAX_RISK_HISTORY:]

        # Risk aggregation.
        if risk is not None:
            self._risk_sum += float(risk)
            self._risk_samples += 1
            self._stats.maximum_fatigue_risk = max(
                self._stats.maximum_fatigue_risk, float(risk)
            )

        if attention is not None:
            alertness = max(0.0, min(100.0, float(attention.alertness_score)))
            distraction = max(0.0, min(100.0, float(attention.distraction_score)))
            environmental = max(
                0.0,
                min(100.0, max(float(getattr(attention, "location_risk", 0.0)), float(getattr(attention, "weather_risk", 0.0)))),
            )
            if self._should_sample_alertness(now):
                sample = AlertnessSample(
                    timestamp_s=now,
                    alertness=alertness,
                    attention_risk=100.0 - alertness,
                    fatigue=float(attention.fatigue_score),
                    distraction=distraction,
                    environmental_risk=environmental,
                    severity=str(attention.level),
                    primary_reason=str(attention.primary_reason or ""),
                )
                self._stats.alertness_history.append(sample)
                if len(self._stats.alertness_history) > MAX_RISK_HISTORY:
                    self._stats.alertness_history = self._stats.alertness_history[-MAX_RISK_HISTORY:]
            self._alertness_sum += alertness
            self._alertness_samples += 1
            self._distraction_sum += distraction
            self._environment_sum += environmental
            if distraction >= 20.0 and self._last_alertness_severity not in ("MODERATE", "HIGH", "CRITICAL"):
                self._stats.distraction_event_count += 1
            if float(attention.fatigue_score) >= 25.0 and self._last_alertness_severity not in ("MODERATE", "HIGH", "CRITICAL"):
                self._stats.fatigue_event_count += 1
            if self._last_alertness_severity in ("MODERATE", "HIGH", "CRITICAL") and attention.level == "SAFE":
                self._stats.recovery_event_count += 1
            zone_event = (
                str(getattr(attention, "location_zone", "none")) not in ("none", "", "unknown")
                and float(getattr(attention, "location_risk", 0.0)) >= 40.0
            )
            if zone_event and not self._last_zone_event:
                self._stats.high_risk_zone_event_count += 1
            self._last_zone_event = zone_event
            self._last_alertness_severity = str(attention.level)

            if alert_result is not None and alert_result.active_severity != "NORMAL":
                if self._active_alert_event is None:
                    self._active_alert_event = AlertEvent(
                        start_time_s=now,
                        end_time_s=None,
                        severity=alert_result.active_severity,
                        primary_reason=str(alert_result.reason or attention.primary_reason),
                        contributing_factors=tuple(alert_result.contributing_factors),
                        alertness_at_start=alertness,
                        minimum_alertness=alertness,
                        action=str(getattr(attention, "recommendation", "") or ""),
                    )
                    self._stats.alert_events.append(self._active_alert_event)
                else:
                    event = self._active_alert_event
                    if priority := {"NORMAL": 0, "LOW_FATIGUE": 1, "ALERT": 2, "WARNING": 3, "DROWSY": 4}.get(alert_result.active_severity, 0):
                        if priority > {"NORMAL": 0, "LOW_FATIGUE": 1, "ALERT": 2, "WARNING": 3, "DROWSY": 4}.get(event.severity, 0):
                            event.severity = alert_result.active_severity
                    event.minimum_alertness = min(event.minimum_alertness, alertness)
            elif self._active_alert_event is not None:
                self._active_alert_event.end_time_s = now
                self._active_alert_event.recovery_time_s = max(0.0, now - self._active_alert_event.start_time_s)
                self._active_alert_event = None

        # Blink rate aggregation (from evidence if available).
        blink_rate = getattr(decision.evidence, "blink_rate", None)
        if blink_rate is not None:
            self._blink_rate_sum += float(blink_rate)
            self._blink_rate_samples += 1

        # Counters.
        self._stats.blink_count = getattr(decision.evidence, "blink_count", self._stats.blink_count)
        self._stats.yawn_count = getattr(decision.evidence, "yawn_count", self._stats.yawn_count)
        self._stats.night_context_active = getattr(
            decision.evidence, "night_context", self._stats.night_context_active
        )

        if self._last_severity != severity:
            if severity == "LOW_FATIGUE":
                self._stats.low_fatigue_count += 1
            elif severity == "ALERT":
                self._stats.alert_count += 1
            elif severity == "WARNING":
                self._stats.warning_count += 1
            elif severity == "DROWSY":
                self._stats.drowsy_count += 1
            self._last_severity = severity

        # Final status = highest severity seen.
        priority = {"NORMAL": 0, "LOW_FATIGUE": 1, "ALERT": 2, "WARNING": 3, "DROWSY": 4}
        if priority.get(severity, 0) > priority.get(self._stats.final_status, 0):
            self._stats.final_status = severity