"""
Driver attention risk layer.

Combines existing fatigue results with distraction monitoring into a single
transparent attention risk (0–100) and level:
  SAFE / MODERATE / HIGH / CRITICAL

Does not rewrite fatigue / blink / yawn algorithms — consumes their scores.
Asymmetric risk decrease: brief improvement must not immediately lower risk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from distraction_monitor import DistractionObservation

# Transparent fusion weights (fatigue remains the dominant contributor).
WEIGHT_FATIGUE = 0.90
WEIGHT_DISTRACTION = 0.38

# Soft context gains (location / weather amplify; they do not replace driver risk).
LOCATION_CONTEXT_GAIN = 0.22
WEATHER_CONTEXT_GAIN = 0.20
CONTEXT_MAX_SOFT = 18.0

# Context urgency (deterministic, ordering-only modifier).
#
# Context NEVER creates an alert by itself. It only shortens the cooldown /
# speeds the escalation of an EXISTING behavioral alert when the external
# conditions are genuinely high-risk. 1.0 = normal context (no effect); lower
# values mean more urgent. Bounded so it can never drive the system beyond
# safe limits.
CONTEXT_URGENCY_NORMAL = 1.0
CONTEXT_URGENCY_MIN = 0.5
CONTEXT_URGENCY_MAX_REDUCTION = CONTEXT_URGENCY_NORMAL - CONTEXT_URGENCY_MIN

# A context axis only counts as "high-risk" at/above these risk scores.
CONTEXT_LOCATION_HIGH_RISK = 40.0
CONTEXT_WEATHER_HIGH_RISK = 40.0

# Maximum urgency reduction contributed by each axis (risk=100 yields the gain).
LOCATION_URGENCY_GAIN = 0.25
WEATHER_URGENCY_GAIN = 0.20
SPEED_CONTEXT_GAIN = 0.12
TIME_CONTEXT_GAIN = 0.10
CONTEXT_SPEED_HIGH_RISK = 40.0
CONTEXT_TIME_HIGH_RISK = 1.0

# Fixed urgency contribution for a verified high-risk zone type. Only applied
# for genuinely risky zones; "none" (normal road) contributes nothing.
ZONE_URGENCY = {
    "school": 0.15,
    "construction": 0.12,
    "sharp_turn": 0.15,
    "highway": 0.05,
    "other": 0.08,
}

# Level bands.
SAFE_MAX = 24
MODERATE_MAX = 49
HIGH_MAX = 74
# >= HIGH_MAX+ → CRITICAL

# A reduced-attention band is only eligible for a lightweight caution when
# environmental context is significant. It never turns ordinary 75-89
# alertness into a continuous alert.
CAUTION_ALERTNESS_MIN = 75.0
CAUTION_ALERTNESS_MAX = 89.0
CAUTION_WEATHER_RISK = 60.0
CAUTION_NIGHT_RISK = 20.0
CAUTION_ZONE_RISK = 55.0

# ALERT sits inside the MODERATE band. Enter ALERT at ALERT_ENTRY, but do not
# immediately drop back to LOW_FATIGUE just below it: use a lower exit
# threshold (hysteresis) so LOW_FATIGUE <-> ALERT flapping is avoided.
ALERT_ENTRY = 35.0
ALERT_EXIT = 32.0

# WARNING (HIGH) requires a short persistence before it is emitted so a
# single-frame / very brief risk spike cannot fire a serious alarm.
#
# The persistence is ACCUMULATED per pending severity (see
# `_derive_alert_severity`): the accumulator only grows while the band that
# maps to that severity is actually present, and it is never wiped by an
# intermediate band crossing. Before this, a risk value that flickered across
# the HIGH/CRITICAL boundary (e.g. 73 <-> 84 around the 74 band edge) reset the
# pending escalation on every crossing, so a 77+ attention risk could stay at
# WARNING (or below) forever and never reach the alert/audio path.
ALERT_PERSISTENCE_S = 0.75
WARNING_PERSISTENCE_S = 1.00
DROWSY_PERSISTENCE_S = 1.50
RECOVERY_PERSISTENCE_S = 1.25

# A context-scaled persistence window may never shrink below this (a single
# frame must never fire a serious alarm, even in the riskiest context).
PERSISTENCE_MIN_S = 0.30
# Pending escalations keep a short retention window after the band drops away so
# a brief flicker cannot discard a genuine rising alert. This is deliberately
# longer than the per-band checks to avoid flapping between higher severities.
PENDING_ESCALATION_TTL_S = max(
    PERSISTENCE_MIN_S,
    ALERT_PERSISTENCE_S,
    WARNING_PERSISTENCE_S,
    DROWSY_PERSISTENCE_S,
    RECOVERY_PERSISTENCE_S,
) * 2.0

# Asymmetric decrease (same philosophy as FatigueEngine).
RISK_DECREASE_HOLD_S = 0.45
RISK_DECREASE_TAU_S = 1.10
RISK_INCREASE_ALPHA = 0.45

# Map attention level → existing audio / alert escalation severities.
# MODERATE spans former LOW_FATIGUE + ALERT bands via score thresholds.
LEVEL_TO_ALERT = {
    "SAFE": "NORMAL",
    "MODERATE": "LOW_FATIGUE",
    "HIGH": "WARNING",
    "CRITICAL": "DROWSY",
}

# ---------------------------------------------------------------------------
# Context-sensitive escalation (ordering-only, deterministic)
# ---------------------------------------------------------------------------
# A driver who is ALREADY demonstrably degrading (both fatigue and distraction
# evidence present) needs less proof before the existing escalation fires.
# This is a modifier on the existing timing only - it never creates risk.
DRIVER_COMBO_RISK = 40.0  # each axis must reach this to count as "combined"
DRIVER_COMBO_URGENCY_GAIN = 0.12

# ---------------------------------------------------------------------------
# Temporal classification of a sustained DOWNWARD head pose.
# ---------------------------------------------------------------------------
# A downward head pose alone is NOT drowsiness (the eyes are simply not
# visible). What it means depends on the driver's state BEFORE the downward
# pose began, so the classification is latched from temporal history:
#
#   prior drowsiness + sustained downward pose -> drowsiness-grade (CASE A)
#   no prior drowsiness + sustained downward   -> mobile-use-like distraction
#                                                 (CASE B, behaviour-based; the
#                                                 camera does NOT prove a phone)
#
# Drowsiness evidence in the trailing window before the downward event starts.
PRIOR_DROWSINESS_WINDOW_S = 20.0
PRIOR_DROWSINESS_RISK = 45.0
PRIOR_DROWSINESS_STATUSES = ("WARNING", "DROWSY")

# Bounded drowsiness-grade amplification for CASE A (never unbounded, never
# applied without genuine pre-existing drowsiness evidence).
DROWSY_DOWNWARD_MIN_RISK = 75.0  # CRITICAL band floor for the combined event
DROWSY_DOWNWARD_GAIN = 10.0

HEAD_DOWN_NONE = "none"
HEAD_DOWN_DROWSY = "drowsy_looking_down"
HEAD_DOWN_MOBILE_USE = "mobile_use_like"


@dataclass(frozen=True)
class AttentionDecision:
    """Transparent attention-layer output."""

    fatigue_risk: float
    distraction_risk: float
    combined_risk: float  # smoothed 0..100
    raw_combined: float
    level: str  # SAFE / MODERATE / HIGH / CRITICAL
    alert_severity: str  # NORMAL / ALERT / WARNING / DROWSY (existing path)
    primary_reason: str
    distraction_primary: str
    face_ok: bool
    distraction_type: str = "none"
    alertness_score: float = 100.0
    fatigue_score: float = 0.0
    distraction_score: float = 0.0
    location_risk: float = 0.0
    weather_risk: float = 0.0
    location_zone: str = "none"
    weather_condition: str = "unknown"
    safe_stop: bool = False
    recommendation: str = ""
    head_direction: str = "unknown"
    context_urgency: float = 1.0
    carla_speed_kmh: float = 0.0
    carla_speed_risk: float = 0.0
    carla_vehicle_state: str = "UNAVAILABLE"
    carla_time_period: str = "UNKNOWN"
    # Temporal head-down classification (PART 8):
    # none | drowsy_looking_down | mobile_use_like
    head_down_classification: str = "none"
    # The severity/level the UI shows always comes from this decision, and the
    # alert/audio path is driven by `alert_severity` from the same computation.
    pending_escalation: str = "NONE"  # escalation being confirmed (for triage)


def risk_level_from_score(score: float | None) -> str:
    if score is None:
        return "SAFE"
    s = max(0.0, min(100.0, float(score)))
    if s <= SAFE_MAX:
        return "SAFE"
    if s <= MODERATE_MAX:
        return "MODERATE"
    if s <= HIGH_MAX:
        return "HIGH"
    return "CRITICAL"


def _distraction_type_factor(primary: str | None, duration_s: float = 0.0) -> float:
    """Different distraction modes carry different baseline importance."""
    name = (primary or "").lower()
    base = {
        "looking_down": 1.35,
        "head_turn_left": 0.9,
        "head_turn_right": 0.9,
        "passenger_side": 0.8,
        "none": 0.0,
    }.get(name, 0.7)
    if duration_s > 0.0 and duration_s < 2.0:
        base *= 0.6
    return float(base)


def fuse_raw_risk(fatigue_risk: float, distraction_risk: float, distraction_primary: str | None = None, distraction_duration_s: float = 0.0) -> float:
    """Weighted fusion that still respects different distraction types and duration."""
    f = max(0.0, min(100.0, float(fatigue_risk)))
    d = max(0.0, min(100.0, float(distraction_risk)))
    type_factor = _distraction_type_factor(distraction_primary, distraction_duration_s)
    d_adjusted = d * type_factor
    combined = WEIGHT_FATIGUE * f + WEIGHT_DISTRACTION * d_adjusted
    if f >= 40.0 and d_adjusted >= 40.0:
        combined = min(100.0, combined + 8.0)
    return float(min(100.0, combined))


def apply_context_boost(
    driver_risk: float,
    distraction_risk: float,
    location_risk: float = 0.0,
    weather_risk: float = 0.0,
    zone_type: str = "none",
    speed_risk: float = 0.0,
    time_risk: float = 0.0,
) -> float:
    """
    Amplify driver risk with location / weather context.

    Examples (moderate distraction ~45):
      + normal road → stays moderate
      + school zone → higher
      + rain → higher
      + school zone + heavy rain → potentially critical
    """
    base = max(0.0, min(100.0, float(driver_risk)))
    d = max(0.0, min(100.0, float(distraction_risk)))
    loc = max(0.0, min(100.0, float(location_risk)))
    wx = max(0.0, min(100.0, float(weather_risk)))
    speed = max(0.0, min(100.0, float(speed_risk)))
    time = max(0.0, min(100.0, float(time_risk)))
    boost = 0.0

    # Soft proportional context when there is already meaningful driver risk.
    if base >= 20.0 or d >= 30.0:
        soft = min(
            CONTEXT_MAX_SOFT,
            LOCATION_CONTEXT_GAIN * loc + WEATHER_CONTEXT_GAIN * wx,
        )
        boost += soft * (0.35 + base / 200.0)
        boost += (SPEED_CONTEXT_GAIN * speed) + (TIME_CONTEXT_GAIN * time)

    # Moderate distraction + elevated road zone (school / construction / sharp turn).
    if d >= 35.0 and loc >= 55.0:
        boost += 12.0
    if d >= 35.0 and zone_type == "school" and float(time_risk or 0.0) <= 0.0:
        boost += 6.0

    # Moderate distraction + rain / severe weather.
    if d >= 35.0 and wx >= 50.0:
        boost += 10.0

    # Moderate distraction + school/high-risk zone + heavy rain → critical territory.
    if d >= 35.0 and loc >= 55.0 and wx >= 70.0:
        boost += 14.0

    return float(min(100.0, base + boost))


def context_urgency_from(
    location_risk: float,
    weather_risk: float,
    zone_type: str = "none",
    speed_risk: float = 0.0,
    time_risk: float = 0.0,
    fatigue_risk: float = 0.0,
    distraction_risk: float = 0.0,
) -> float:
    """Deterministic context urgency modifier (1.0 = normal, lower = more urgent).

    Derived from location risk, weather risk, and zone type. Returns exactly
    1.0 when no high-risk context is present, and reverts to 1.0 as soon as the
    high-risk context disappears. The result is clamped to
    ``[CONTEXT_URGENCY_MIN, CONTEXT_URGENCY_NORMAL]`` so it can never exceed
    safe bounds.

    ``fatigue_risk`` / ``distraction_risk`` add the *driver* half of the same
    ordering rule: when the driver is already demonstrably degrading on BOTH
    axes (combined fatigue + distraction), an existing behavioural alert may
    escalate sooner. Both default to 0.0, so the context-only behaviour (and
    every existing caller/test) is unchanged.

    This value is an *ordering* modifier for alert escalation (shorter
    cooldown / faster escalation). It does not raise the raw risk score and can
    never create a fatigue or distraction claim on its own.
    """
    loc = max(0.0, min(100.0, float(location_risk or 0.0)))
    wx = max(0.0, min(100.0, float(weather_risk or 0.0)))
    speed = max(0.0, min(100.0, float(speed_risk or 0.0)))
    time = max(0.0, min(100.0, float(time_risk or 0.0)))
    fatigue = max(0.0, min(100.0, float(fatigue_risk or 0.0)))
    distraction = max(0.0, min(100.0, float(distraction_risk or 0.0)))
    zone = (zone_type or "none").strip().lower()

    reduction = 0.0
    if loc >= CONTEXT_LOCATION_HIGH_RISK:
        reduction += LOCATION_URGENCY_GAIN * (
            (loc - CONTEXT_LOCATION_HIGH_RISK) / (100.0 - CONTEXT_LOCATION_HIGH_RISK)
        )
    if wx >= CONTEXT_WEATHER_HIGH_RISK:
        reduction += WEATHER_URGENCY_GAIN * (
            (wx - CONTEXT_WEATHER_HIGH_RISK) / (100.0 - CONTEXT_WEATHER_HIGH_RISK)
        )
    if speed >= CONTEXT_SPEED_HIGH_RISK:
        reduction += 0.10 * ((speed - CONTEXT_SPEED_HIGH_RISK) / (100.0 - CONTEXT_SPEED_HIGH_RISK))
    if time >= CONTEXT_TIME_HIGH_RISK:
        reduction += 0.08
    if zone in ZONE_URGENCY:
        if zone == "school" and time >= CONTEXT_TIME_HIGH_RISK:
            reduction += 0.2 * ZONE_URGENCY[zone]
        else:
            reduction += ZONE_URGENCY[zone]
    # Combined fatigue + distraction: the weakest axis still has to be real
    # evidence (>= DRIVER_COMBO_RISK), and the gain scales with the weaker axis
    # so a mild combination never becomes urgent.
    if fatigue >= DRIVER_COMBO_RISK and distraction >= DRIVER_COMBO_RISK:
        weakest = min(fatigue, distraction)
        reduction += DRIVER_COMBO_URGENCY_GAIN * (
            (weakest - DRIVER_COMBO_RISK) / (100.0 - DRIVER_COMBO_RISK)
        )

    reduction = min(CONTEXT_URGENCY_MAX_REDUCTION, reduction)
    return float(max(CONTEXT_URGENCY_MIN, CONTEXT_URGENCY_NORMAL - reduction))


class DownwardGazeTemporalClassifier:
    """Classify a sustained downward head pose using PRE-EVENT history.

    The current frame alone cannot decide what a downward head pose means, so
    the classifier keeps temporal state:

    * It continuously records when genuine drowsiness evidence was last present
      (fatigue risk at/above ``PRIOR_DROWSINESS_RISK`` or a fatigue status of
      WARNING/DROWSY - prolonged eye closure reaches those bands on its own).
    * When a sustained downward-looking distraction event STARTS, the pre-event
      drowsiness state is latched once and kept for the whole event.
    * ``drowsy_looking_down``  - drowsiness existed BEFORE the downward pose
      (CASE A: the combined event is drowsiness-grade, not plain distraction).
    * ``mobile_use_like``      - no drowsiness was established before the
      downward pose began (CASE B: behaviour-based mobile-use-like distraction).
      This is a behavioural classification only: no phone/hand model exists, so
      the system never claims a phone was detected.

    Deliberately NOT implemented: ``if looking_down: drowsy``.
    """

    def __init__(
        self,
        prior_window_s: float = PRIOR_DROWSINESS_WINDOW_S,
        prior_risk: float = PRIOR_DROWSINESS_RISK,
        prior_statuses: tuple[str, ...] = PRIOR_DROWSINESS_STATUSES,
    ) -> None:
        self.prior_window_s = max(0.0, float(prior_window_s))
        self.prior_risk = float(prior_risk)
        self.prior_statuses = tuple(prior_statuses)
        self._last_drowsy_s: float | None = None
        self._drowsy_since_s: float | None = None
        self._down_event_active = False
        self._down_event_started_s: float | None = None
        self._down_event_prior_drowsy = False

    def reset(self) -> None:
        self._last_drowsy_s = None
        self._drowsy_since_s = None
        self._down_event_active = False
        self._down_event_started_s = None
        self._down_event_prior_drowsy = False

    @property
    def prior_drowsiness_present(self) -> bool:
        """True while the trailing drowsiness window still holds evidence."""
        return self._last_drowsy_s is not None

    @property
    def down_event_prior_drowsy(self) -> bool:
        """The drowsiness state captured before the current downward event."""
        return bool(self._down_event_prior_drowsy)

    def _looking_down_active(self, distraction) -> bool:
        if distraction is None:
            return False
        channel = getattr(distraction, "looking_down", None)
        if channel is None:
            return False
        if getattr(channel, "active", False):
            return True
        # A primary "looking_down" classification is equally valid evidence.
        return str(getattr(distraction, "primary", "") or "") == "looking_down"

    def update(
        self,
        fatigue_risk: float,
        distraction,
        now_s: float,
        fatigue_status: str = "SAFE",
    ) -> str:
        """Return none | drowsy_looking_down | mobile_use_like for this frame."""
        risk = max(0.0, float(fatigue_risk or 0.0))
        status = str(fatigue_status or "").strip().upper()
        drowsy = risk >= self.prior_risk or status in self.prior_statuses
        if drowsy:
            self._last_drowsy_s = float(now_s)
            if self._drowsy_since_s is None:
                self._drowsy_since_s = float(now_s)
        else:
            self._drowsy_since_s = None

        looking_down = self._looking_down_active(distraction)
        if looking_down:
            if not self._down_event_active:
                # Event START: latch the drowsiness state that existed BEFORE
                # this downward pose (trailing window, never the current frame).
                self._down_event_active = True
                self._down_event_started_s = float(now_s)
                self._down_event_prior_drowsy = bool(
                    self._last_drowsy_s is not None
                    and (float(now_s) - self._last_drowsy_s) <= self.prior_window_s
                )
            return (
                HEAD_DOWN_DROWSY
                if self._down_event_prior_drowsy
                else HEAD_DOWN_MOBILE_USE
            )

        # Event ended: the next downward pose is classified from fresh history.
        self._down_event_active = False
        self._down_event_started_s = None
        self._down_event_prior_drowsy = False
        return HEAD_DOWN_NONE


class AttentionEngine:
    """Fuses fatigue + distraction with held/gradual risk decrease."""

    def __init__(
        self,
        weight_fatigue: float = WEIGHT_FATIGUE,
        weight_distraction: float = WEIGHT_DISTRACTION,
        risk_decrease_hold_s: float = RISK_DECREASE_HOLD_S,
        risk_decrease_tau_s: float = RISK_DECREASE_TAU_S,
        risk_increase_alpha: float = RISK_INCREASE_ALPHA,
    ) -> None:
        self.weight_fatigue = float(weight_fatigue)
        self.weight_distraction = float(weight_distraction)
        self.risk_decrease_hold_s = max(0.0, float(risk_decrease_hold_s))
        self.risk_decrease_tau_s = max(0.05, float(risk_decrease_tau_s))
        self.risk_increase_alpha = min(0.99, max(0.05, float(risk_increase_alpha)))
        # Temporal classification of a sustained downward head pose (PART 8).
        self.head_down_classifier = DownwardGazeTemporalClassifier()
        self.reset()

    def reset(self) -> None:
        self._smoothed: float | None = None
        self._improve_since: float | None = None
        self._last_raw: float | None = None
        self._last_update_s: float | None = None
        self._alert_severity = "NORMAL"
        self._warning_pending_since: float | None = None
        # Pending escalation (accumulated, never wiped by an intermediate band):
        # the severity waiting for its persistence window, the time already
        # spent at/above that severity, and the previous evaluation moment used
        # to accumulate the elapsed time.
        self._pending_alert_target: str | None = None
        self._pending_alert_priority: int = 0
        self._pending_alert_accum_s: float = 0.0
        self._last_escalation_eval_s: float | None = None
        self._recovery_since: float | None = None
        if getattr(self, "head_down_classifier", None) is not None:
            self.head_down_classifier.reset()

    def _derive_alert_severity(
        self,
        smoothed: float | None,
        level: str,
        now_s: float,
        context_urgency: float = CONTEXT_URGENCY_NORMAL,
    ) -> str:
        """Map smoothed attention risk to an alert severity with hysteresis.

        - LOW_FATIGUE <-> ALERT uses entry/exit hysteresis inside MODERATE.
        - WARNING (HIGH) / DROWSY (CRITICAL) require a short *accumulated*
          persistence before being emitted, so a single-frame spike cannot fire
          a serious alarm, while a persistently high risk value always does.
        - The persistence window is scaled by the existing deterministic context
          urgency: in a genuinely high-risk context (night / adverse weather /
          high-risk zone / high speed / combined fatigue+distraction) the same
          high risk value escalates sooner. Normal context keeps the configured
          window unchanged.
        """
        if level == "SAFE":
            target = "NORMAL"
        elif level == "MODERATE":
            if smoothed is None:
                target = "LOW_FATIGUE"
            elif self._alert_severity == "ALERT" and smoothed >= ALERT_EXIT:
                target = "ALERT"
            elif smoothed >= ALERT_ENTRY:
                target = "ALERT"
            else:
                target = "LOW_FATIGUE"
        elif level == "HIGH":
            target = "WARNING"
        else:  # CRITICAL
            target = "DROWSY"

        priority = {"NORMAL": 0, "LOW_FATIGUE": 1, "ALERT": 2, "WARNING": 3, "DROWSY": 4}
        urgency = max(CONTEXT_URGENCY_MIN, min(CONTEXT_URGENCY_NORMAL, float(context_urgency)))
        persistence = {
            "ALERT": max(PERSISTENCE_MIN_S, ALERT_PERSISTENCE_S * urgency),
            "WARNING": max(PERSISTENCE_MIN_S, WARNING_PERSISTENCE_S * urgency),
            "DROWSY": max(PERSISTENCE_MIN_S, DROWSY_PERSISTENCE_S * urgency),
        }

        # Elapsed time since the previous evaluation (this is the amount the
        # current frame can contribute to / take away from the persistence).
        previous_eval_s = self._last_escalation_eval_s
        self._last_escalation_eval_s = float(now_s)
        dt = 0.0 if previous_eval_s is None else max(0.0, float(now_s) - previous_eval_s)

        current_priority = priority.get(self._alert_severity, 0)
        target_priority = priority.get(target, 0)

        # A pending escalation is dropped only when it can no longer escalate
        # (its severity is no longer above the confirmed severity) or when the
        # band that produced it has been absent for longer than the retention
        # window. A short crossing of a lower band (the HIGH/CRITICAL flicker in
        # the original bug) must NOT discard it.
        if self._pending_alert_target is not None and self._pending_alert_priority <= current_priority:
            self._pending_alert_target = None
            self._pending_alert_priority = 0
            self._pending_alert_accum_s = 0.0
            self._pending_alert_last_seen_s = None
        if (
            self._pending_alert_target is not None
            and self._pending_alert_last_seen_s is not None
            and (float(now_s) - self._pending_alert_last_seen_s) > PENDING_ESCALATION_TTL_S
        ):
            self._pending_alert_target = None
            self._pending_alert_priority = 0
            self._pending_alert_accum_s = 0.0
            self._pending_alert_last_seen_s = None

        if target_priority > current_priority and target in persistence:
            if self._pending_alert_target is None or target_priority > self._pending_alert_priority:
                # A new / strictly stronger escalation starts its own window.
                self._pending_alert_target = target
                self._pending_alert_priority = target_priority
                self._pending_alert_accum_s = dt
                self._pending_alert_last_seen_s = float(now_s)
            elif target_priority == self._pending_alert_priority:
                # Same pending severity is present again -> its evidence time
                # keeps accumulating (this is what makes a repeatedly recurring
                # high risk escalate instead of flickering below the latch).
                self._pending_alert_accum_s += dt
                self._pending_alert_last_seen_s = float(now_s)

            pending = self._pending_alert_target
            if pending is None or self._pending_alert_accum_s < persistence.get(pending, 0.0):
                return self._alert_severity
            self._pending_alert_target = None
            self._pending_alert_priority = 0
            self._pending_alert_accum_s = 0.0
            self._pending_alert_last_seen_s = None

        if target_priority < current_priority:
            if self._recovery_since is None:
                self._recovery_since = now_s
                return self._alert_severity
            if now_s - self._recovery_since < RECOVERY_PERSISTENCE_S:
                return self._alert_severity
            self._recovery_since = None
        else:
            self._recovery_since = None

        self._alert_severity = target
        return target

    def _update_smoothed(self, raw: float | None, now_s: float) -> float | None:
        if raw is None:
            self._smoothed = None
            self._improve_since = None
            self._last_raw = None
            self._last_update_s = None
            return None

        raw = max(0.0, min(100.0, float(raw)))
        if self._smoothed is None:
            self._smoothed = raw
            self._last_raw = raw
            self._last_update_s = now_s
            self._improve_since = None
            return self._smoothed

        current = float(self._smoothed)

        # Rising / equal danger: increase promptly; cancel decrease hold.
        if raw >= current:
            self._smoothed = current + self.risk_increase_alpha * (raw - current)
            self._smoothed = max(0.0, min(100.0, float(self._smoothed)))
            self._improve_since = None
            self._last_raw = raw
            self._last_update_s = now_s
            return self._smoothed

        # Danger returning while still below held risk → stop decreasing.
        if self._last_raw is not None and raw > self._last_raw:
            self._improve_since = None
            self._last_raw = raw
            self._last_update_s = now_s
            return current

        if self._improve_since is None:
            self._improve_since = now_s
            self._last_raw = raw
            self._last_update_s = now_s
            return current

        if now_s - self._improve_since < self.risk_decrease_hold_s:
            self._last_raw = raw
            return current

        last_t = self._last_update_s
        dt = max(0.0, now_s - last_t) if last_t is not None else 0.0
        if dt <= 0.0:
            self._last_raw = raw
            return current

        decay = math.exp(-dt / self.risk_decrease_tau_s)
        self._smoothed = raw + (current - raw) * decay
        self._smoothed = max(0.0, min(100.0, float(self._smoothed)))
        self._last_raw = raw
        self._last_update_s = now_s
        return self._smoothed

    def update(
        self,
        fatigue_risk: float | None,
        distraction: DistractionObservation | None,
        now_s: float,
        face_ok: bool = True,
        location_risk: float = 0.0,
        weather_risk: float = 0.0,
        location_zone: str = "none",
        weather_condition: str = "unknown",
        carla_context: Mapping[str, Any] | None = None,
        fatigue_status: str = "SAFE",
    ) -> AttentionDecision:
        if not face_ok:
            self.reset()
            return AttentionDecision(
                fatigue_risk=0.0,
                distraction_risk=0.0,
                combined_risk=0.0,
                raw_combined=0.0,
                level="SAFE",
                alert_severity="NORMAL",
                primary_reason="No valid face",
                distraction_primary="none",
                distraction_type="none",
                face_ok=False,
                location_risk=0.0,
                weather_risk=0.0,
                location_zone=str(location_zone or "none"),
                weather_condition=str(weather_condition or "unknown"),
                safe_stop=False,
                recommendation="",
                head_direction="unknown",
            )

        f = float(fatigue_risk or 0.0)
        d = float(distraction.distraction_risk) if distraction is not None else 0.0
        loc = max(0.0, min(100.0, float(location_risk or 0.0)))
        wx = max(0.0, min(100.0, float(weather_risk or 0.0)))
        carla = carla_context or {}
        carla_weather = carla.get("weather") or {}
        carla_time = carla.get("time") or {}
        carla_location = carla.get("location") or {}
        carla_speed_kmh = max(0.0, float(carla.get("speed_kmh") or 0.0))
        carla_speed_risk = max(0.0, min(100.0, float(carla.get("speed_risk") or 0.0)))
        carla_time_risk = max(0.0, min(100.0, float(carla_time.get("risk") or 0.0)))
        # Initialise zone and weather_condition from the caller-supplied values,
        # then let CARLA simulated values take precedence when the simulator is
        # connected (CARLA simulated context is the ground truth in simulation).
        zone = str(location_zone or "none")
        wx_cond = str(weather_condition or "unknown")
        if carla.get("carla_connected"):
            # In CARLA simulation, the simulator's weather is the ground truth —
            # real-world API weather is irrelevant, so CARLA values take full
            # precedence and replace caller-supplied values.
            carla_weather_risk = max(0.0, min(100.0, float(carla_weather.get("risk") or 0.0)))
            wx = carla_weather_risk
            carla_cond = str(carla_weather.get("condition") or "unknown")
            wx_cond = carla_cond
            # CARLA simulated location replaces GPS-derived values when connected.
            carla_loc_risk = max(0.0, min(100.0, float(carla_location.get("risk_level") or 0.0)))
            loc = carla_loc_risk
            carla_zone = str(carla_location.get("zone_type") or "none")
            if carla_zone and carla_zone != "none":
                zone = carla_zone
            else:
                # CARLA explicitly says "none" (or empty) — use that, don't invent
                # a zone from the caller-supplied zone_name.
                zone = "none"
        vehicle_state = str(carla.get("vehicle_state") or "UNAVAILABLE")
        time_period = str(carla_time.get("period") or "UNKNOWN")
        head_dir = "unknown"
        if distraction is not None and distraction.head is not None:
            head_dir = str(distraction.head.direction or "unknown")
        dist_type = getattr(distraction, "distraction_type", "none") if distraction is not None else "none"

        distraction_duration_s = 0.0
        if distraction is not None:
            if distraction.primary == "looking_down":
                distraction_duration_s = distraction.looking_down.duration_s
            elif distraction.primary == "head_turn_left":
                distraction_duration_s = distraction.head_turn_left.duration_s
            elif distraction.primary == "head_turn_right":
                distraction_duration_s = distraction.head_turn_right.duration_s
            elif distraction.primary == "passenger_side":
                distraction_duration_s = distraction.passenger_side.duration_s

        # PART 8: what a sustained downward head pose means is decided from the
        # drowsiness state that existed BEFORE the downward pose began (temporal
        # history), never from the current frame alone.
        head_down_classification = self.head_down_classifier.update(
            fatigue_risk=f,
            distraction=distraction,
            now_s=now_s,
            fatigue_status=fatigue_status,
        )

        # Context urgency (ordering only): environmental axes plus the driver
        # state (fatigue + distraction together). It speeds an existing
        # escalation up - it never creates risk or an alert on its own.
        context_urgency = context_urgency_from(
            loc, wx, zone, carla_speed_risk, carla_time_risk, f, d
        )
        raw = fuse_raw_risk(f, d, distraction.primary if distraction is not None else None, distraction_duration_s)
        # Honour configured weights if callers customized them.
        if (
            abs(self.weight_fatigue - WEIGHT_FATIGUE) > 1e-9
            or abs(self.weight_distraction - WEIGHT_DISTRACTION) > 1e-9
        ):
            adjusted_d = d * _distraction_type_factor(distraction.primary if distraction is not None else None, distraction_duration_s)
            raw = min(
                100.0,
                self.weight_fatigue * f + self.weight_distraction * adjusted_d,
            )
            if f >= 40.0 and adjusted_d >= 40.0:
                raw = min(100.0, raw + 8.0)

        # Location / weather amplify the fused driver risk (do not rewrite detectors).
        raw = apply_context_boost(
            raw, d, loc, wx, zone, carla_speed_risk, carla_time_risk
        )

        # PART 8 CASE A: drowsiness evidence that was already present BEFORE a
        # sustained downward head pose makes the combined event drowsiness-grade,
        # not plain distraction. Bounded (floor + fixed gain); CASE B (no prior
        # drowsiness) keeps the normal distraction pathway below.
        if head_down_classification == HEAD_DOWN_DROWSY:
            raw = max(float(raw) + DROWSY_DOWNWARD_GAIN, DROWSY_DOWNWARD_MIN_RISK)

        raw = min(100.0, max(0.0, float(raw)))

        smoothed = self._update_smoothed(raw, now_s)
        if smoothed is not None:
            smoothed = min(100.0, max(0.0, float(smoothed)))
        level = risk_level_from_score(smoothed)
        alert = self._derive_alert_severity(smoothed, level, now_s, context_urgency)

        reason = "Attentive"
        dist_primary = distraction.primary if distraction is not None else "none"
        if smoothed is not None and smoothed > SAFE_MAX:
            if head_down_classification == HEAD_DOWN_DROWSY:
                reason = (
                    "Drowsiness confirmed before a sustained downward posture "
                    "(very high sleepiness risk)"
                )
            elif f >= d and f >= loc and f >= wx:
                if f >= 70.0:
                    reason = "Prolonged eye closure"
                elif f >= 45.0:
                    reason = "Fatigue risk elevated"
                else:
                    reason = "Fatigue risk increasing"
            elif d >= loc and d >= wx and dist_primary != "none":
                if dist_type == "downward":
                    if head_down_classification == HEAD_DOWN_MOBILE_USE:
                        # Behaviour-based classification: sustained downward
                        # attention is mobile-use-like. No phone model exists, so
                        # this never claims the camera detected a phone.
                        reason = (
                            "Sustained downward distraction away from the road "
                            "(mobile-use-like behaviour)"
                        )
                    else:
                        reason = "Sustained downward distraction away from the road"
                elif dist_type == "passenger_conversation":
                    reason = "Sustained passenger-side distraction away from the road with talking activity"
                elif dist_type == "head_turn":
                    side = ""
                    if head_dir in ("left", "right"):
                        side = f" {head_dir}"
                    reason = (
                        f"Sustained head-turn{side} distraction away from the road "
                        "— please focus on the road"
                    )
                else:
                    reason = "Distraction risk elevated"
            elif loc >= wx and loc >= 40.0:
                reason = f"Location risk: {zone.replace('_', ' ')}"
            elif wx >= 40.0:
                reason = f"Weather risk: {wx_cond.replace('_', ' ')}"
            else:
                reason = "Attention risk elevated"

        if smoothed is not None:
            alertness_score = max(0.0, min(100.0, 100.0 - float(smoothed)))
        else:
            alertness_score = 100.0

        contextual_caution = (
            alert == "NORMAL"
            and CAUTION_ALERTNESS_MIN <= alertness_score <= CAUTION_ALERTNESS_MAX
            and (
                wx >= CAUTION_WEATHER_RISK
                or (carla_time_risk >= CAUTION_NIGHT_RISK and wx >= 35.0)
                or loc >= CAUTION_ZONE_RISK
            )
        )
        if contextual_caution:
            alert = "LOW_FATIGUE"
            if reason == "Attentive":
                reason = "Reduced attention in significant driving context"

        safe_stop = level == "CRITICAL"
        recommendation = (
            "Take a safe stop now."
            if safe_stop
            else ("Consider a break soon." if level == "HIGH" else "")
        )
        if safe_stop and reason == "Attentive":
            reason = "Critical attention risk"

        return AttentionDecision(
            fatigue_risk=f,
            distraction_risk=d,
            combined_risk=float(smoothed if smoothed is not None else 0.0),
            raw_combined=float(raw),
            level=level,
            alert_severity=alert,
            primary_reason=reason,
            distraction_primary=dist_primary,
            distraction_type=dist_type,
            face_ok=True,
            alertness_score=alertness_score,
            fatigue_score=f,
            distraction_score=d,
            location_risk=loc,
            weather_risk=wx,
            location_zone=zone,
            weather_condition=wx_cond,
            safe_stop=safe_stop,
            recommendation=recommendation,
            head_direction=head_dir,
            context_urgency=context_urgency,
            carla_speed_kmh=carla_speed_kmh,
            carla_speed_risk=carla_speed_risk,
            carla_vehicle_state=vehicle_state,
            carla_time_period=time_period,
            head_down_classification=head_down_classification,
            pending_escalation=str(self._pending_alert_target or "NONE"),
        )
