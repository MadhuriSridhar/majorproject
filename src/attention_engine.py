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

from distraction_monitor import DistractionObservation

# Transparent fusion weights (fatigue remains the dominant contributor).
WEIGHT_FATIGUE = 0.90
WEIGHT_DISTRACTION = 0.38

# Soft context gains (location / weather amplify; they do not replace driver risk).
LOCATION_CONTEXT_GAIN = 0.22
WEATHER_CONTEXT_GAIN = 0.20
CONTEXT_MAX_SOFT = 18.0

# Level bands.
SAFE_MAX = 24
MODERATE_MAX = 49
HIGH_MAX = 74
# >= HIGH_MAX+ → CRITICAL

# ALERT sits inside the MODERATE band. Enter ALERT at ALERT_ENTRY, but do not
# immediately drop back to LOW_FATIGUE just below it: use a lower exit
# threshold (hysteresis) so LOW_FATIGUE <-> ALERT flapping is avoided.
ALERT_ENTRY = 35.0
ALERT_EXIT = 32.0

# WARNING (HIGH) requires a short persistence before it is emitted so a
# single-frame / very brief risk spike cannot fire a serious alarm.
ALERT_PERSISTENCE_S = 0.75
WARNING_PERSISTENCE_S = 1.00
DROWSY_PERSISTENCE_S = 1.50
RECOVERY_PERSISTENCE_S = 1.25

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
    boost = 0.0

    # Soft proportional context when there is already meaningful driver risk.
    if base >= 20.0 or d >= 30.0:
        soft = min(
            CONTEXT_MAX_SOFT,
            LOCATION_CONTEXT_GAIN * loc + WEATHER_CONTEXT_GAIN * wx,
        )
        boost += soft * (0.35 + base / 200.0)

    # Moderate distraction + elevated road zone (school / construction / sharp turn).
    if d >= 35.0 and loc >= 55.0:
        boost += 12.0
    if d >= 35.0 and zone_type == "school":
        boost += 6.0

    # Moderate distraction + rain / severe weather.
    if d >= 35.0 and wx >= 50.0:
        boost += 10.0

    # Moderate distraction + school/high-risk zone + heavy rain → critical territory.
    if d >= 35.0 and loc >= 55.0 and wx >= 70.0:
        boost += 14.0

    return float(min(100.0, base + boost))


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
        self.reset()

    def reset(self) -> None:
        self._smoothed: float | None = None
        self._improve_since: float | None = None
        self._last_raw: float | None = None
        self._last_update_s: float | None = None
        self._alert_severity = "NORMAL"
        self._warning_pending_since: float | None = None
        self._pending_alert_target: str | None = None
        self._pending_alert_since: float | None = None
        self._recovery_since: float | None = None

    def _derive_alert_severity(
        self, smoothed: float | None, level: str, now_s: float
    ) -> str:
        """Map smoothed attention risk to an alert severity with hysteresis.

        - LOW_FATIGUE <-> ALERT uses entry/exit hysteresis inside MODERATE.
        - WARNING (HIGH) requires a short persistence before being emitted.
        - CRITICAL escalates to DROWSY immediately (genuinely severe evidence).
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
        persistence = {
            "ALERT": ALERT_PERSISTENCE_S,
            "WARNING": WARNING_PERSISTENCE_S,
            "DROWSY": DROWSY_PERSISTENCE_S,
        }
        current_priority = priority.get(self._alert_severity, 0)
        target_priority = priority.get(target, 0)
        if target_priority > current_priority and target in persistence:
            if self._pending_alert_target != target or self._pending_alert_since is None:
                self._pending_alert_target = target
                self._pending_alert_since = now_s
                return self._alert_severity
            if now_s - self._pending_alert_since < persistence[target]:
                return self._alert_severity
            self._pending_alert_target = None
            self._pending_alert_since = None
        elif target_priority <= current_priority:
            self._pending_alert_target = None
            self._pending_alert_since = None

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
        zone = str(location_zone or "none")
        wx_cond = str(weather_condition or "unknown")
        head_dir = "unknown"
        if distraction is not None and distraction.head is not None:
            head_dir = str(distraction.head.direction or "unknown")

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
        raw = apply_context_boost(raw, d, loc, wx, zone)
        raw = min(100.0, max(0.0, float(raw)))

        smoothed = self._update_smoothed(raw, now_s)
        if smoothed is not None:
            smoothed = min(100.0, max(0.0, float(smoothed)))
        level = risk_level_from_score(smoothed)
        alert = self._derive_alert_severity(smoothed, level, now_s)

        reason = "Attentive"
        dist_primary = distraction.primary if distraction is not None else "none"
        if smoothed is not None and smoothed > SAFE_MAX:
            if f >= d and f >= loc and f >= wx:
                if f >= 70.0:
                    reason = "Prolonged eye closure"
                elif f >= 45.0:
                    reason = "Fatigue risk elevated"
                else:
                    reason = "Fatigue risk increasing"
            elif d >= loc and d >= wx and dist_primary != "none":
                if dist_primary == "looking_down":
                    reason = "Sustained attention away from the road"
                elif dist_primary.startswith("head_turn"):
                    reason = "Head direction away from the road"
                elif dist_primary == "passenger_side":
                    reason = "Extended lateral attention"
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
        )
