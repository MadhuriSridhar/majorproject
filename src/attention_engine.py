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
WEIGHT_FATIGUE = 0.62
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


def fuse_raw_risk(fatigue_risk: float, distraction_risk: float) -> float:
    """Simple transparent weighted fusion + mild dual-signal boost."""
    f = max(0.0, min(100.0, float(fatigue_risk)))
    d = max(0.0, min(100.0, float(distraction_risk)))
    combined = WEIGHT_FATIGUE * f + WEIGHT_DISTRACTION * d
    # Both elevated → small explainable boost (does not invent new detectors).
    if f >= 40.0 and d >= 40.0:
        combined = min(100.0, combined + 8.0)
    # Existing fatigue (and strong distraction) must still surface fully.
    combined = max(combined, f, d * 0.90)
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

        raw = fuse_raw_risk(f, d)
        # Honour configured weights if callers customized them.
        if (
            abs(self.weight_fatigue - WEIGHT_FATIGUE) > 1e-9
            or abs(self.weight_distraction - WEIGHT_DISTRACTION) > 1e-9
        ):
            raw = min(
                100.0,
                max(
                    self.weight_fatigue * f + self.weight_distraction * d,
                    f,
                    d * 0.90,
                ),
            )
            if f >= 40.0 and d >= 40.0:
                raw = min(100.0, raw + 8.0)

        # Location / weather amplify the fused driver risk (do not rewrite detectors).
        raw = apply_context_boost(raw, d, loc, wx, zone)
        raw = min(100.0, max(0.0, float(raw)))

        smoothed = self._update_smoothed(raw, now_s)
        if smoothed is not None:
            smoothed = min(100.0, max(0.0, float(smoothed)))
        level = risk_level_from_score(smoothed)
        # Within MODERATE, escalate soft→ALERT once risk clears former ALERT entry.
        if level == "MODERATE" and smoothed is not None and smoothed >= 35.0:
            alert = "ALERT"
        else:
            alert = LEVEL_TO_ALERT.get(level, "NORMAL")

        reason = "Attentive"
        dist_primary = distraction.primary if distraction is not None else "none"
        if smoothed is not None and smoothed > SAFE_MAX:
            if f >= d and f >= loc and f >= wx:
                reason = "Fatigue risk elevated"
            elif d >= loc and d >= wx and dist_primary != "none":
                reason = f"Distraction: {dist_primary.replace('_', ' ')}"
            elif loc >= wx and loc >= 40.0:
                reason = f"Location context: {zone.replace('_', ' ')}"
            elif wx >= 40.0:
                reason = f"Weather context: {wx_cond.replace('_', ' ')}"
            else:
                reason = "Distraction risk elevated"

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
            location_risk=loc,
            weather_risk=wx,
            location_zone=zone,
            weather_condition=wx_cond,
            safe_stop=safe_stop,
            recommendation=recommendation,
            head_direction=head_dir,
        )
