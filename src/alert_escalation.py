"""
Phase 2: Alert escalation state machine (5 levels).

The FatigueEngine decides severity (NORMAL / LOW_FATIGUE / ALERT / WARNING /
DROWSY). This module only *responds* to that severity - it does NOT
re-implement fatigue detection.

Responsibilities:
- Trigger audio exactly once per severity state ENTRY (not every frame).
- Enforce per-level audio cooldowns.
- Enforce per-level display durations (with countdown).
- Enforce priority: DROWSY > WARNING > ALERT > LOW_FATIGUE > NORMAL.
- Reset alert eligibility on recovery so a later episode can alert again.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Configurable cooldowns / display durations (seconds)
# ---------------------------------------------------------------------------

LOW_FATIGUE_AUDIO_COOLDOWN = 30.0
ALERT_AUDIO_COOLDOWN = 25.0
WARNING_AUDIO_COOLDOWN = 20.0
DROWSY_AUDIO_COOLDOWN = 15.0

# Display windows stay short so overlays do not linger past the audio.
LOW_FATIGUE_DISPLAY_SECONDS = 2.0
ALERT_DISPLAY_SECONDS = 3.0
WARNING_DISPLAY_SECONDS = 4.0
DROWSY_DISPLAY_SECONDS = 5.0

# Global minimum gap between *different* audio alerts. This runs in addition to
# the per-severity cooldowns and prevents LOW_FATIGUE -> ALERT -> WARNING from
# firing back-to-back as if they were one continuous alarm. A genuinely
# higher-priority escalation is still allowed to interrupt within this gap.
GLOBAL_AUDIO_GAP_S = 2.0
EPISODE_RECOVERY_S = 1.5
EPISODE_COOLDOWN_S = 5.0

# Severity priority (higher wins).
SEVERITY_PRIORITY = {"NORMAL": 0, "LOW_FATIGUE": 1, "ALERT": 2, "WARNING": 3, "DROWSY": 4}

# Context urgency bounds. Context urgency scales the per-severity audio cooldown
# ONLY: lower (more urgent) => shorter cooldown => an existing alert may repeat
# sooner under genuinely high-risk conditions. It never changes severity and
# never creates an alert by itself. 1.0 = normal (no effect).
CONTEXT_URGENCY_NORMAL = 1.0
CONTEXT_URGENCY_MIN = 0.5


@dataclass(frozen=True)
class AlertEscalationResult:
    """Pure state-machine output (no audio side effects)."""

    active_severity: str  # The highest-priority severity currently displayed
    should_play_audio: bool  # True only on severity state ENTRY (once)
    audio_severity: str  # Which severity's audio to play
    display_remaining_s: float  # Countdown for the active alert overlay
    display_active: bool  # Whether an alert overlay is currently shown
    audio_eligible: bool  # Whether audio can trigger for the active severity
    reason: str = ""
    contributing_factors: tuple[str, ...] = ()
    alertness_score: float = 0.0
    fatigue_score: float = 0.0
    distraction_score: float = 0.0


class AlertEscalationController:
    """
    Deterministic alert escalation state machine.

    - Tracks the current severity from the FatigueEngine.
    - On severity ENTRY (transition from a lower/other level), sets
      should_play_audio=True exactly once.
    - While the same severity persists, should_play_audio stays False
      (no per-frame audio spam).
    - Cooldown gates repeat audio for the same severity after recovery.
    - Display timers control how long the alert overlay stays visible.
    """

    def __init__(
        self,
        low_fatigue_cooldown_s: float = LOW_FATIGUE_AUDIO_COOLDOWN,
        alert_cooldown_s: float = ALERT_AUDIO_COOLDOWN,
        warning_cooldown_s: float = WARNING_AUDIO_COOLDOWN,
        drowsy_cooldown_s: float = DROWSY_AUDIO_COOLDOWN,
        low_fatigue_display_s: float = LOW_FATIGUE_DISPLAY_SECONDS,
        alert_display_s: float = ALERT_DISPLAY_SECONDS,
        warning_display_s: float = WARNING_DISPLAY_SECONDS,
        drowsy_display_s: float = DROWSY_DISPLAY_SECONDS,
        global_audio_gap_s: float = GLOBAL_AUDIO_GAP_S,
        episode_recovery_s: float = EPISODE_RECOVERY_S,
        episode_cooldown_s: float = EPISODE_COOLDOWN_S,
    ) -> None:
        self.low_fatigue_cooldown_s = float(low_fatigue_cooldown_s)
        self.alert_cooldown_s = float(alert_cooldown_s)
        self.warning_cooldown_s = float(warning_cooldown_s)
        self.drowsy_cooldown_s = float(drowsy_cooldown_s)
        self.low_fatigue_display_s = float(low_fatigue_display_s)
        self.alert_display_s = float(alert_display_s)
        self.warning_display_s = float(warning_display_s)
        self.drowsy_display_s = float(drowsy_display_s)
        self.global_audio_gap_s = float(global_audio_gap_s)
        self.episode_recovery_s = max(0.0, float(episode_recovery_s))
        self.episode_cooldown_s = max(0.0, float(episode_cooldown_s))

        self._current_severity = "NORMAL"
        self._last_audio_severity: str | None = None
        self._last_audio_time_s: float | None = None
        self._display_start_s: float | None = None
        self._display_severity: str | None = None

        # True once a genuine recovery (NORMAL) has been seen, so the next
        # same-level / DROWSY entry starts a fresh episode that can alert again.
        self._pending_recovery = False
        self._episode_active = False
        self._episode_id = 0
        self._episode_started_s: float | None = None
        self._episode_closed_s: float | None = None
        self._recovery_started_s: float | None = None
        self._episode_reason = ""
        self._episode_factors: tuple[str, ...] = ()
        self._last_update_s: float | None = None
        self._context_urgency = CONTEXT_URGENCY_NORMAL

    def reset(self) -> None:
        """Reset all alert state (new session / recovery)."""
        self._current_severity = "NORMAL"
        self._last_audio_severity = None
        self._last_audio_time_s = None
        self._display_start_s = None
        self._display_severity = None
        self._pending_recovery = False
        self._episode_active = False
        self._episode_id = 0
        self._episode_started_s = None
        self._episode_closed_s = None
        self._recovery_started_s = None
        self._episode_reason = ""
        self._episode_factors = ()
        self._last_update_s = None

    @property
    def debug_state(self) -> dict:
        """Current alert state for diagnostics without per-frame logging."""
        now = self._last_update_s
        return {
            "active_alert_episode": self._episode_active,
            "episode_id": self._episode_id,
            "primary_reason": self._episode_reason,
            "recovery_state": "RECOVERING" if self._recovery_started_s is not None else ("ACTIVE" if self._episode_active else "IDLE"),
            "confirmed_severity": self._current_severity,
            "time_in_current_state": None if now is None or self._display_start_s is None else max(0.0, now - self._display_start_s),
        }

    def _cooldown_for(self, severity: str) -> float:
        if severity == "LOW_FATIGUE":
            return self.low_fatigue_cooldown_s
        if severity == "ALERT":
            return self.alert_cooldown_s
        if severity == "WARNING":
            return self.warning_cooldown_s
        if severity == "DROWSY":
            return self.drowsy_cooldown_s
        return 0.0

    def _effective_cooldown_for(self, severity: str) -> float:
        """Cooldown scaled by the current context urgency.

        A high-risk context (urgency < 1.0) shortens the cooldown so an
        existing alert becomes eligible to repeat sooner. Normal context
        (urgency = 1.0) leaves the configured cooldown unchanged.
        """
        return self._cooldown_for(severity) * self._context_urgency

    def _display_duration_for(self, severity: str) -> float:
        if severity == "LOW_FATIGUE":
            return self.low_fatigue_display_s
        if severity == "ALERT":
            return self.alert_display_s
        if severity == "WARNING":
            return self.warning_display_s
        if severity == "DROWSY":
            return self.drowsy_display_s
        return 0.0

    def _is_audio_eligible(self, severity: str, now_s: float) -> bool:
        """Audio can trigger if cooldown has elapsed since last audio for this severity."""
        if self._last_audio_severity != severity:
            return True
        if self._last_audio_time_s is None:
            return True
        return (now_s - self._last_audio_time_s) >= self._effective_cooldown_for(severity)

    def _global_gap_ok(self, severity: str, now_s: float) -> bool:
        """Enforce a minimum gap between different audio alerts.

        A genuinely higher-priority escalation is still allowed to interrupt a
        lower-priority alert within the gap; a critical / drowsy escalation is
        never blocked simply because a lower alert was recently played.
        """
        if self._last_audio_time_s is None:
            return True
        if now_s - self._last_audio_time_s >= self.global_audio_gap_s:
            return True
        last_priority = SEVERITY_PRIORITY.get(self._last_audio_severity or "NORMAL", 0)
        return SEVERITY_PRIORITY[severity] > last_priority

    def update(
        self,
        severity: str,
        now_s: float,
        reason: str | None = None,
        contributing_factors: tuple[str, ...] | list[str] | None = None,
        alertness_score: float | None = None,
        fatigue_score: float | None = None,
        distraction_score: float | None = None,
        context_urgency: float = 1.0,
    ) -> AlertEscalationResult:
        """
        Feed the latest attention severity and a monotonic timestamp.

        Returns the alert state for the UI / audio system.
        """
        # Normalize unknown severities to NORMAL.
        if severity not in SEVERITY_PRIORITY:
            severity = "NORMAL"

        reason_text = str(reason or "")
        factors = tuple(str(item) for item in (contributing_factors or ()) if item)
        score = float(alertness_score if alertness_score is not None else 0.0)
        fatigue = float(fatigue_score if fatigue_score is not None else 0.0)
        distraction = float(distraction_score if distraction_score is not None else 0.0)
        self._context_urgency = max(
            CONTEXT_URGENCY_MIN,
            min(
                CONTEXT_URGENCY_NORMAL,
                float(context_urgency if context_urgency is not None else 1.0),
            ),
        )

        prev = self._current_severity
        self._last_update_s = now_s

        if self._episode_active and self._episode_reason:
            reason_text = self._episode_reason
            factors = self._episode_factors or factors

        # A NORMAL feed marks recovery so a later same-level episode can alert
        # again (including when NORMAL arrives while a display hold is still
        # showing the previous alert).
        if severity == "NORMAL":
            if self._episode_active:
                if self._recovery_started_s is None:
                    self._recovery_started_s = now_s
                if now_s - self._recovery_started_s < self.episode_recovery_s:
                    severity = prev
                else:
                    self._episode_active = False
                    self._episode_closed_s = now_s
                    self._recovery_started_s = None
                    self._pending_recovery = True
                    self._display_severity = None
                    self._display_start_s = None
                    self._current_severity = "NORMAL"
            else:
                self._pending_recovery = True
        elif self._recovery_started_s is not None:
            if now_s - self._recovery_started_s >= self.episode_recovery_s:
                self._episode_active = False
                self._episode_closed_s = self._recovery_started_s + self.episode_recovery_s
                self._recovery_started_s = None
                self._pending_recovery = True
                self._display_severity = None
                self._display_start_s = None
                self._current_severity = "NORMAL"
            else:
                # A brief deterioration during recovery belongs to the same
                # episode and must not create a new audio/popup event.
                self._recovery_started_s = None
                self._pending_recovery = False

        episode_cooldown_active = (
            not self._episode_active
            and self._episode_closed_s is not None
            and now_s - self._episode_closed_s < self.episode_cooldown_s
        )

        # Determine whether a higher-priority alert display is being held open
        # (its display window has not yet expired).
        held_severity = self._display_severity
        hold_active = (
            held_severity is not None
            and self._display_start_s is not None
            and (now_s - self._display_start_s) < self._display_duration_for(held_severity)
        )

        # Commit the inflow live severity as the authoritative current state
        # UNLESS a higher-priority alert display is being held open (ensures a
        # transient lower dip while a higher alert is active is NOT treated as a
        # fresh entry). A genuine recovery (NORMAL) and an escalation ABOVE the
        # held severity still commit normally so cooldown / re-entry resolution
        # keeps working.
        if (
            not hold_active
            or severity == "NORMAL"
            or (held_severity is not None and SEVERITY_PRIORITY[severity] > SEVERITY_PRIORITY[held_severity])
        ):
            self._current_severity = severity

        # Sticky DROWSY: while the DROWSY alert's own display window is still
        # open, temporary dips below the DROWSY threshold (WARNING / ALERT /
        # LOW_FATIGUE) must NOT downgrade the alert or re-trigger WARNING /
        # DROWSY audio. The relatch is bounded by the DROWSY display window so
        # a genuinely-lower live severity can resume once DROWSY expires.
        relatch_active = (
            held_severity == "DROWSY"
            and self._display_start_s is not None
            and (now_s - self._display_start_s) < self._display_duration_for("DROWSY")
        )
        if (
            relatch_active
            and not self._pending_recovery
            and prev == "DROWSY"
            and severity in ("WARNING", "ALERT", "LOW_FATIGUE")
        ):
            severity = "DROWSY"

        # If a higher-priority alert is already active, allow it to keep
        # showing until its display window expires even if the incoming
        # severity briefly drops to NORMAL or a lower value. This avoids the
        # "alert disappears as soon as eyes reopen" bug.
        active_severity = self._display_severity
        if active_severity is not None:
            remaining = self._display_duration_for(active_severity) - (now_s - self._display_start_s)
            if remaining > 0.0:
                current_active = active_severity
                if severity == "NORMAL":
                    return AlertEscalationResult(
                        active_severity=current_active,
                        should_play_audio=False,
                        audio_severity="NORMAL",
                        display_remaining_s=max(0.0, remaining),
                        display_active=True,
                        audio_eligible=self._is_audio_eligible(current_active, now_s),
                        reason=reason_text,
                        contributing_factors=factors,
                        alertness_score=score,
                        fatigue_score=fatigue,
                        distraction_score=distraction,
                    )
                if SEVERITY_PRIORITY[severity] <= SEVERITY_PRIORITY[current_active]:
                    # After a genuine recovery (NORMAL), a same-level
                    # re-entry must start a fresh episode instead of being
                    # held by the old display window.
                    if self._pending_recovery and SEVERITY_PRIORITY[severity] == SEVERITY_PRIORITY[current_active]:
                        pass  # fall through to fresh-entry resolution
                    else:
                        return AlertEscalationResult(
                            active_severity=current_active,
                            should_play_audio=False,
                            audio_severity="NORMAL",
                            display_remaining_s=max(0.0, remaining),
                            display_active=True,
                            audio_eligible=self._is_audio_eligible(current_active, now_s),
                            reason=reason_text,
                            contributing_factors=factors,
                            alertness_score=score,
                            fatigue_score=fatigue,
                            distraction_score=distraction,
                        )

        # No active alert is still within its display window, so resolve the
        # incoming severity normally.
        if severity == "NORMAL":
            self._display_severity = None
            self._display_start_s = None
            self._current_severity = "NORMAL"
            return AlertEscalationResult(
                active_severity="NORMAL",
                should_play_audio=False,
                audio_severity="NORMAL",
                display_remaining_s=0.0,
                display_active=False,
                audio_eligible=False,
                reason=reason_text,
                contributing_factors=factors,
                alertness_score=score,
                fatigue_score=fatigue,
                distraction_score=distraction,
            )

        if episode_cooldown_active:
            self._current_severity = "NORMAL"
            return AlertEscalationResult(
                active_severity="NORMAL",
                should_play_audio=False,
                audio_severity="NORMAL",
                display_remaining_s=0.0,
                display_active=False,
                audio_eligible=False,
                reason=reason_text,
                contributing_factors=factors,
                alertness_score=score,
                fatigue_score=fatigue,
                distraction_score=distraction,
            )

        # Severity ENTRY: transition from a different severity.
        is_entry = severity != prev

        # Audio trigger: only on entry, only when not downgrading below the
        # previously active severity, AND cooldown elapsed, AND the global
        # minimum gap between different alerts has passed. A lower-priority
        # alert must never fight a higher-priority one.
        is_downgrade = SEVERITY_PRIORITY[severity] < SEVERITY_PRIORITY[prev]
        should_play = False
        if (
            is_entry
            and not is_downgrade
            and self._is_audio_eligible(severity, now_s)
            and self._global_gap_ok(severity, now_s)
        ):
            should_play = True
            self._last_audio_severity = severity
            self._last_audio_time_s = now_s

        # Display: start a new display timer on entry (or when a higher
        # severity takes over). A fresh entry also clears the recovery latch.
        if is_entry or self._display_severity != severity:
            if not self._episode_active:
                self._episode_active = True
                self._episode_id += 1
                self._episode_started_s = now_s
                self._episode_reason = reason_text
                self._episode_factors = factors
            self._display_severity = severity
            self._display_start_s = now_s
            self._pending_recovery = False
            self._current_severity = severity

        # Compute remaining display time.
        assert self._display_start_s is not None
        elapsed_display = now_s - self._display_start_s
        duration = self._display_duration_for(severity)
        remaining = max(0.0, duration - elapsed_display)
        display_active = remaining > 0.0

        return AlertEscalationResult(
            active_severity=severity,
            should_play_audio=should_play,
            audio_severity=severity if should_play else "NORMAL",
            display_remaining_s=remaining,
            display_active=display_active,
            audio_eligible=self._is_audio_eligible(severity, now_s),
            reason=reason_text,
            contributing_factors=factors,
            alertness_score=score,
            fatigue_score=fatigue,
            distraction_score=distraction,
        )