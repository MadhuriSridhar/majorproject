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

# Severity priority (higher wins).
SEVERITY_PRIORITY = {"NORMAL": 0, "LOW_FATIGUE": 1, "ALERT": 2, "WARNING": 3, "DROWSY": 4}


@dataclass(frozen=True)
class AlertEscalationResult:
    """Pure state-machine output (no audio side effects)."""

    active_severity: str  # The highest-priority severity currently displayed
    should_play_audio: bool  # True only on severity state ENTRY (once)
    audio_severity: str  # Which severity's audio to play
    display_remaining_s: float  # Countdown for the active alert overlay
    display_active: bool  # Whether an alert overlay is currently shown
    audio_eligible: bool  # Whether audio can trigger for the active severity


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
    ) -> None:
        self.low_fatigue_cooldown_s = float(low_fatigue_cooldown_s)
        self.alert_cooldown_s = float(alert_cooldown_s)
        self.warning_cooldown_s = float(warning_cooldown_s)
        self.drowsy_cooldown_s = float(drowsy_cooldown_s)
        self.low_fatigue_display_s = float(low_fatigue_display_s)
        self.alert_display_s = float(alert_display_s)
        self.warning_display_s = float(warning_display_s)
        self.drowsy_display_s = float(drowsy_display_s)

        self._current_severity = "NORMAL"
        self._last_audio_severity: str | None = None
        self._last_audio_time_s: float | None = None
        self._display_start_s: float | None = None
        self._display_severity: str | None = None

        # True once a genuine recovery (NORMAL) has been seen, so the next
        # same-level / DROWSY entry starts a fresh episode that can alert again.
        self._pending_recovery = False

    def reset(self) -> None:
        """Reset all alert state (new session / recovery)."""
        self._current_severity = "NORMAL"
        self._last_audio_severity = None
        self._last_audio_time_s = None
        self._display_start_s = None
        self._display_severity = None
        self._pending_recovery = False

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
        return (now_s - self._last_audio_time_s) >= self._cooldown_for(severity)

    def update(self, severity: str, now_s: float) -> AlertEscalationResult:
        """
        Feed the latest FatigueEngine severity and a monotonic timestamp.

        Returns the alert state for the UI / audio system.
        """
        # Normalize unknown severities to NORMAL.
        if severity not in SEVERITY_PRIORITY:
            severity = "NORMAL"

        prev = self._current_severity

        # A NORMAL feed marks recovery so a later same-level episode can alert
        # again (including when NORMAL arrives while a display hold is still
        # showing the previous alert).
        if severity == "NORMAL":
            self._pending_recovery = True

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
            )

        # Severity ENTRY: transition from a different severity.
        is_entry = severity != prev

        # Audio trigger: only on entry, only when not downgrading below the
        # previously active severity, AND cooldown elapsed. A lower-priority
        # alert must never fight a higher-priority one.
        is_downgrade = SEVERITY_PRIORITY[severity] < SEVERITY_PRIORITY[prev]
        should_play = False
        if is_entry and not is_downgrade and self._is_audio_eligible(severity, now_s):
            should_play = True
            self._last_audio_severity = severity
            self._last_audio_time_s = now_s

        # Display: start a new display timer on entry (or when a higher
        # severity takes over). A fresh entry also clears the recovery latch.
        if is_entry or self._display_severity != severity:
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
        )