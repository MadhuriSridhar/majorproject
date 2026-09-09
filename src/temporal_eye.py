"""
Stage 4: Temporal eye-closure analysis (blink vs prolonged closure).

Uses average EAR over time. Does not change Stage 3 EAR math.
Does not implement fatigue scoring or alarms.
"""

from __future__ import annotations

from dataclasses import dataclass

from eye_features import EAR_CLOSED_THRESHOLD

# ---------------------------------------------------------------------------
# Configuration (starting values only — tune with your laptop camera later)
# ---------------------------------------------------------------------------

# Close when EAR falls below this (same Stage 3 closed threshold by default).
EAR_CLOSE_THRESHOLD = EAR_CLOSED_THRESHOLD

# Small hysteresis: while already closed, require EAR to rise above this
# before counting the eyes as open again. Reduces flicker near the threshold.
EAR_OPEN_THRESHOLD = EAR_CLOSED_THRESHOLD + 0.02

# Closure duration counted as one normal blink when eyes reopen.
MIN_BLINK_DURATION_S = 0.10
MAX_BLINK_DURATION_S = 0.80

# Continuous closure at/above this duration is prolonged-closure evidence.
PROLONGED_CLOSURE_S = 1.50

# UI label: early part of a closure shows CLOSING, then CLOSED.
CLOSING_LABEL_MAX_S = 0.20


@dataclass(frozen=True)
class TemporalEyeState:
    """Live temporal eye status for one update() call."""

    ear: float | None
    eye: str  # OPEN / CLOSING / CLOSED / UNKNOWN
    blink_count: int
    current_closure_s: float
    prolonged_closure: bool
    blink_rate: float | None = None
    recent_blink_rate: float | None = None


class EyeClosureTracker:
    """
    Tracks eye-closure events over time.

    - Starts a closure when EAR drops below EAR_CLOSE_THRESHOLD.
    - Ends it when EAR rises above EAR_OPEN_THRESHOLD.
    - Counts exactly one blink for short closures in [min, max] seconds.
    - Marks prolonged closure after PROLONGED_CLOSURE_S of continuous closure.
    - Prolonged events are not counted as blinks when the eyes reopen.
    - Missing EAR (no face) aborts an in-progress closure without a blink.
    """

    def __init__(
        self,
        close_threshold: float = EAR_CLOSE_THRESHOLD,
        open_threshold: float = EAR_OPEN_THRESHOLD,
        min_blink_s: float = MIN_BLINK_DURATION_S,
        max_blink_s: float = MAX_BLINK_DURATION_S,
        prolonged_s: float = PROLONGED_CLOSURE_S,
        closing_label_max_s: float = CLOSING_LABEL_MAX_S,
    ) -> None:
        if open_threshold < close_threshold:
            raise ValueError("open_threshold must be >= close_threshold")

        self.close_threshold = close_threshold
        self.open_threshold = open_threshold
        self.min_blink_s = min_blink_s
        self.max_blink_s = max_blink_s
        self.prolonged_s = prolonged_s
        self.closing_label_max_s = closing_label_max_s

        self.blink_count = 0
        self._closed = False
        self._closure_start_s: float | None = None
        self._prolonged = False
        self._blink_counted_for_event = False
        self._blink_timestamps: list[float] = []

    def reset(self) -> None:
        """Clear tracker state (does not change configuration)."""
        self.blink_count = 0
        self._blink_timestamps.clear()
        self._abort_closure()

    def _abort_closure(self) -> None:
        """Stop an in-progress closure without counting a blink."""
        self._closed = False
        self._closure_start_s = None
        self._prolonged = False
        self._blink_counted_for_event = False

    def _rate_for_window(self, timestamp_s: float, window_s: float) -> float | None:
        if not self._blink_timestamps:
            return None
        cutoff = timestamp_s - window_s
        count = sum(1 for ts in self._blink_timestamps if ts >= cutoff)
        return (count / window_s) * 60.0 if window_s > 0 else float(count)

    def _duration(self, timestamp_s: float) -> float:
        if self._closure_start_s is None:
            return 0.0
        return max(0.0, timestamp_s - self._closure_start_s)

    def _eye_label(self, timestamp_s: float) -> str:
        if not self._closed:
            return "OPEN"
        duration = self._duration(timestamp_s)
        if duration < self.closing_label_max_s:
            return "CLOSING"
        return "CLOSED"

    def update(self, average_ear: float | None, timestamp_s: float) -> TemporalEyeState:
        """
        Feed one frame's average EAR and a monotonic timestamp (seconds).

        Returns the updated temporal state for UI display.
        """
        if average_ear is None:
            # Do not keep a closure timer running without valid eye data.
            self._abort_closure()
            return TemporalEyeState(
                ear=None,
                eye="UNKNOWN",
                blink_count=self.blink_count,
                current_closure_s=0.0,
                prolonged_closure=False,
                blink_rate=None,
                recent_blink_rate=None,
            )

        if not self._closed:
            should_be_closed = average_ear < self.close_threshold
        else:
            # Hysteresis: stay closed until EAR clearly rises.
            should_be_closed = average_ear < self.open_threshold

        if should_be_closed and not self._closed:
            self._closed = True
            self._closure_start_s = timestamp_s
            self._prolonged = False
            self._blink_counted_for_event = False

        elif should_be_closed and self._closed:
            duration = self._duration(timestamp_s)
            if duration >= self.prolonged_s:
                self._prolonged = True
                self._blink_counted_for_event = True

        elif (not should_be_closed) and self._closed:
            duration = self._duration(timestamp_s)
            was_prolonged = self._prolonged
            # Count one blink only for short, completed, non-prolonged closures.
            if (
                (not was_prolonged)
                and (not self._blink_counted_for_event)
                and duration >= self.min_blink_s
            ):
                self.blink_count += 1
                self._blink_timestamps.append(timestamp_s)
                self._blink_counted_for_event = True
            self._abort_closure()

        duration_now = self._duration(timestamp_s) if self._closed else 0.0

        blink_rate = self._rate_for_window(timestamp_s, 60.0)
        recent_blink_rate = self._rate_for_window(timestamp_s, 15.0)

        return TemporalEyeState(
            ear=average_ear,
            eye=self._eye_label(timestamp_s),
            blink_count=self.blink_count,
            current_closure_s=duration_now,
            prolonged_closure=self._prolonged,
            blink_rate=blink_rate,
            recent_blink_rate=recent_blink_rate,
        )