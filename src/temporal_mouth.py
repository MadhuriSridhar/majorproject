"""
Stage 5: Temporal mouth-opening / yawn event tracking.

Independent of Stage 4 EyeClosureTracker.
Does not implement fatigue scoring or alarms.
"""

from __future__ import annotations

from dataclasses import dataclass

from mouth_features import MAR_YAWN_THRESHOLD

# ---------------------------------------------------------------------------
# Configuration (starting values only — tune with your laptop camera later)
# ---------------------------------------------------------------------------

# Start (or continue) a mouth-open event when MAR is at/above this value.
MAR_OPEN_THRESHOLD = MAR_YAWN_THRESHOLD

# Hysteresis: while already open, require MAR to fall below this to close.
MAR_CLOSE_THRESHOLD = MAR_YAWN_THRESHOLD - 0.08

# Mouth must stay open at least this long to count exactly one yawn.
YAWN_MIN_DURATION_S = 1.20


@dataclass(frozen=True)
class TemporalMouthState:
    """Live temporal mouth / yawn status for one update() call."""

    mar: float | None
    mouth: str  # OPEN / CLOSED / UNKNOWN
    current_mouth_open_s: float
    yawn_count: int
    yawn: bool  # True while a counted yawn event is still in progress
    yawn_frequency: float | None = None
    recent_yawn_frequency: float | None = None
    yawn_duration_s: float | None = None


class YawnTracker:
    """
    Tracks sustained mouth opening to detect yawn events.

    - Starts an open event when MAR >= MAR_OPEN_THRESHOLD.
    - Ends it when MAR falls below MAR_CLOSE_THRESHOLD.
    - Counts exactly one yawn once open duration reaches YAWN_MIN_DURATION_S.
    - Does not increment again for the same continuous open event.
    - Missing MAR (no face) aborts an in-progress event without a new yawn.
    """

    def __init__(
        self,
        open_threshold: float = MAR_OPEN_THRESHOLD,
        close_threshold: float = MAR_CLOSE_THRESHOLD,
        yawn_min_s: float = YAWN_MIN_DURATION_S,
    ) -> None:
        if close_threshold > open_threshold:
            raise ValueError("close_threshold must be <= open_threshold")

        self.open_threshold = open_threshold
        self.close_threshold = close_threshold
        self.yawn_min_s = yawn_min_s

        self.yawn_count = 0
        self._open = False
        self._open_start_s: float | None = None
        self._yawn_counted_for_event = False
        self._yawn_timestamps: list[float] = []
        self._monitoring_start_s: float | None = None

    def reset(self) -> None:
        self.yawn_count = 0
        self._yawn_timestamps.clear()
        self._monitoring_start_s = None
        self._abort_open()

    def _abort_open(self) -> None:
        self._open = False
        self._open_start_s = None
        self._yawn_counted_for_event = False

    def _rate_for_window(self, timestamp_s: float, window_s: float) -> float | None:
        if not self._yawn_timestamps:
            return None
        cutoff = timestamp_s - window_s
        count = sum(1 for ts in self._yawn_timestamps if ts >= cutoff)
        # Cold-start: while the session has been running for less than the full
        # window, divide by the actual elapsed monitoring time instead of the
        # full window so the displayed rate is not artificially low.
        divisor = window_s
        if self._monitoring_start_s is not None:
            elapsed = timestamp_s - self._monitoring_start_s
            if elapsed > 0.0:
                divisor = min(window_s, elapsed)
        return (count / divisor) * 60.0 if divisor > 0 else float(count)

    def _duration(self, timestamp_s: float) -> float:
        if self._open_start_s is None:
            return 0.0
        return max(0.0, timestamp_s - self._open_start_s)

    def update(self, mar: float | None, timestamp_s: float) -> TemporalMouthState:
        """
        Feed one frame's MAR and a monotonic timestamp (seconds).

        Returns the updated temporal state for UI display.
        """
        if self._monitoring_start_s is None:
            self._monitoring_start_s = timestamp_s

        if mar is None:
            self._abort_open()
            return TemporalMouthState(
                mar=None,
                mouth="UNKNOWN",
                current_mouth_open_s=0.0,
                yawn_count=self.yawn_count,
                yawn=False,
                yawn_frequency=None,
                recent_yawn_frequency=None,
                yawn_duration_s=None,
            )

        if not self._open:
            should_be_open = mar >= self.open_threshold
        else:
            # Hysteresis: stay open until MAR clearly falls.
            should_be_open = mar >= self.close_threshold

        if should_be_open and not self._open:
            self._open = True
            self._open_start_s = timestamp_s
            self._yawn_counted_for_event = False

        elif should_be_open and self._open:
            duration = self._duration(timestamp_s)
            if (not self._yawn_counted_for_event) and duration >= self.yawn_min_s:
                self.yawn_count += 1
                self._yawn_timestamps.append(timestamp_s)
                self._yawn_counted_for_event = True

        elif (not should_be_open) and self._open:
            # Mouth closed: end event. Count is kept; active yawn flag clears.
            self._abort_open()

        duration_now = self._duration(timestamp_s) if self._open else 0.0
        yawn_active = self._open and self._yawn_counted_for_event
        yawn_frequency = self._rate_for_window(timestamp_s, 60.0)
        recent_yawn_frequency = self._rate_for_window(timestamp_s, 15.0)

        return TemporalMouthState(
            mar=mar,
            mouth="OPEN" if self._open else "CLOSED",
            current_mouth_open_s=duration_now,
            yawn_count=self.yawn_count,
            yawn=yawn_active,
            yawn_frequency=yawn_frequency,
            recent_yawn_frequency=recent_yawn_frequency,
            yawn_duration_s=duration_now if yawn_active else None,
        )