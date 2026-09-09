"""
Stage 6B: Continuous driving-duration tracking.

Tracks elapsed session time against a configurable threshold.
Does not classify the driver as DROWSY by itself.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Real project threshold (continuous driving).
DRIVING_DURATION_THRESHOLD_HOURS = 3.0

# Demo/test threshold so the logic can be verified without waiting 3 hours.
DEMO_DRIVING_DURATION_THRESHOLD_SECONDS = 30.0

# When True, the demo threshold is used and the UI must show DEMO MODE.
# Set to False for the real 3-hour project threshold.
USE_DEMO_DRIVING_DURATION = False


@dataclass(frozen=True)
class DrivingDurationResult:
    """Current driving-session duration status."""

    elapsed_s: float
    driving_time_text: str  # HH:MM:SS
    threshold_s: float
    duration_risk: str  # NORMAL / ELEVATED
    threshold_reached: bool
    demo_mode: bool


def format_hms(elapsed_s: float) -> str:
    """Format non-negative seconds as HH:MM:SS."""
    total = max(0, int(elapsed_s))
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def active_threshold_seconds(
    use_demo: bool = USE_DEMO_DRIVING_DURATION,
    demo_seconds: float = DEMO_DRIVING_DURATION_THRESHOLD_SECONDS,
    real_hours: float = DRIVING_DURATION_THRESHOLD_HOURS,
) -> float:
    """Return the threshold currently in force (seconds)."""
    if use_demo:
        return float(demo_seconds)
    return float(real_hours) * 3600.0


class DrivingDurationTracker:
    """
    Session timer for continuous driving duration.

    Start time is fixed when the tracker is created (or when reset() is called).
    Elapsed time is computed from timestamps — it does not reset every frame.
    """

    def __init__(
        self,
        start_s: float,
        use_demo: bool = USE_DEMO_DRIVING_DURATION,
        demo_seconds: float = DEMO_DRIVING_DURATION_THRESHOLD_SECONDS,
        real_hours: float = DRIVING_DURATION_THRESHOLD_HOURS,
    ) -> None:
        self._start_s = float(start_s)
        self.use_demo = bool(use_demo)
        self.demo_seconds = float(demo_seconds)
        self.real_hours = float(real_hours)

    def reset(self, start_s: float) -> None:
        """Restart the session clock from `start_s`."""
        self._start_s = float(start_s)

    @property
    def start_s(self) -> float:
        return self._start_s

    @property
    def threshold_s(self) -> float:
        return active_threshold_seconds(
            use_demo=self.use_demo,
            demo_seconds=self.demo_seconds,
            real_hours=self.real_hours,
        )

    def update(self, now_s: float) -> DrivingDurationResult:
        elapsed = max(0.0, float(now_s) - self._start_s)
        threshold = self.threshold_s
        reached = elapsed >= threshold
        return DrivingDurationResult(
            elapsed_s=elapsed,
            driving_time_text=format_hms(elapsed),
            threshold_s=threshold,
            duration_risk="ELEVATED" if reached else "NORMAL",
            threshold_reached=reached,
            demo_mode=self.use_demo,
        )