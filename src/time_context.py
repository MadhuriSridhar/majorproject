"""
Stage 6A: Time-of-day context (DAY / NIGHT).

Uses the computer's local system time only.
Does not classify the driver as DROWSY by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time

# ---------------------------------------------------------------------------
# Configuration (project requirement defaults)
# ---------------------------------------------------------------------------

# NIGHT window: 10:00 PM through 6:00 AM (crosses midnight).
NIGHT_START = time(22, 0, 0)  # 10:00 PM
NIGHT_END = time(6, 0, 0)  # 6:00 AM

# Optional demo override so you can verify NIGHT without changing the PC clock.
# Examples: "23:15:32", "14:30:10", or None to use real local time.
DEMO_FORCE_LOCAL_TIME: str | None = None
# For Friday night-context testing ONLY (do not change the PC clock), temporarily set:
# DEMO_FORCE_LOCAL_TIME = "23:15:32"
# Leave as None for normal / daytime demo operation.


@dataclass(frozen=True)
class TimeContextResult:
    """Current time-of-day context for display / later fatigue fusion."""

    local_time: datetime
    time_text: str  # HH:MM:SS
    context: str  # DAY / NIGHT
    night_context_active: bool
    using_forced_time: bool


def _parse_hms(value: str) -> time:
    parts = value.strip().split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"Expected HH:MM or HH:MM:SS, got {value!r}")
    hour = int(parts[0])
    minute = int(parts[1])
    second = int(parts[2]) if len(parts) == 3 else 0
    return time(hour, minute, second)


def is_night(
    moment: datetime | time,
    night_start: time = NIGHT_START,
    night_end: time = NIGHT_END,
) -> bool:
    """
    Return True if the given local time falls in the night window.

    Supports midnight crossing when night_start > night_end
    (e.g. 22:00 -> 06:00).
    """
    current = moment.time() if isinstance(moment, datetime) else moment

    if night_start == night_end:
        # Degenerate config: treat entire day as night.
        return True

    if night_start < night_end:
        # Same-day window, e.g. 01:00 -> 05:00
        return night_start <= current < night_end

    # Crosses midnight, e.g. 22:00 -> 06:00
    return current >= night_start or current < night_end


def get_time_context(
    now: datetime | None = None,
    night_start: time = NIGHT_START,
    night_end: time = NIGHT_END,
    force_local_time: str | None = DEMO_FORCE_LOCAL_TIME,
) -> TimeContextResult:
    """
    Build the current DAY/NIGHT context.

    Priority for the clock source:
    1. Explicit `now` argument (used by unit tests)
    2. DEMO_FORCE_LOCAL_TIME / force_local_time override (demo verification)
    3. Real local system time
    """
    using_forced = False

    if now is not None:
        local = now
    elif force_local_time:
        forced = _parse_hms(force_local_time)
        base = datetime.now().replace(
            hour=forced.hour,
            minute=forced.minute,
            second=forced.second,
            microsecond=0,
        )
        local = base
        using_forced = True
    else:
        local = datetime.now()

    night = is_night(local, night_start=night_start, night_end=night_end)
    context = "NIGHT" if night else "DAY"

    return TimeContextResult(
        local_time=local,
        time_text=local.strftime("%H:%M:%S"),
        context=context,
        night_context_active=night,
        using_forced_time=using_forced,
    )