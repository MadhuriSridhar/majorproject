"""Event-level (not per-frame) alert-decision logging.

Produces one concise, self-contained line per meaningful alert-decision state
change so a future debugger can reconstruct why an alert fired or was
suppressed. The formatting helper is pure and unit-tested; nothing here plays
audio or changes any decision.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("driver_attention.alert")


def format_alert_decision(
    severity: str,
    reason: str,
    alertness: float | None,
    fatigue: float | None,
    distraction: float | None,
    context_urgency: float | None,
    head_direction: str,
    eye_state: str,
    face_detected: bool,
    audio: str,
    suppressed: bool = False,
    suppression_reason: str = "",
) -> str:
    """Render one alert-decision record as a single human-readable line."""

    def _n(value: float | None) -> str:
        return "--" if value is None else f"{float(value):.0f}"

    parts = [
        "ALERT DECISION" if not suppressed else "ALERT SUPPRESSED",
        f"severity={severity}",
        f"reason={reason or 'none'}",
        f"alertness={_n(alertness)}",
        f"fatigue={_n(fatigue)}",
        f"distraction={_n(distraction)}",
        f"context_urgency={('--' if context_urgency is None else f'{float(context_urgency):.2f}')}",
        f"head={head_direction}",
        f"eye={eye_state}",
        f"face_detected={face_detected}",
        f"audio={audio}",
        f"suppressed={suppressed}",
    ]
    if suppressed and suppression_reason:
        parts.append(f"suppression_reason={suppression_reason}")
    return " | ".join(parts)


def log_alert_decision(**fields) -> None:
    """Write one alert-decision line at INFO level (never raises)."""
    try:
        logger.info(format_alert_decision(**fields))
    except Exception:
        # Logging must never affect driving monitoring.
        pass
