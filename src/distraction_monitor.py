"""
Distraction monitoring from existing face-landmark / head-pose signals.

Channels (temporal persistence / hysteresis):
- sustained head turning left / right
- looking down / off-road
- sustained passenger-side attention (longer lateral hold)

Mobile phone detection is intentionally NOT implemented: no phone/hand model
exists in this project (only face_landmarker.task). Fake phone heuristics are
forbidden.

Brief glances do not become serious distraction; longer sustained events raise
severity; returning to forward clears gradually via channel filters.
"""

from __future__ import annotations

from dataclasses import dataclass

from head_pose import HeadPoseObservation, HeadPoseTracker

# Duration gates (seconds) — brief movement ignored.
HEAD_TURN_MIN_S = 1.5
LOOK_DOWN_MIN_S = 1.8
PASSENGER_MIN_S = 3.5

# Extra hold after head-pose activates before a channel scores.
CHANNEL_ACTIVATE_S = 0.40
CHANNEL_CLEAR_S = 0.50


@dataclass(frozen=True)
class DistractionChannel:
    name: str
    active: bool
    duration_s: float
    score: float


@dataclass(frozen=True)
class DistractionObservation:
    """Aggregated distraction state for the attention layer."""

    head: HeadPoseObservation
    head_turn_left: DistractionChannel
    head_turn_right: DistractionChannel
    looking_down: DistractionChannel
    passenger_side: DistractionChannel
    distraction_risk: float  # 0..100
    primary: str  # none | head_turn_left | head_turn_right | looking_down | passenger_side
    active: bool
    phone_available: bool = False  # always False — no phone model in repo


class _DurationChannel:
    """Time-based activate/clear hysteresis for one distraction channel."""

    def __init__(self, name: str, activate_s: float, clear_s: float) -> None:
        self.name = name
        self.activate_s = float(activate_s)
        self.clear_s = float(clear_s)
        self.active = False
        self._raw_since: float | None = None
        self._clear_since: float | None = None
        self._active_since: float | None = None

    def reset(self) -> None:
        self.active = False
        self._raw_since = None
        self._clear_since = None
        self._active_since = None

    def update(self, raw: bool, now_s: float) -> tuple[bool, float]:
        if raw:
            self._clear_since = None
            if self._raw_since is None:
                self._raw_since = now_s
            if not self.active and (now_s - self._raw_since) >= self.activate_s:
                self.active = True
                self._active_since = now_s
        else:
            self._raw_since = None
            if self.active:
                if self._clear_since is None:
                    self._clear_since = now_s
                elif (now_s - self._clear_since) >= self.clear_s:
                    self.active = False
                    self._active_since = None
                    self._clear_since = None
            else:
                self._clear_since = None

        duration = 0.0
        if self.active and self._active_since is not None:
            duration = max(0.0, now_s - self._active_since)
        return self.active, duration


def _score_from_duration(
    active: bool,
    duration_s: float,
    min_s: float,
    base: float,
    ramp: float,
    cap: float,
) -> float:
    """Map sustained duration into 0..cap. Below min_s → low / monitor-only."""
    if not active:
        return 0.0
    if duration_s < min_s:
        # Active but still in the glance window — monitor only.
        return float(min(18.0, 4.0 + duration_s * 3.0))
    over = duration_s - min_s
    return float(min(cap, base + over * ramp))


class DistractionMonitor:
    """
    Distraction layer on top of HeadPoseTracker.

    Does not rewrite eye/mouth/fatigue detection. Phone channel omitted.
    """

    def __init__(
        self,
        head_tracker: HeadPoseTracker | None = None,
        channel_activate_s: float = CHANNEL_ACTIVATE_S,
        channel_clear_s: float = CHANNEL_CLEAR_S,
    ) -> None:
        self.head_tracker = head_tracker or HeadPoseTracker()
        self._left = _DurationChannel("head_turn_left", channel_activate_s, channel_clear_s)
        self._right = _DurationChannel("head_turn_right", channel_activate_s, channel_clear_s)
        self._down = _DurationChannel("looking_down", channel_activate_s, channel_clear_s)
        self._passenger = _DurationChannel("passenger_side", channel_activate_s, channel_clear_s)

    def reset(self) -> None:
        self.head_tracker.reset()
        self._left.reset()
        self._right.reset()
        self._down.reset()
        self._passenger.reset()

    def update(
        self, landmarks, frame_w: int, frame_h: int, now_s: float
    ) -> DistractionObservation:
        head = self.head_tracker.update(landmarks, frame_w, frame_h, now_s)

        left_raw = bool(head.active and head.direction == "left")
        right_raw = bool(head.active and head.direction == "right")
        down_raw = bool(head.active and head.direction == "down")
        # Passenger-side: sustained lateral attention (either side once long enough
        # via PASSENGER_MIN_S scoring; raw follows left/right head turn).
        passenger_raw = bool(
            head.active and head.direction in ("left", "right") and head.duration_s >= 1.0
        )

        left_on, left_dur = self._left.update(left_raw, now_s)
        right_on, right_dur = self._right.update(right_raw, now_s)
        down_on, down_dur = self._down.update(down_raw, now_s)
        pass_on, pass_dur = self._passenger.update(passenger_raw, now_s)

        # Prefer head-pose continuous duration when channel is active (more stable).
        if left_on and head.direction == "left":
            left_dur = max(left_dur, head.duration_s)
        if right_on and head.direction == "right":
            right_dur = max(right_dur, head.duration_s)
        if down_on and head.direction == "down":
            down_dur = max(down_dur, head.duration_s)
        if pass_on and head.direction in ("left", "right"):
            pass_dur = max(pass_dur, head.duration_s)

        left_ch = DistractionChannel(
            "head_turn_left",
            left_on,
            left_dur,
            _score_from_duration(left_on, left_dur, HEAD_TURN_MIN_S, 28.0, 10.0, 85.0),
        )
        right_ch = DistractionChannel(
            "head_turn_right",
            right_on,
            right_dur,
            _score_from_duration(right_on, right_dur, HEAD_TURN_MIN_S, 28.0, 10.0, 85.0),
        )
        down_ch = DistractionChannel(
            "looking_down",
            down_on,
            down_dur,
            _score_from_duration(down_on, down_dur, LOOK_DOWN_MIN_S, 32.0, 12.0, 90.0),
        )
        pass_ch = DistractionChannel(
            "passenger_side",
            pass_on,
            pass_dur,
            _score_from_duration(pass_on, pass_dur, PASSENGER_MIN_S, 35.0, 11.0, 92.0),
        )

        ranked = [
            (left_ch.name, left_ch.score),
            (right_ch.name, right_ch.score),
            (down_ch.name, down_ch.score),
            (pass_ch.name, pass_ch.score),
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        distraction_risk = float(min(100.0, ranked[0][1]))
        primary = ranked[0][0] if ranked[0][1] >= 20.0 else "none"
        active = distraction_risk >= 15.0 and any(
            c.active for c in (left_ch, right_ch, down_ch, pass_ch)
        )

        return DistractionObservation(
            head=head,
            head_turn_left=left_ch,
            head_turn_right=right_ch,
            looking_down=down_ch,
            passenger_side=pass_ch,
            distraction_risk=distraction_risk,
            primary=primary,
            active=active,
            phone_available=False,
        )
