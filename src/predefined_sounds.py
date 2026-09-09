"""
Predefined local alarm sounds (Phase 2).

Generates small WAV files locally using the standard-library `wave` module so
the application has bundled, offline alarm tones.

No internet downloads. No user-imported songs. If generation fails, the app
continues without sound (visual + voice fallback).

The original `playsound` package is incompatible with Python 3.12, so sound
playback is handled by `winsound.PlaySound` (Windows built-in WAV player) in
the audio engine. This module only *creates* the .wav files.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOUNDS_DIR = PROJECT_ROOT / "assets" / "sounds"

# ---------------------------------------------------------------------------
# Alert audio durations (explicit configuration values)
# ---------------------------------------------------------------------------
# Every alert severity uses an alarm/beep first, then a voice message.
# The sound must finish before the voice starts.

# Short alerts: sound finishes quickly, then voice. Avoid long looping beeps.
LOW_FATIGUE_ALARM_DURATION_SECONDS = 1.0
ALERT_ALARM_DURATION_SECONDS = 1.5
WARNING_ALARM_DURATION_SECONDS = 2.5
DROWSY_ALARM_DURATION_SECONDS = 3.5

# ---------------------------------------------------------------------------
# Predefined sound names (driver selectable in Settings)
# ---------------------------------------------------------------------------
# LOW FATIGUE uses the same soft-beep family as the alert sounds.

ALERT_SOUNDS = ("soft_beep_1", "soft_beep_2")
WARNING_SOUNDS = ("warning_1", "warning_2")
DROWSY_SOUNDS = ("strong_alarm_1", "strong_alarm_2")

ALL_SOUND_NAMES = (
    *ALERT_SOUNDS,
    *WARNING_SOUNDS,
    *DROWSY_SOUNDS,
)

# ---------------------------------------------------------------------------
# Tone synthesis helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 22050


def _write_wav(path: Path, samples: list[float], sample_rate: int = SAMPLE_RATE) -> None:
    """Write a mono 16-bit PCM WAV file from float samples in [-1, 1]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for value in samples:
            clamped = max(-1.0, min(1.0, value))
            frames += struct.pack("<h", int(clamped * 32767))
        wf.writeframes(bytes(frames))


def _tone(
    frequency_hz: float,
    duration_s: float,
    volume: float = 0.5,
    sample_rate: int = SAMPLE_RATE,
    attack_s: float = 0.01,
    release_s: float = 0.05,
) -> list[float]:
    """Generate a single sine tone with a short attack/release envelope."""
    n = int(sample_rate * duration_s)
    samples: list[float] = []
    for i in range(n):
        t = i / sample_rate
        env = 1.0
        attack_n = int(attack_s * sample_rate)
        release_n = int(release_s * sample_rate)
        if i < attack_n:
            env = i / max(1, attack_n)
        elif i >= n - release_n:
            env = max(0.0, (n - i) / max(1, release_n))
        value = volume * env * math.sin(2.0 * math.pi * frequency_hz * t)
        samples.append(value)
    return samples


def _beep_sequence(
    frequencies: list[float],
    note_s: float,
    gap_s: float,
    repeats: int,
    volume: float = 0.5,
) -> list[float]:
    """Concatenate several tones with silent gaps."""
    samples: list[float] = []
    gap = [0.0] * int(SAMPLE_RATE * gap_s)
    for _ in range(repeats):
        for freq in frequencies:
            samples.extend(_tone(freq, note_s, volume=volume))
            samples.extend(gap)
    return samples


# ---------------------------------------------------------------------------
# Sound definitions
# ---------------------------------------------------------------------------

SOUND_DEFINITIONS: dict[str, list[float]] = {}


def _build_sounds() -> None:
    """Populate SOUND_DEFINITIONS with synthesized tones.

    Default sounds are tuned to the configured durations:
    - ALERT ~2-3 s (soft beeps, reused)
    - WARNING ~5-7 s
    - DROWSY ~10 s
    """
    # Soft beeps (ALERT) - gentle, short tones (~2-3 s total).
    SOUND_DEFINITIONS["soft_beep_1"] = _beep_sequence(
        frequencies=[660.0], note_s=0.35, gap_s=0.15, repeats=4, volume=0.35
    )
    SOUND_DEFINITIONS["soft_beep_2"] = _beep_sequence(
        frequencies=[520.0, 660.0], note_s=0.30, gap_s=0.12, repeats=3, volume=0.35
    )

    # Warning alarms (WARNING) - a bit stronger, two-tone (~5 s total).
    # NOTE: each repeat plays every frequency in the list, so a 2-tone sound
    # lasts 2 * (note_s + gap_s) per repeat.
    SOUND_DEFINITIONS["warning_1"] = _beep_sequence(
        frequencies=[880.0, 660.0], note_s=0.30, gap_s=0.10, repeats=6, volume=0.5
    )
    SOUND_DEFINITIONS["warning_2"] = _beep_sequence(
        frequencies=[740.0], note_s=0.40, gap_s=0.12, repeats=9, volume=0.5
    )

    # Strong alarms (DROWSY) - loud, fast alternating high tones (~10 s total).
    # 2-tone sound: each repeat lasts 2 * (note_s + gap_s) = 0.56s;
    # 18 repeats ~= 10.1 s.
    SOUND_DEFINITIONS["strong_alarm_1"] = _beep_sequence(
        frequencies=[1200.0, 900.0], note_s=0.22, gap_s=0.06, repeats=18, volume=0.7
    )
    SOUND_DEFINITIONS["strong_alarm_2"] = _beep_sequence(
        frequencies=[1000.0, 1400.0, 1000.0], note_s=0.20, gap_s=0.05, repeats=16, volume=0.7
    )


_build_sounds()


def ensure_sounds_generated(sounds_dir: Path | None = None) -> Path:
    """
    Generate (if missing) all predefined .wav files.

    Returns the directory containing the wav files. Never raises for audio
    problems - failures are skipped and the audio engine handles missing files.
    """
    target = Path(sounds_dir) if sounds_dir else SOUNDS_DIR
    target.mkdir(parents=True, exist_ok=True)
    for name, samples in SOUND_DEFINITIONS.items():
        wav_path = target / f"{name}.wav"
        if not wav_path.is_file():
            try:
                _write_wav(wav_path, samples)
            except Exception:
                # Keep going; the audio engine handles missing files gracefully.
                continue
    return target


def sound_path(name: str, sounds_dir: Path | None = None) -> Path | None:
    """Return the path to a generated sound, or None if unavailable."""
    target = Path(sounds_dir) if sounds_dir else SOUNDS_DIR
    path = target / f"{name}.wav"
    return path if path.is_file() else None


def sound_duration_seconds(name: str) -> float:
    """Approximate duration of a generated sound (used for sequencing)."""
    samples = SOUND_DEFINITIONS.get(name, [])
    return len(samples) / SAMPLE_RATE


if __name__ == "__main__":
    directory = ensure_sounds_generated()
    print(f"Generated sounds in: {directory}")
    for name in ALL_SOUND_NAMES:
        print(f"  {name}: {sound_duration_seconds(name):.2f}s")