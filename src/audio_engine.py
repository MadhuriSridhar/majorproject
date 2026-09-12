"""
Phase 2: Non-blocking, interruptible audio engine (predefined sound -> voice).

Plays a predefined WAV alarm, waits for it to finish, then speaks the voice
message using pyttsx3. Sound and voice NEVER overlap. All playback runs in a
background thread so the webcam / MediaPipe / fatigue engine / UI stay
responsive.

Key behaviours:
- PRIORITY INTERRUPTION: a higher-priority alert (DROWSY > WARNING > ALERT >
  LOW FATIGUE) stops the currently playing lower-priority alert so it can play.
  A lower-priority alert never interrupts a higher-priority one.
- INTERRUPTIBLE PLAYBACK: WAV playback uses a temporary file with
  `SND_FILENAME | SND_ASYNC` so playback returns immediately and can be stopped
  with `SND_PURGE` from another thread. A polling loop checks the `interrupted`
  event every ~30ms so a higher-priority alert can cut in promptly. The sound
  is length-limited to the configured duration (looping a short sound safely,
  stopping a long one early). The original asset file is never modified.
- LOW FATIGUE includes a 2-second alarm before the spoken warning.
- VOLUME: `alert_volume` controls the LOW_FATIGUE / ALERT alarm and voice
  volume; `alarm_volume` controls the WARNING / DROWSY alarms and their voice.
  Changing volume applies to subsequent alerts. Volume scaling is applied to
  PCM samples at runtime - the original asset WAV files are never modified.
- DURATION: alarm length is bounded to the configured duration
  (LOW_FATIGUE ~1s, ALERT ~1.5s, WARNING ~2.5s, DROWSY ~3.5s), safely looping
  a short sound when needed.

The original `playsound` package is incompatible with Python 3.12, so WAV
playback uses `winsound.PlaySound` (Windows built-in). If audio fails, the
app continues with visual alerts and logs the failure.
"""

from __future__ import annotations

import array
import struct
import threading
import time
import traceback
import tempfile
import os
from dataclasses import dataclass
from pathlib import Path

from predefined_sounds import (
    ALERT_ALARM_DURATION_SECONDS,
    DROWSY_ALARM_DURATION_SECONDS,
    LOW_FATIGUE_ALARM_DURATION_SECONDS,
    WARNING_ALARM_DURATION_SECONDS,
    ensure_sounds_generated,
    sound_path,
)

# ---------------------------------------------------------------------------
# Voice messages (exact per spec)
# ---------------------------------------------------------------------------

VOICE_MESSAGES: dict[str, str] = {
    "LOW_FATIGUE": "Please stay alert.",
    "ALERT": "Your alertness appears to be decreasing. Please stay attentive.",
    "WARNING": "Fatigue warning. Please stay alert and consider taking a break soon.",
    "DROWSY": "Driver fatigue detected. Please take a safe break now.",
}


def build_voice_message(severity: str, reason: str | None = None, contributing_factors: tuple[str, ...] | list[str] | None = None) -> str:
    """Return a severity-appropriate message based on the actual reason."""
    if severity not in VOICE_MESSAGES:
        return VOICE_MESSAGES.get("ALERT", "Please stay alert.")
    if not reason:
        return VOICE_MESSAGES[severity]

    details = (reason or "").strip()
    low = details.lower()
    if severity == "DROWSY":
        if "closure" in low or "eye" in low:
            return "Critical warning. Prolonged eye closure is reducing your alertness. Please take a safe stop now."
        if "road" in low or "head" in low or "gaze" in low or "away" in low:
            return "Critical warning. You have been looking away from the road for too long. Please focus on the road and take a safe stop."
        if "weather" in low:
            return "Critical warning. Attention risk is elevated because of poor weather conditions. Please drive carefully and take a safe stop."
        if "location" in low or "zone" in low:
            return "Critical warning. Road conditions ahead are increasing your risk. Please slow down and stay focused."
        if "yawn" in low:
            return "Critical warning. Your alertness is dangerously low due to repeated yawning. Please take a safe stop now."
        return "Critical warning. Your alertness is dangerously low due to the current driving conditions. Please take a safe stop now."
    if severity == "WARNING":
        if "closure" in low or "eye" in low:
            return "Warning. Prolonged eye closure is reducing your alertness. Please stay focused and consider taking a safe break."
        if "road" in low or "head" in low or "gaze" in low or "away" in low:
            return "Warning. You have been looking away from the road for too long. Please focus on the road."
        if "yawn" in low:
            return "Warning. Your alertness is decreasing due to repeated yawning. Please stay attentive."
        if "weather" in low:
            return "Warning. Attention risk is elevated because of poor weather conditions. Please drive carefully."
        return "Warning. Your attention is degrading. Please stay focused and consider a short break."
    if severity == "ALERT":
        if "closure" in low or "eye" in low:
            return "Attention warning. Prolonged eye closure is reducing your alertness. Please stay focused."
        if "road" in low or "head" in low or "gaze" in low or "away" in low:
            return "Attention warning. Your attention is drifting away from the road. Please look forward."
        if "yawn" in low:
            return "Attention warning. Repeated yawning is reducing your alertness. Please stay attentive."
        if "weather" in low:
            return "Attention warning. Poor weather is increasing your attention risk. Please drive carefully."
        return "Your alertness appears to be decreasing. Please stay attentive."
    if severity == "LOW_FATIGUE":
        if "road" in low or "away" in low:
            return "Please stay focused on the road."
        if "eye" in low or "closure" in low:
            return "Please keep your eyes on the road and stay alert."
        if "weather" in low:
            return "Weather conditions are adding risk. Please stay alert and drive carefully."
        return "Please stay alert."
    return VOICE_MESSAGES[severity]

# Default sound selection per severity (driver can change in Settings).
DEFAULT_SOUNDS: dict[str, str] = {
    "LOW_FATIGUE": "soft_beep_2",
    "ALERT": "soft_beep_2",
    "WARNING": "warning_1",
    "DROWSY": "strong_alarm_1",
}

# Severity priority (higher wins).
_SEVERITY_PRIORITY = {"NORMAL": 0, "LOW_FATIGUE": 1, "ALERT": 2, "WARNING": 3, "DROWSY": 4}


@dataclass(frozen=True)
class AudioSettings:
    """Snapshot of current audio settings used by the engine.

    ``alert_volume`` and ``alarm_volume`` are intentionally independent:
    - ``alert_volume`` controls LOW_FATIGUE / ALERT sound and voice.
    - ``alarm_volume`` controls WARNING / DROWSY alarm sound and voice.
    """

    alert_volume: float = 0.6  # 0.0 .. 1.0 -> LOW_FATIGUE / ALERT sound + voice
    alarm_volume: float = 0.7  # 0.0 .. 1.0 -> WARNING / DROWSY alarm + voice
    voice_enabled: bool = True
    sound_enabled: bool = True
    alert_sound: str = "soft_beep_2"
    warning_sound: str = "warning_1"
    drowsy_sound: str = "strong_alarm_1"


# ---------------------------------------------------------------------------
# Pure WAV helpers (testable without speakers)
# ---------------------------------------------------------------------------


def _wav_duration(data: bytes) -> float:
    """Return the duration (seconds) of a PCM WAV from its header, or 0.0."""
    if len(data) < 44:
        return 0.0
    try:
        sample_rate = struct.unpack_from("<I", data, 24)[0]
        bits_per_sample = struct.unpack_from("<H", data, 34)[0]
        channels = struct.unpack_from("<H", data, 22)[0]
        data_size = struct.unpack_from("<I", data, 40)[0]
    except Exception:
        return 0.0
    bytes_per_sec = sample_rate * channels * (bits_per_sample // 8)
    if bytes_per_sec <= 0:
        return 0.0
    return data_size / bytes_per_sec


def _scale_wav(data: bytes, volume: float) -> bytes:
    """
    Scale 16-bit PCM WAV amplitude by volume (0..1).

    Returns a copy of the WAV with scaled samples. The original WAV file is
    never modified. Non-16-bit or unreadable data is returned unchanged.
    """
    if len(data) < 44:
        return data
    try:
        bits_per_sample = struct.unpack_from("<H", data, 34)[0]
        data_size = struct.unpack_from("<I", data, 40)[0]
    except Exception:
        return data

    volume = max(0.0, min(1.0, volume))
    if bits_per_sample != 16 or data_size <= 0 or volume >= 0.999:
        return data

    pcm_start = 44
    pcm_end = min(len(data), pcm_start + data_size)
    pcm = data[pcm_start:pcm_end]
    if len(pcm) < 2:
        return data

    try:
        samples = array.array("h")
        samples.frombytes(pcm)
        for i in range(len(samples)):
            samples[i] = int(round(samples[i] * volume))
        out = bytearray(data[:pcm_start])
        out += samples.tobytes()
        return bytes(out)
    except Exception:
        return data


class AudioBackend(threading.Thread):
    """A single playback job: play WAV (interruptibly), then speak.

    The job is interruptible: if `interrupted` is set while the alarm is
    playing, playback is stopped immediately and the voice message is NOT
    spoken (a higher-priority alert has taken over).
    """

    def __init__(
        self,
        severity: str,
        wav_path: Path | None,
        settings: AudioSettings,
        on_error: callable | None = None,
        on_done: callable | None = None,
        reason: str | None = None,
        contributing_factors: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        super().__init__(name=f"audio-{severity}", daemon=True)
        self.severity = severity
        self.wav_path = wav_path
        self.settings = settings
        self.on_error = on_error
        self.on_done = on_done
        self.reason = reason or ""
        self.contributing_factors = tuple(contributing_factors or ())
        self.interrupted = threading.Event()
        self._sound_interrupted = False
        self._voice_interrupted = False

    # ------------------------------------------------------------------
    # Volume / duration selection
    # ------------------------------------------------------------------

    def _volume_for(self, severity: str) -> float:
        """Alert volume for LOW_FATIGUE / ALERT; alarm volume for WARNING/DROWSY."""
        if severity in ("LOW_FATIGUE", "ALERT"):
            return max(0.0, min(1.0, float(self.settings.alert_volume)))
        return max(0.0, min(1.0, float(self.settings.alarm_volume)))

    def _required_duration(self) -> float:
        """Configured alarm duration (seconds) for this severity."""
        if self.severity == "LOW_FATIGUE":
            return float(LOW_FATIGUE_ALARM_DURATION_SECONDS)
        if self.severity == "ALERT":
            return float(ALERT_ALARM_DURATION_SECONDS)
        if self.severity == "WARNING":
            return float(WARNING_ALARM_DURATION_SECONDS)
        if self.severity == "DROWSY":
            return float(DROWSY_ALARM_DURATION_SECONDS)
        return 0.0

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def _play_wav(self, path: Path) -> None:
        """
        Play a WAV interruptibly at the configured volume.

        Uses `winsound.PlaySound` with `SND_FILENAME | SND_ASYNC` so playback
        returns immediately and can be stopped with `SND_PURGE` from another
        thread. A polling loop checks the `interrupted` event every ~30ms so a
        higher-priority alert can cut in promptly. The sound is length-limited
        to the configured duration (looping a short sound safely, stopping a
        long one early). The original asset file is never modified.

        The scaled WAV data is written to a temporary file because
        `winsound.SND_MEMORY | winsound.SND_ASYNC` is not supported on Windows
        (causes "Cannot play asynchronously from memory").
        """
        import winsound

        data = path.read_bytes()
        duration = _wav_duration(data)
        if duration <= 0.0:
            duration = 1.0
        volume = self._volume_for(self.severity)
        scaled = _scale_wav(data, volume)
        budget = self._required_duration()

        # Write scaled WAV to a temporary file for reliable Windows playback.
        # winsound.SND_MEMORY | winsound.SND_ASYNC is not supported on Windows;
        # use SND_FILENAME with a real .wav file instead.
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            with os.fdopen(fd, "wb") as tmp_file:
                tmp_file.write(scaled)
            playback_path = tmp_path
        except Exception:
            # Fall back to original path if temp file creation fails.
            playback_path = path

        start = time.monotonic()
        while True:
            if self.interrupted.is_set():
                try:
                    winsound.PlaySound(None, winsound.SND_PURGE)
                except Exception:
                    pass
                self._sound_interrupted = True
                return

            # Play the sound asynchronously using the file path.
            # SND_FILENAME | SND_ASYNC is reliably supported on Windows.
            try:
                winsound.PlaySound(str(playback_path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            except Exception:
                pass

            elapsed = time.monotonic() - start
            segment = (
                min(duration, budget - elapsed)
                if budget > 0.0 and budget > elapsed
                else duration
            )
            if segment <= 0.0:
                break

            seg_end = time.monotonic() + segment
            while time.monotonic() < seg_end:
                if self.interrupted.is_set():
                    try:
                        winsound.PlaySound(None, winsound.SND_PURGE)
                    except Exception:
                        pass
                    self._sound_interrupted = True
                    return
                time.sleep(0.03)

            # Loop if the configured duration is longer than the sound file.
            if budget > 0.0 and time.monotonic() - start >= budget:
                break

        # Clean up temporary file if one was created.
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def _speak(self, message: str) -> None:
        """Speak the voice message in a background worker thread.

        This uses a single pyttsx3 engine per audio sequence and blocks only in
        the audio thread, not the Tkinter UI thread. If the engine fails, the
        exception is logged visibly and the application continues.
        """
        if not self.settings.voice_enabled or self.interrupted.is_set():
            self._voice_interrupted = True
            return
        try:
            import pyttsx3
        except Exception as exc:
            print(f"AUDIO ERROR: pyttsx3 import failed for {self.severity}: {exc}")
            self._voice_interrupted = True
            return

        engine = None
        try:
            print(f"AUDIO: {self.severity}")
            print(f"Voice: ENABLED")
            engine = pyttsx3.init()
            volume = self._volume_for(self.severity)
            try:
                engine.setProperty("volume", volume)
            except Exception:
                pass

            if self.interrupted.is_set():
                self._voice_interrupted = True
                try:
                    engine.stop()
                except Exception:
                    pass
                return

            print("VOICE STARTED")
            engine.say(message)
            engine.runAndWait()
            if self.interrupted.is_set():
                self._voice_interrupted = True
                try:
                    engine.stop()
                except Exception:
                    pass
                return
            print("VOICE FINISHED")
        except Exception as exc:
            print(f"AUDIO ERROR: {self.severity} voice failed: {exc}")
            self._voice_interrupted = True
        finally:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass

    def run(self) -> None:
        """
        Run the playback job: sound -> voice.

        If the sound is interrupted (by a higher-priority alert), the voice is
        NOT spoken. Audio errors never crash the app - they are logged and the
        job ends safely.
        """
        self._sound_interrupted = False
        self._voice_interrupted = False
        played = False
        try:
            print(f"AUDIO: {self.severity}")
            print(f"Sound: {self.wav_path.name if self.wav_path is not None else 'NONE'}")
            print(f"Duration: {self._required_duration():.1f}s")
            print(f"Voice: {'ENABLED' if self.settings.voice_enabled else 'DISABLED'}")

            if (
                self.settings.sound_enabled
                and self.wav_path is not None
                and self._required_duration() > 0.0
            ):
                print("AUDIO PLAYBACK STARTED")
                self._play_wav(self.wav_path)
                print("ALARM FINISHED")
                played = True

            if self.interrupted.is_set() or self._sound_interrupted:
                return

            message = build_voice_message(
                self.severity,
                self.reason,
                self.contributing_factors,
            )
            if message and self.settings.voice_enabled:
                if not self.interrupted.is_set():
                    print("STARTING VOICE")
                    self._speak(message)
                    played = True
        except Exception as exc:
            print(f"AUDIO ERROR: {self.severity}: {exc}")
            if self.on_error is not None:
                try:
                    self.on_error(f"Audio failed for {self.severity}: {exc}")
                except Exception:
                    traceback.print_exc()
            else:
                traceback.print_exc()
        finally:
            if played and self.on_done is not None:
                try:
                    self.on_done(self.severity)
                except Exception:
                    pass


class AudioEngine:
    """
    Manages audio playback jobs, ensuring only one sequence plays at a time
    and higher-priority alerts preempt lower ones.
    """

    PRIORITY = {"NORMAL": 0, "LOW_FATIGUE": 1, "ALERT": 2, "WARNING": 3, "DROWSY": 4}

    def __init__(self, settings: AudioSettings | None = None) -> None:
        self.settings = settings or AudioSettings()
        self._lock = threading.Lock()
        self._current: AudioBackend | None = None
        self._active_severity: str | None = None
        self._last_error: str | None = None
        # Generate sounds on startup (best effort).
        try:
            ensure_sounds_generated()
        except Exception:
            self._last_error = "Sound generation failed."

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def active_severity(self) -> str | None:
        """The severity whose audio is active (requested and not yet done)."""
        return self._active_severity

    def _sound_for(self, severity: str) -> str:
        if severity == "LOW_FATIGUE":
            return self.settings.alert_sound or "soft_beep_2"
        if severity == "ALERT":
            return self.settings.alert_sound
        if severity == "WARNING":
            return self.settings.warning_sound
        if severity == "DROWSY":
            return self.settings.drowsy_sound
        return ""

    def _on_job_done(self, severity: str) -> None:
        """A playback job finished naturally - release its priority slot."""
        with self._lock:
            if self._active_severity == severity:
                self._active_severity = None

    def play_alert(
        self,
        severity: str,
        reason: str | None = None,
        contributing_factors: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        """
        Start the sound -> voice sequence for the given severity.

        Priority: a lower-priority alert never interrupts a higher-priority
        one. A higher-priority (or equal) alert interrupts the current job so
        it can play immediately. Audio is never re-triggered every frame - the
        alert escalation controller decides when to call this (on state entry).
        """
        if severity not in self.PRIORITY or severity == "NORMAL":
            return
        with self._lock:
            # Priority guard: a lower-priority alert must never preempt a
            # higher-priority one that is still active.
            if self._active_severity is not None:
                current_priority = self.PRIORITY.get(self._active_severity, 0)
                new_priority = self.PRIORITY[severity]
                if new_priority < current_priority:
                    return

            current = self._current
            # Same severity already playing → do not restart / overlap.
            if (
                current is not None
                and current.is_alive()
                and self._active_severity == severity
            ):
                return

            if current is not None and current.is_alive():
                # Stop the current alarm so the new
                # higher-priority alert takes over immediately, and prevent its
                # voice from being spoken.
                current.interrupted.set()

            self._active_severity = severity
            wav = sound_path(self._sound_for(severity))
            job = AudioBackend(
                severity=severity,
                wav_path=wav,
                settings=self.settings,
                on_error=self._record_error,
                on_done=self._on_job_done,
                reason=reason,
                contributing_factors=contributing_factors,
            )
            self._current = job
            job.start()

    def _record_error(self, message: str) -> None:
        self._last_error = message
        print(f"AudioEngine: {message}")

    def interrupt_all(self) -> None:
        """Stop any active audio (used on session stop / shutdown)."""
        with self._lock:
            current = self._current
            self._current = None
            self._active_severity = None
        if current is not None:
            current.interrupted.set()

    def shutdown(self) -> None:
        self.interrupt_all()