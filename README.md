# Driver Drowsiness Prediction and Prevention System

## 1. Project Overview

A **real-time**, **laptop-webcam**, **computer-vision**, **rule-based** driver
drowsiness prediction and prevention prototype.

The system estimates driver fatigue risk from facial behaviour (eye closure,
blinks, yawns) plus contextual factors (night, driving duration), and warns the
driver with progressive audio and visual alerts before risk becomes severe.

> **Safety / honesty:** This is a final-year BE CSE prototype, **not** a
> certified automotive safety system. Thresholds and weights are engineering
> defaults tuned on a laptop webcam. There is **no** formal labelled-dataset
> accuracy study, and no scientific accuracy percentage should be claimed in a
> viva.

---

## 2. Objectives

The project implements:

- Real-time facial monitoring from a laptop webcam
- Eye aspect ratio (EAR) and eye open/closed classification
- Blink detection and blink-rate estimation
- Mouth aspect ratio (MAR) and yawn detection
- Prolonged eye-closure detection
- Night context (10:00 PM – 6:00 AM, local time)
- Driving-duration context (real threshold: 3 hours)
- Rule-based fatigue risk scoring (0–100)
- Progressive alert escalation (NORMAL → LOW FATIGUE → ALERT → WARNING → DROWSY)
- Session monitoring (metrics, risk history)
- Session summary with risk-factor breakdown
- Local PDF export of the session summary

---

## 3. Current Scope

The following are **implemented** in this repository:

- Laptop webcam capture (OpenCV)
- MediaPipe Face Landmarker (`.task` model)
- EAR (eye aspect ratio)
- Blink detection / blink count
- Blink rate
- Eye closure tracking / prolonged eye closure
- MAR (mouth aspect ratio)
- Yawn detection / yawn count
- Yawn frequency
- Night context
- Driving-duration context
- Rule-based fatigue engine
- Risk score (0–100)
- Temporal smoothing, persistence, hysteresis, recovery
- Progressive alert escalation (5 levels: NORMAL, LOW FATIGUE, ALERT, WARNING, DROWSY)
- pyttsx3 voice alerts
- Predefined local alarm sounds
- Alert priority interruption (higher severity cuts in; lower never blocks higher)
- Settings (volumes, voice, visual alerts, predefined sound selection)
- Dashboard customization (persisted locally)
- Risk trend / table breakdown
- Session statistics
- Session summary
- PDF export
- Automated tests (deterministic, no webcam required)

---

## 4. Explicitly Out of Scope (NOT implemented)

The following are **not** part of this project and must not be implied as
implemented:

- Machine learning classifier
- Firebase / cloud storage
- Driver login
- Sunglasses detection
- Eye-occlusion detection
- Low-light enhancement / night vision
- Head movement / head-pose detection
- Weather context
- IR camera
- GPS
- CAN
- OBD-II
- Vehicle motion detection
- Ignition detection
- Raspberry Pi
- External automotive hardware

---

## 5. System Architecture

```text
Laptop Webcam
     |
     v
OpenCV (VideoCapture)
     |
     v
MediaPipe Face Landmarker  <--- models/face_landmarker.task
     |
     v
EAR / MAR  (eye_features.py, mouth_features.py)
     |
     v
Temporal Eye / Mouth Analysis  (temporal_eye.py, temporal_mouth.py)
     |
     v
Context Analysis  (time_context.py, driving_duration.py)
     |
     v
Evidence Bridge  (evidence_bridge.py)
     |
     v
Fatigue Engine  (fatigue_engine.py)  <--- authoritative decision-maker
     |
     v
FatigueDecision (risk, breakdown, severity, reasons)
     |
     v
Alert Controller  (alert_escalation.py -> audio_engine.py)
     |
     v
UI / Audio / Session Summary
```

**Authoritative decision-maker:** `FatigueEngine` is the only component that
computes the risk score, factor breakdown, severity, and reasons. The UI, alert
system, and session summary only *display* those values — none of them
re-implement scoring.

**One authoritative alert path:** `AlertEscalationController` + `AudioEngine`
handle all alerts. Audio triggers once per severity state entry (never every
frame) and supports priority interruption.

---

## 6. Fatigue Features

### Eyes — EAR

- `eye_features.py` computes left/right/average **EAR** from six landmark
  points per eye.
- `temporal_eye.py` turns the per-frame EAR into temporal **blink** vs
  **prolonged closure** events.

### Mouth — MAR

- `mouth_features.py` computes **MAR** from lip landmarks.
- `temporal_mouth.py` turns the per-frame MAR into **yawn** events.

### Temporal behaviour (why a single frame is not enough)

- **One blink** is a short, re-opening eye closure → very low / no fatigue risk.
- **Brief mouth opening** is not a yawn — the mouth must stay open long enough.
- **Prolonged eye closure** is treated as a **strong** direct fatigue indicator.

| Parameter | Value |
|-----------|-------|
| EAR close threshold | 0.21 |
| EAR open (hysteresis) | 0.23 |
| Blink min / max | 0.10 / 0.80 s |
| Prolonged eye closure | ≥ 1.50 s |
| MAR open threshold | 0.60 |
| MAR close (hysteresis) | 0.52 |
| Yawn minimum duration | 1.20 s |

---

## 7. Fatigue Risk Engine

The system is **rule-based** — no machine learning.

- Risk score ranges **0–100** (clamped).
- Risk factors: **Eye**, **Blink**, **Yawn**, **Night**, **Driving Duration**.
- **Prolonged eye closure** is the strongest direct signal and can independently
  reach DROWSY.
- **Yawn** and **blink behaviour** are supporting signals.
- **Night** and **driving duration** are contextual signals.
- Context alone (night and/or duration) cannot automatically produce DROWSY.

| Weight / threshold | Value |
|--------------------|-------|
| Prolonged eye closure | +70 |
| Yawn event | +15 (bounded, max 15) |
| Night context | +5 |
| Driving duration risk | +10 (max) |
| LOW FATIGUE entry | 25 |
| ALERT entry | 35 |
| WARNING entry | 50 |
| DROWSY entry | 70 |
| Smoothing alpha | 0.35 |

Severity mapping (using the authoritative risk score 0–100):

| Range | Severity |
|-------|----------|
| 0–19 | NORMAL |
| 20–34 | LOW FATIGUE |
| 35–49 | ALERT |
| 50–69 | WARNING |
| 70–100 | DROWSY |

Persistence, hysteresis, smoothing, and recovery protect against flicker and
false alarms. A genuine prolonged eye closure (a strong direct signal) can reach
DROWSY on its own; night and driving duration alone never do.

---

## 8. Driving Duration

- **Real project threshold: 3 hours** (`DRIVING_DURATION_THRESHOLD_HOURS = 3.0`).
- The contribution is **gradual and contextual**: negligible before 3 hours,
  starts after 3 hours, higher after 4 hours, and max at 5+ hours.
- **Driving duration alone never triggers DROWSY.**

### Demo mode (testing only)

- `USE_DEMO_DRIVING_DURATION = True` uses a **30-second** shortcut threshold so
  the logic can be verified without waiting 3 hours.
- The UI shows **DEMO MODE** when this is active.
- The 30-second value is a **demonstration shortcut only**. Never present
  "30 seconds of driving" as the real fatigue threshold.

---

## 9. Night Context

- Active window: **10:00 PM – 6:00 AM** (`NIGHT_START = time(22,0,0)`,
  `NIGHT_END = time(6,0,0)`).
- Uses the computer's **local system time**.
- Night is a **contextual factor only**; night alone cannot trigger DROWSY.
- There is **no** low-light enhancement or night-vision processing — the camera
  operates normally.

---

## 10. Alert Escalation

Alert levels and priority:

```text
NORMAL
  ↓
LOW FATIGUE
  ↓
ALERT
  ↓
WARNING
  ↓
DROWSY
```

Priority: DROWSY > WARNING > ALERT > LOW FATIGUE > NORMAL.

| Level | Alarm | Duration | Voice message |
|-------|-------|----------|---------------|
| LOW FATIGUE | voice only (no alarm) | — | "Please stay alert." |
| ALERT | soft beep | ~3 s | "Please stay attentive." |
| WARNING | warning alarm | ~7 s | "Fatigue warning. Please stay alert." |
| DROWSY | strong alarm | ~10 s | "Driver fatigue detected. Please take a safe break." |

Alert rules:

- Audio triggers **on state entry** (once) and does **not** repeat every frame.
- **Sound finishes before voice** — they never overlap.
- **Higher severity interrupts lower severity**: e.g. ALERT audio is stopped
  when WARNING arrives; the ALERT voice is then *not* spoken.
- **Lower severity never interrupts higher severity**: e.g. WARNING cannot
  interrupt DROWSY.
- **Recovery resets alert eligibility** so a later episode can alert again.
- **Audio failures never crash** the app — detection and UI continue.

---

## 11. Audio Settings

The **Settings** screen controls playback:

- **Alert Volume** — controls the LOW FATIGUE / ALERT alarm and the voice volume.
- **Alarm Volume** — controls the WARNING and DROWSY alarm volume.
- **Voice alerts** — enable/disable pyttsx3 voice.
- **Visual alerts** — enable/disable on-screen alert overlays.
- **Alarm sounds** — enable/disable the alarm tones.
- **Predefined sound selection** — choose from bundled local alarm sounds.

Volume scaling is applied to the PCM samples at runtime; the original
`assets/sounds/*.wav` files are **never modified**. Only predefined local alarm
sounds are used — users cannot import custom music files.

---

## 12. No Face / Multiple Face Safety

- **No face detected:** the dashboard shows **FACE NOT DETECTED**, facial
  fatigue calculations pause, and no stale EAR/MAR/blink/yawn values are reused.
  No fatigue alert is generated from missing facial data.
- **Multiple faces:** the dashboard shows **MULTIPLE FACES DETECTED**, driver
  analysis pauses, and the system does **not** arbitrarily choose one face.
- When a single valid face returns, analysis resumes safely.

---

## 13. User Interface

The dashboard (`src/app_ui.py`) is a professional Tkinter application.

- **Start Driving screen:** camera preview + START DRIVING button.
- **Top bar:** Driver Drowsiness Monitor, current time, camera status, Settings.
- **Main dashboard:** live camera with facial landmarks, fatigue risk,
  driver metrics, driving session, STOP DRIVING.
- **Driver metrics:** blinks, blink rate, yawns, yawn frequency, eye status,
  eye closure, fatigue risk, status.
- **Risk:** TREND and TABLE risk-factor breakdown (Eye, Blink, Yawn, Night,
  Driving Duration).
- **Context:** current time, night context, driving duration.
- **Bottom bar:** FPS, camera/system state.
- **Session summary:** after STOP DRIVING (EXPORT PDF / NEW DRIVING SESSION /
  EXIT).

---

## 14. Settings

The Settings screen includes only supported options:

- Alert volume
- Alarm volume
- Voice alerts (pyttsx3)
- Visual alerts
- Alarm sounds (enable/disable)
- Predefined alarm selection (ALERT / WARNING / DROWSY)
- Dashboard customization

**Dashboard customization** lists only implemented components:

- Fatigue Risk
- Eye Monitoring
- Blink Count
- Blink Rate
- Yawn Count
- Yawn Frequency
- Eye Closure
- Driving Duration
- Current Time
- Night Context
- Camera FPS
- Camera Status

Preferences are persisted locally (JSON).

---

## 15. Session Summary

After **STOP DRIVING**, the session summary shows:

- Driving duration
- Average fatigue risk
- Maximum fatigue risk
- Total blinks / average blink rate
- Total yawns
- Low fatigue / warning / drowsy event counts
- Risk-factor breakdown (Eye, Blink, Yawn, Night, Driving Duration)
- Final status
- Night context

**PDF export** (ReportLab) writes a local report to `reports/` and includes the
session ID, date, start/end times, driving duration, risk statistics, event
counts, risk-factor breakdown, and a sampled risk trend.

---

## 16. Project Structure

```text
majorproj/
├── .gitignore
├── README.md
├── DEMO.md
├── requirements.txt
├── models/
│   └── face_landmarker.task        # MediaPipe model (required)
├── assets/
│   └── sounds/                     # generated predefined WAVs (git-ignored)
├── src/
│   ├── app_ui.py                   # Tkinter dashboard (main application entry)
│   ├── drowsiness_demo.py          # OpenCV webcam monitor (alert escalation)
│   ├── facial_landmarks.py         # MediaPipe Face Landmarker helpers
│   ├── eye_features.py             # EAR math
│   ├── temporal_eye.py             # Blink / prolonged closure tracker
│   ├── mouth_features.py           # MAR math
│   ├── temporal_mouth.py           # Yawn tracker
│   ├── time_context.py             # DAY / NIGHT context
│   ├── driving_duration.py         # session timer + duration risk
│   ├── evidence_bridge.py          # maps trackers/context -> FatigueEvidence
│   ├── fatigue_engine.py           # authoritative rule-based fatigue engine
│   ├── alert_escalation.py         # 5-level alert state machine
│   ├── audio_engine.py             # interruptible sound -> voice playback
│   ├── predefined_sounds.py        # local sound synthesis + durations
│   ├── session_manager.py          # session stats / history / counters
│   ├── session_summary.py          # summary text + PDF export
│   ├── settings_store.py           # settings + dashboard preferences
│   └── stage demos                 # webcam_test, landmark_demo, ear_demo,
│                                   # blink_demo, yawn_demo, context_demo
└── tests/
    ├── test_alert_escalation.py
    ├── test_alert_level.py         # 5-level alert system (ALERT severity)
    ├── test_audio_engine.py
    ├── test_audio_priority.py      # audio interruption / volume
    ├── test_dashboard_prefs.py
    ├── test_driving_duration.py
    ├── test_evidence_bridge.py
    ├── test_fatigue_engine.py
    ├── test_foundation_cleanup.py
    ├── test_foundation_engine.py
    ├── test_session_manager.py
    ├── test_session_summary.py
    ├── test_temporal_eye.py
    ├── test_temporal_mouth.py
    └── test_time_context.py
```

---

## 17. Installation

### Requirements

- Windows 10/11
- Python 3.12 recommended
- Working webcam (camera privacy permission enabled)
- Speakers/headphones for audio alerts
- MediaPipe `.task` model at `models/face_landmarker.task`

### Setup

```powershell
cd C:\Users\samud\Desktop\MAJOR_PROJECT\majorproj
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Confirm the model file exists:

```powershell
dir models\face_landmarker.task
```

If missing, download the official MediaPipe Face Landmarker `.task` model and
save it as `models/face_landmarker.task` (URL is documented in
`facial_landmarks.py`).

`requirements.txt`:

```text
opencv-contrib-python>=4.10.0,<5.0.0
mediapipe>=0.10.14,<2.0.0
pyttsx3>=2.90
reportlab>=4.0.0
pillow>=10.0.0
```

> MediaPipe needs `opencv-contrib-python`. Do **not** also install plain
> `opencv-python` in the same venv (they conflict).

---

## 18. How to Run

### Main dashboard application (entry point)

```powershell
.\.venv\Scripts\Activate.ps1
python src\app_ui.py
```

### Webcam monitor (OpenCV HUD)

```powershell
python src\drowsiness_demo.py
```

Press `q` in the OpenCV window to quit.

### Run the tests

```powershell
python -m unittest discover -s tests -v
```

---

## 19. Demo Procedure

A simple first-review demonstration:

1. Run `python src\app_ui.py`.
2. Click **START DRIVING**.
3. Show the normal state (green, low risk).
4. Blink normally → blink count increases, risk stays low.
5. Yawn → small temporary risk increase (bounded).
6. Repeated yawns / context → **LOW FATIGUE** (voice only, "Please stay alert.").
7. More evidence → **ALERT** (~3 s soft beep → "Please stay attentive.").
8. Stronger evidence → **WARNING** (~7 s warning alarm → voice).
9. Prolonged eye closure / severe combined fatigue → **DROWSY** (~10 s strong
   alarm → voice).
10. Open the eyes → **recovery**, audio stops, status recovers.
11. Click **STOP DRIVING** → session summary.
12. **EXPORT PDF** → local report in `reports/`.
13. **NEW DRIVING SESSION** or **EXIT**.

> During the demo, if `USE_DEMO_DRIVING_DURATION = True`, the 30-second driving
> duration is a **demo shortcut** and the UI shows **DEMO MODE**. The real
> project threshold is **3 hours**.

---

## 20. Testing

The test suite is deterministic and requires no webcam or speakers. It covers
temporal logic, fatigue scoring, alert escalation (including the ALERT level),
audio ordering, audio priority interruption, volume settings, session
management, summary/PDF, dashboard preferences, monotonic timing, yawn cap, and
no-face / multiple-face safety.

Run:

```powershell
python -m unittest discover -s tests -v
```

The suite passes with the current test modules (140 foundational + 19 audio
priority/volume + 16 alert-level tests = 175 tests).

---

## 21. Current Limitations

- Depends on a visible, roughly frontal face in a laptop webcam.
- Lighting-sensitive (no low-light enhancement).
- Thresholds and weights are engineering defaults, not a validated accuracy
  study.
- No formal labelled-dataset accuracy evaluation.
- Not automotive-certified.
- No actual vehicle-movement detection.

---

## 22. Viva Explanation (short script)

1. **Problem:** Drowsy driving is dangerous; an early-warning system using a
   common laptop camera helps.
2. **Solution:** Detect face landmarks → measure eye openness (EAR) and mouth
   openness (MAR) → track events over time → add night/long-driving context →
   score risk with a rule-based engine → warn with sound + voice.
3. **Why MediaPipe?** Reliable on-device facial landmarks without training a
   detector.
4. **Why EAR/MAR?** Simple geometric ratios that change as eyes/mouth open or
   close.
5. **Why temporal logic?** One closed frame is usually a blink; duration
   separates blinks from dangerous prolonged closure (same for yawns vs
   talking).
6. **Why a rule-based fatigue engine?** One authoritative, explainable place
   for fusion; nothing else re-implements scoring.
7. **Why context?** Night and long driving raise risk a little but cannot
   alone mark DROWSY.
8. **Why alarms escalate?** Progressive sound → voice, once per state entry,
   with priority interruption and recovery/cooldown protection.
9. **Prevention:** Sound + voice + visual alerts prompt the driver to take a
   break.
10. **Honesty:** Rule-based engineering prototype; formal dataset accuracy is
    future work.

---

## 23. Honesty / Safety Statement

This is a **prototype**. Thresholds and weights are engineering defaults tuned
on a laptop webcam. No scientific accuracy percentage should be claimed without
a formal evaluation. The system should **not** be treated as a certified safety
device.

---

## 24. Change Log

**Current update — final alert escalation pass:**

- **5-level alert system:** added the **ALERT** severity level between LOW
  FATIGUE and WARNING. Severity mapping: NORMAL (0–19), LOW FATIGUE (20–34),
  ALERT (35–49), WARNING (50–69), DROWSY (70–100).
- **LOW FATIGUE is voice only** (no loud alarm). ALERT uses a ~3 s soft beep,
  WARNING ~7 s, DROWSY ~10 s — each followed by its voice message.
- **Progressive, state-entry, cancellable alerts:** higher severity interrupts
  lower severity; lower never interrupts higher; no repeat while the same state
  persists; recovery resets alert eligibility.
- **Audio priority interruption fixed:** interruptible WAV playback
  (`winsound.SND_MEMORY | SND_ASYNC` with `SND_PURGE`). The interrupted alarm's
  voice is not spoken.
- **Audio volume controls now work:** `alert_volume` controls LOW FATIGUE /
  ALERT and voice; `alarm_volume` controls WARNING and DROWSY. Volume is applied
  to the PCM samples at runtime — the original `assets/sounds/*.wav` files are
  never modified.
- **Natural-completion fix:** an alert that finishes naturally releases its
  priority slot, so later alerts (e.g. after recovery) can play again.
- **README updated** to reflect the current implementation only.