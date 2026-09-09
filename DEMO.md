# Friday Demo & Manual Test Guide

This project is a **rule-based** driver drowsiness detection system using a laptop webcam,
MediaPipe Face Landmarker, OpenCV, and the Tkinter dashboard. This guide covers the
recommended settings and the manual test flow.

## Run the integrated monitor (webcam)

```powershell
.\.venv\Scripts\Activate.ps1
python src\drowsiness_demo.py
```

Press `q` in the OpenCV window to quit.

## Run the full dashboard application

```powershell
python src\app_ui.py
```

The dashboard is the main application: START DRIVING → live dashboard → STOP DRIVING →
SESSION SUMMARY → EXPORT PDF / NEW DRIVING SESSION / EXIT.

## Demo settings (recommended)

### Driving duration — keep DEMO MODE on

File: `src/driving_duration.py`

```python
USE_DEMO_DRIVING_DURATION = True
DEMO_DRIVING_DURATION_THRESHOLD_SECONDS = 30.0
DRIVING_DURATION_THRESHOLD_HOURS = 3.0   # real project threshold; do not reduce
```

- UI shows **DEMO MODE**.
- After ~30 s, Driving Risk becomes ELEVATED (small contextual increase).
- **Never** present the 30 s demo threshold as the real fatigue threshold. The real
  project threshold is **3 hours**; driving duration alone never triggers DROWSY.

### Night context — optional override

File: `src/time_context.py`

```python
DEMO_FORCE_LOCAL_TIME = None        # normal (use real local time)
# DEMO_FORCE_LOCAL_TIME = "23:15:32"  # force NIGHT without changing the clock
```

Night context is active **only** when a driving session is active AND local time is
**10:00 PM – 6:00 AM**. Night is a contextual factor only; night alone never triggers DROWSY.

## Verified parameter values (do not change casually)

### Eyes — `src/eye_features.py` + `src/temporal_eye.py`

| Parameter | Value |
|-----------|-------|
| EAR close threshold | 0.21 |
| EAR open threshold (hysteresis) | 0.23 |
| Blink duration window | 0.10–0.80 s |
| Prolonged eye closure | 1.50 s |

Includes hysteresis, temporal blink vs prolonged logic, and no single-frame drowsiness.

### Mouth — `src/mouth_features.py` + `src/temporal_mouth.py`

| Parameter | Value |
|-----------|-------|
| MAR open threshold | 0.60 |
| MAR close threshold (hysteresis) | 0.52 |
| Yawn minimum duration | 1.20 s |

Includes hysteresis, duration gate, and one yawn per continuous open event. Yawn risk is
**bounded** by its configured weight (15) so repeated yawns cannot overwhelm eye behaviour.

### Fatigue — `src/fatigue_engine.py` (rule-based, authoritative)

| Weight / threshold | Value |
|--------------------|-------|
| Prolonged eye closure | +70 |
| Yawn event | +15 (bounded, max 15) |
| Night context | +5 |
| Driving duration risk | +10 |
| LOW_FATIGUE entry | 25 |
| ALERT entry | 35 |
| WARNING entry | 50 |
| DROWSY entry | 70 |

Prolonged eye closure is the **dominant** direct fatigue signal and can independently reach
DROWSY. Persistence, recovery, and hysteresis protect against flicker. Only `FatigueEngine`
computes the risk score, factor breakdown, severity, and reasons.

## Alert escalation + audio (Phase 2)

Alerts are handled by `AlertEscalationController` + `AudioEngine`. Audio triggers **once**
on severity state entry (never every frame). Levels: NORMAL → LOW FATIGUE → ALERT →
WARNING → DROWSY.

| Severity | Sound | Duration | Voice |
|----------|-------|----------|-------|
| LOW FATIGUE | voice only (no alarm) | — | "Please stay alert." |
| ALERT | soft beep | ~3 s | "Please stay attentive." |
| WARNING | warning alarm | ~7 s | "Fatigue warning. Please stay alert." |
| DROWSY | strong alarm | ~10 s | "Driver fatigue detected. Please take a safe break." |

Sequence per level with an alarm: **sound finishes → then voice** (never simultaneous).
Priority: DROWSY > WARNING > ALERT > LOW FATIGUE > NORMAL. A higher severity interrupts a
lower one (the interrupted alarm's voice is not spoken); a lower severity never interrupts
a higher one. Recovery resets alert eligibility so a later episode can alert again.
Cooldowns gate repeat audio.

## Manual test flow (webcam)

1. `python src\drowsiness_demo.py`
2. **START** → sit normally, blink normally → NORMAL / LOW score.
3. One genuine yawn → small temporary increase (bounded), not sustained.
4. Repeated yawns / context → **LOW FATIGUE** → verify voice only ("Please stay alert.").
5. More evidence → **ALERT** → verify ~3 s soft beep then voice ("Please stay attentive.").
6. Stronger evidence → **WARNING** → verify ~7 s alarm then voice.
7. Prolonged eye closure / severe combined fatigue → **DROWSY** → verify ~10 s strong
   alarm then voice.
8. Open eyes → recovery → audio stops, status recovers.
9. **No face** (cover camera) → FACE NOT DETECTED, analysis pauses, no stale risk.
10. **Multiple faces** → MULTIPLE FACES, analysis pauses (no arbitrary face selection).
11. Press `q` to quit.

## Dashboard manual checks

In `src\app_ui.py`:

- START DRIVING → live dashboard (camera, fatigue risk, driver metrics, context).
- TREND / TABLE risk breakdown (Eye, Blink, Yawn, Night, Driving Duration).
- SETTINGS (alert/alarm volume, voice ON/OFF, visual alerts, predefined sounds).
- CUSTOMIZE DASHBOARD (only supported components, persisted locally).
- STOP DRIVING → SESSION SUMMARY (driving time, average/max risk, blink/yawn counts,
  event counts, final status, risk-factor breakdown).
- EXPORT PDF (local ReportLab PDF), NEW DRIVING SESSION, EXIT.

## Unit tests

```powershell
python -m unittest discover -s tests
```

All tests are deterministic and require no webcam.