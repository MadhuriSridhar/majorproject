# Neuro-Symbolic Driver Attention Monitoring System

## 1. Project Overview

This repository contains a real-time driver attention and alertness monitoring
prototype. It reads a laptop webcam with OpenCV, detects facial landmarks with
the MediaPipe Face Landmarker model, extracts eye and mouth measurements, and
uses deterministic, explainable rules to estimate fatigue and distraction.

The system combines:

- facial and visual cues such as eye closure, blinks, yawns, head turns, and
  downward gaze;
- fatigue and visual-distraction risk;
- contextual risk from time, driving duration, location/zone, weather, and
  (when CARLA is connected) simulated speed, location, weather, and
  day/night state;
- progressive alarm, voice, and dashboard notifications.

CARLA integration is implemented as a simulator context provider and as a
standalone route/weather demo. The normal dashboard attempts to use CARLA on
`localhost:2000` when it is available, but it can also run with webcam and
non-CARLA context.

This is a final-year engineering prototype, not a certified automotive safety
system. Dataset-based accuracy validation and real-vehicle deployment are not
currently complete.

## 2. Features

### Implemented

- OpenCV webcam capture and live Tkinter dashboard.
- MediaPipe Face Landmarker facial landmark detection.
- Eye Aspect Ratio (EAR) analysis.
- Blink detection, blink-rate tracking, and temporal eye-closure analysis.
- Prolonged eye-closure detection.
- Mouth Aspect Ratio (MAR) analysis and temporal yawning detection.
- Head-pose estimation and visual distraction classification, including head
  turns and sustained downward gaze. A downward gaze is a behaviour-based
  distraction signal; the system does not claim to detect a phone.
- Rule-based fatigue risk estimation and combined attention/alertness scoring.
- Context-aware risk fusion for fatigue, distraction, location, weather,
  simulated speed, time, and configured CARLA zones.
- Explainable risk factors, reasons, severity bands, risk trend, and session
  statistics.
- Alert escalation through `NORMAL`, `LOW_FATIGUE`, `ALERT`, `WARNING`, and
  `DROWSY`.
- Predefined WAV alarm tones and `pyttsx3` voice notification.
- Priority interruption, persistence, hysteresis, cooldowns, and recovery
  behaviour.
- Safe handling of no-face and multiple-face frames: analysis pauses instead
  of reusing stale facial measurements.
- Dashboard settings, persisted dashboard preferences, session summary, and
  local PDF export.
- Location context using Windows geolocation when available, IP fallback, and
  cache. Configurable geofences are supported; the checked-in
  `config/geofences.json` is empty.
- Weather context using Open-Meteo with throttling, cache, and failure-safe
  fallback.
- CARLA context integration:
  - vehicle lookup or generic vehicle spawn;
  - vehicle speed and speed-risk context;
  - CARLA world location/zone context;
  - CARLA weather context;
  - simulated day/night context from sun altitude;
  - smooth spectator-camera following;
  - official CARLA `GlobalRoutePlanner` plus `BasicAgent` route navigation;
  - five-preset weather/day-night demo.

### Not currently implemented

- Traffic simulation as a project feature.
- Mobile-phone or hand-object detection.
- Speech, passenger-talking, sunglasses, or eye-occlusion detection.
- Low-light enhancement or night vision.
- CAN, OBD-II, ignition, vehicle-motion sensors, or real-vehicle control.
- A machine-learning fatigue classifier beyond the MediaPipe landmark model.

## 3. System Architecture

```text
Webcam
  |
  v
Face Detection / MediaPipe Face Landmarks
  |
  v
Feature Extraction (EAR, MAR, blink, yawn, head pose)
  |
  v
Fatigue Analysis + Visual Distraction Analysis
  |                         ^
  |                         |
  +------------ CARLA context (zone, weather, day/night, speed)
  |                         |
  v                         |
Attention Engine <----------+
  |
  v
Alert Escalation Controller
  |
  v
Alarm / pyttsx3 Voice / Tkinter Dashboard
```

`FatigueEngine` owns fatigue scoring. `AttentionEngine` fuses fatigue,
distraction, and context. `AlertEscalationController` owns alert state and
timing. The UI displays those outputs rather than recalculating risk.

## 4. Technology Stack

The versions below are the constraints declared by `requirements.txt`; exact
installed patch versions may differ.

| Technology | Use | Declared version |
|---|---|---|
| Python | Runtime | Python 3.12.3 in the checked-in `.venv`; Python 3.12 is the verified runtime |
| OpenCV (`opencv-contrib-python`) | Camera frames and image processing | `>=4.10.0,<5.0.0` |
| MediaPipe | Face Landmarker Tasks API | `>=0.10.14,<2.0.0` |
| NumPy | Landmark and numeric feature calculations | Installed transitively/with OpenCV and MediaPipe; not pinned directly |
| Pillow | Tkinter camera-image rendering | `>=10.0.0` |
| Tkinter/`ttk` | Dashboard and settings UI | Python standard-library GUI |
| `pyttsx3` | Offline text-to-speech voice alerts | `>=2.90` |
| Windows `winsound` | Interruptible WAV alarm playback | Python/Windows standard library |
| ReportLab | Session-summary PDF export | `>=4.0.0` |
| CARLA Python API | CARLA client and simulator types | Required by `src/carla_context.py`; not pinned in `requirements.txt` |
| CARLA Python agents | `GlobalRoutePlanner`, `BasicAgent`, `RoadOption` | Supplied by the CARLA PythonAPI |
| Open-Meteo | Weather context HTTP endpoint | No Python package; standard-library `urllib` client |

CARLA navigation may also require the dependencies shipped with the CARLA
PythonAPI, including `networkx` and `Shapely`, as reported by the loader when
agent imports fail.

## 5. System Requirements

- Windows is the supported and verified operating system. The audio backend
  uses Windows `winsound`.
- Python 3.12 is verified in the repository virtual environment.
- A webcam accessible by OpenCV is required for the live dashboard.
- The MediaPipe model file `models/face_landmarker.task` is required.
- CARLA is required only for CARLA demos or CARLA context. The repository does
  not pin a CARLA release; use a CARLA installation whose PythonAPI matches
  the installed client package.
- A normal desktop CPU is sufficient for the rule-based processing, but no
  formal CPU, GPU, or RAM minimum has been established in this repository.
- Internet access is optional for the core webcam pipeline. It is used for
  IP geolocation fallback and Open-Meteo weather; cached or unavailable
  context is handled without stopping the camera loop.
- Disk space is needed for Python dependencies, the MediaPipe `.task` model,
  and optional generated reports/audio assets.

## 6. Project Structure

```text
majorproject/
├── src/
│   ├── app_ui.py              # Main Tkinter dashboard entry point
│   ├── carla_context.py       # CARLA client, context, route and weather demo
│   ├── facial_landmarks.py    # MediaPipe Face Landmarker wrapper
│   ├── eye_features.py        # EAR and eye feature extraction
│   ├── temporal_eye.py        # Blink and prolonged-closure tracking
│   ├── mouth_features.py      # MAR and mouth feature extraction
│   ├── temporal_mouth.py      # Yawn tracking
│   ├── head_pose.py           # Head-pose estimation
│   ├── distraction_monitor.py # Visual distraction state
│   ├── fatigue_engine.py      # Authoritative fatigue score
│   ├── attention_engine.py    # Fatigue/distraction/context fusion
│   ├── alert_escalation.py    # Alert state machine and cooldowns
│   ├── audio_engine.py        # WAV playback and voice alerts
│   ├── location_context.py    # GPS/IP/cache/geofence context
│   ├── weather_context.py     # Open-Meteo context
│   ├── session_manager.py     # Session metrics and events
│   └── session_summary.py     # Summary and PDF export
├── tests/                     # Deterministic unit tests
├── config/
│   ├── geofences.json         # Current configured geofences (empty)
│   ├── geofences.example.json # Geofence schema example
│   └── *cache.json            # Location/weather caches
├── models/face_landmarker.task # Required MediaPipe model
├── assets/sounds/             # Local alarm WAV assets
├── requirements.txt           # Python dependency constraints
├── settings.json              # Persisted audio/settings values
└── dashboard_prefs.json       # Persisted dashboard component selection
```

## 7. Environment Setup

From the repository root, create and activate the existing Windows
PowerShell virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the declared dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The current repository already includes `models/face_landmarker.task`. If that
file is absent in another checkout, download Google's Face Landmarker task
model from the URL in `src/facial_landmarks.py` and save it at exactly
`models/face_landmarker.task`.

CARLA is not installed by `requirements.txt`. Install CARLA separately and
make its Python client importable in the virtual environment. The project
automatically searches the standard `C:\CARLA\PythonAPI\carla` layout and
other local Python paths; manually setting `CARLA_ROOT` or `PYTHONPATH` is not
required for the normal setup.

## 8. CARLA Setup

1. Install CARLA separately and ensure the CARLA Python client is importable
   from the activated environment.
2. Start the CARLA server. If it is installed in the repository's supported
   standard location, the command is:

   ```powershell
   & "C:\CARLA\CarlaUE4.exe"
   ```

   If CARLA is installed elsewhere, start that installation's executable
   instead. The code connects to `localhost:2000` by default.
3. `src/carla_context.py` uses the world/map already loaded in the CARLA
   server. It does not load a specific map, so the map is whatever CARLA
   reports as current.
4. The context first reuses the first existing vehicle. If none exists, it
   selects the first available `vehicle.*` blueprint and tries the first map
   spawn point.
5. `--route` dynamically chooses a validated destination spawn point and uses
   the official CARLA `GlobalRoutePlanner` and `BasicAgent` to follow the
   route at a target speed of 35 km/h. It prints route progress and the next
   maneuver, and selects another route after reaching a destination.
6. The CARLA spectator is moved smoothly behind and above the vehicle
   (10 m back, 5 m high) rather than snapping directly each update.
7. The standalone demo stops the vehicle with brake control on exit. There is
   no separate vehicle-reset command or world-reset mechanism in the
   repository; restart the CARLA server or rerun the demo to obtain a fresh
   session.

CARLA PythonAPI agents are loaded directly first, then searched from the
standard CARLA installation and import paths. If agents are found but their
navigation dependencies cannot import, install the `networkx` and `Shapely`
dependencies supplied/recommended by that CARLA release.

## 9. Running the Project

Use separate PowerShell windows for CARLA and the Python application.

### A. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### B. Start CARLA

```powershell
& "C:\CARLA\CarlaUE4.exe"
```

The executable path is installation-specific; the repository does not contain
the CARLA server.

### C. Run the CARLA route demo

```powershell
python -m src.carla_context --route
```

### D. Run the CARLA weather/day-night demo

```powershell
python -m src.carla_context --route --weather-demo
```

The weather demo changes every 30 seconds and cycles
`ClearNoon → SoftRainNoon → HardRainNight → FoggyNight → ClearNight`.

### E. Run the full dashboard/UI

```powershell
python src/app_ui.py
```

With CARLA running, the dashboard attempts to connect automatically. Without
CARLA, its CARLA fields remain unavailable and the webcam/context pipeline can
continue.

Other repository demos include `python src\drowsiness_demo.py`,
`python src\webcam_test.py`, and the focused EAR/MAR/landmark demos.

## 10. CARLA Context

When CARLA is active, `CarlaContext.get_context()` supplies:

- vehicle position and a CARLA-simulated location/zone classification;
- vehicle speed, speed level, and speed risk (`LOW`, `MODERATE`, `HIGH`, or
  `VERY_HIGH`);
- CARLA weather condition/risk from cloudiness, precipitation, fog, and
  wetness;
- `DAY` or `NIGHT` from the CARLA weather sun altitude (night adds a bounded
  context risk);
- vehicle state and driving-session duration.

The currently checked-in `config/geofences.json` contains no configured real
geofences. Therefore the normal location provider reports no configured
high-risk zone unless a developer populates that file. CARLA's virtual
location classifier is separate from real GPS/IP location.

The `--weather-demo` cycle is elapsed-time based: each of five presets remains
active for 30 seconds, then the cycle repeats. This is a CARLA environment
demo, not traffic simulation. CARLA context amplifies existing driver risk;
it does not by itself prove fatigue.

## 11. Attention Engine

The implementation is deterministic and rule-based:

- `FatigueEngine` scores prolonged eye closure, yawn evidence, night context,
  and driving-duration risk. Blink behaviour is tracked and can contribute as
  supporting evidence; normal blinks do not independently mean drowsiness.
- `DistractionMonitor` evaluates head turns and sustained downward gaze.
- `AttentionEngine` fuses fatigue and distraction with fatigue as the dominant
  contributor (`0.90` fatigue weight and `0.38` distraction weight), then
  applies bounded location/weather/speed/time context boosts only when driver
  risk is already present.
- The combined score is smoothed asymmetrically: risk rises promptly and
  recovery is held/gradual to avoid flicker.
- Attention levels are `SAFE` (0–24), `MODERATE` (25–49), `HIGH` (50–74),
  and `CRITICAL` (75–100).
- Alert severity maps to `NORMAL`, `LOW_FATIGUE`/`ALERT`, `WARNING`, or
  `DROWSY`. Higher severities require short persistence; recovery requires
  persistent improvement.

The README does not treat these engineering thresholds as accuracy results.

## 12. Alerts

Severity order is:

```text
NORMAL → LOW_FATIGUE → ALERT → WARNING → DROWSY
```

The current alert path provides:

- `LOW_FATIGUE`: spoken caution;
- `ALERT`: soft beep followed by voice;
- `WARNING`: warning alarm followed by voice;
- `DROWSY`: strong alarm followed by voice.

Sound and voice do not overlap. A higher-priority alert interrupts a lower
priority one; a lower-priority alert cannot interrupt a higher one. Audio is
triggered on severity entry rather than every camera frame. Per-severity
cooldowns are 30 s, 25 s, 20 s, and 15 s respectively, with episode recovery
and global-gap protection. Alert display windows are 2 s, 3 s, 4 s, and 5 s.
Recovery resets eligibility for a later episode. Audio failures are reported
while visual monitoring continues.

## 13. Testing

The `tests/` directory contains deterministic unit tests for fatigue scoring,
attention/distraction fusion, alert levels and stability, audio priority,
context urgency, location/geofence transitions, weather, temporal eye/mouth
tracking, face-loss safety, sessions, summaries, dashboard preferences, and
startup behaviour.

Run the complete suite from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests -v
```

The tests do not require a webcam. Dataset-based evaluation has not been
completed; no accuracy, precision, recall, or F1 number is currently
established by this repository.

## 14. Troubleshooting

### CARLA connection problems

- Confirm the CARLA server is running and listening on `localhost:2000`.
- Start the Python command only after the server has finished loading a world.
- The dashboard can run without CARLA, but CARLA fields will show unavailable.
- `CARLA_HOST` and `CARLA_PORT` are optional advanced overrides; they are not
  required for the normal setup.

### CARLA PythonAPI or agent import errors

- Confirm `import carla` works in the activated `.venv`.
- Confirm the CARLA PythonAPI contains
  `agents/navigation/global_route_planner.py`,
  `basic_agent.py`, and `local_planner.py`.
- If the loader reports missing agent dependencies, install the matching
  CARLA PythonAPI dependencies such as `networkx` and `Shapely`.
- Do not install an unrelated CARLA client version without matching it to the
  server/PythonAPI.

### Vehicle or route problems

- The current world must provide map spawn points.
- Existing vehicles are reused; stop/close the previous demo if it owns the
  desired vehicle.
- A route is selected from available spawn points and can fail if the loaded
  map has no suitable finite route. Try a standard CARLA road map and rerun.
- Stop with `Ctrl+C`; the standalone demo applies the brake in its cleanup
  path.

### Dashboard, Tkinter, or camera problems

- Run from the repository root so `models/face_landmarker.task` and project
  modules resolve correctly.
- Verify that the webcam is connected and not exclusively open in another
  application. `python src\webcam_test.py` is the repository camera check.
- Verify that Python was installed with Tk support. The dashboard requires
  Tkinter and Pillow.
- If no face or multiple faces are visible, analysis intentionally pauses.
- If the model error says the `.task` file is missing, restore it at
  `models/face_landmarker.task`.

### Audio problems

- This implementation uses Windows `winsound` for WAV playback and
  `pyttsx3` for speech; Windows is therefore the supported audio environment.
- Check Windows output-device and speech settings.
- Disable voice or alarm sounds in Settings if a device/driver is unavailable;
  visual monitoring continues.
- Keep `assets/sounds/` available. Missing/generated tones are reported by the
  audio path rather than silently changing the detection result.

### Dependency or context problems

- Activate `.venv` before installing or running commands.
- Install exactly the declared dependencies with
  `python -m pip install -r requirements.txt`.
- Location and Open-Meteo network failures are expected to fall back to cache
  or unavailable context; they do not replace camera measurements with fake
  values.

## 15. Known Limitations

- This is a webcam/CARLA simulation prototype, not real-vehicle deployment.
- Dataset validation and formal accuracy measurement are pending.
- Traffic simulation is not currently included.
- A webcam can lose landmarks under poor lighting, occlusion, extreme head
  pose, camera movement, or multiple faces.
- No low-light enhancement, night vision, phone detection, speech detection,
  or driver identity model is included.
- CARLA uses the world already loaded by the server; the repository does not
  guarantee a particular map or CARLA release.
- Real geofences are empty by default and require verified configuration.
- Weather/location context depends on simulator/network/cache availability.
- Hardware minimums and production safety thresholds have not been formally
  established.

## 16. Future Work

The following are planned or suitable future extensions, not current features:

- traffic simulation and richer surrounding-vehicle context;
- a first-person/in-cabin CARLA view;
- additional labelled datasets and repeatable dataset evaluation;
- richer context modelling and real-world location validation;
- further alert/audio personalization and validation;
- real-world vehicle and hardware validation.

## 17. Development / Contribution Notes

Core changes belong in the relevant module under `src/`; tests for decision
logic and state transitions belong under `tests/`. Keep dependency changes in
`requirements.txt` and update this README when setup or public entry points
change. The project currently uses ordinary Git branch-based development; no
additional contribution workflow is defined in the repository.

Avoid committing generated reports, caches, virtual-environment files, or
machine-specific CARLA installations. Changes to scoring, escalation, or
context should include deterministic tests where practical.

## 18. License / Acknowledgements

No license file or explicit repository license is currently defined, so no
license is asserted here.

The project uses Google's MediaPipe Face Landmarker task model and the CARLA
simulator/PythonAPI. The model and simulator remain subject to their
respective upstream terms.
