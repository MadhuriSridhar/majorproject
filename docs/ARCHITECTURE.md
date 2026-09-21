# Architecture (as actually implemented)

This document describes the **runtime** architecture of the project as it
exists in the code — not aspirational/future behaviour. CARLA simulation is
**not** implemented yet; it is described only as a future input mode.

## Pipeline overview

```
Driver Camera
      ▼
MediaPipe Face Landmarker (perception / neural model)
      ▼
Feature extraction (EAR, blink, MAR, yawn, head pose)
      ▼
Fatigue features        Distraction features
      ▼                       ▼
FatigueEngine           DistractionMonitor          (symbolic / rule-based)
      └───────────┬───────────┘
                  ▼
        AttentionEngine  (symbolic fusion)
        + location risk + weather risk         (contextual inputs)
                  ▼
          Alertness score (0–100)
                  ▼
        AlertEscalationController  (persistence / hysteresis / cooldowns)
                  ▼
        AudioEngine (alarms / TTS)  +  Tkinter UI
```

## The two layers

- **Perception (neural):** MediaPipe Face Landmarker produces facial landmarks.
  No other ML model is bundled.
- **Symbolic / rule-based reasoning:** `fatigue_engine.py`,
  `distraction_monitor.py`, `attention_engine.py`, `alert_escalation.py`
  consume extracted features and produce a deterministic, explainable score
  and escalation. These files own the decision-making; nothing else
  re-implements scoring.

## Context inputs

Three environmental/contextual inputs feed the Attention Engine:

| Input | Source | Where it enters |
|-------|--------|-----------------|
| Location / zone | `location_context.py` (Windows GPS → IP geolocation → cache → Unavailable) | `AttentionEngine.update(..., location_risk, location_zone)` |
| Weather | `weather_context.py` (Open-Meteo → cache → Unavailable) | `AttentionEngine.update(..., weather_risk, weather_condition)` |
| Night context | `time_context.py` (local system time, 22:00–06:00) | `build_fatigue_evidence()` → `FatigueEngine` (night_risk) |
| Driving duration | `driving_duration.py` / `session_manager.py` | `build_fatigue_evidence()` → `FatigueEngine` (duration_risk) |

## Runtime flow (camera worker)

1. `DrowsinessApp._start_driving()` → `session_manager.start()` +
   `camera_worker.reset_session_state()`.
2. `CameraWorker.start_camera()` starts the camera loop **and** a separate
   `_context_worker` background thread.
3. `_context_worker` (1 s cadence) updates `LocationRiskProvider` (GPS →
   IP → cache → Unavailable) and `WeatherContextProvider` (Open-Meteo →
   cache → Unavailable), publishing the latest snapshot under a lock. When a
   location is available its coordinates are fed to the weather provider.
4. `_camera_loop` (real-time) reads the latest context snapshot, computes
   `loc_risk` / `wx_risk`, and calls `AttentionEngine.update(...)` with
   `location_risk`, `weather_risk`, `location_zone`, `weather_condition`.
5. The camera loop also feeds a `ZoneTransitionTracker` so zone entry/exit is
   an **event** (not a per-frame message), recorded into the session via
   `SessionManager.record_zone_transition()`.
6. The UI reads the latest state from the queue; it never computes risk.

## Location / geofence behaviour

- `LocationRiskProvider` tries, in order: Windows GPS (winrt), IP geolocation
  (`ip-api.com` → `ipapi.co`), a local cache, and finally `Unavailable`.
- A resolved coordinate is matched against `config/geofences.json` using
  haversine distance (`classify_geofence`). The first matching zone wins.
- If no zone is configured / matched, the system reports **"No configured
  high-risk zone"** with **"Location risk: Baseline"** — it does **not**
  claim the road is verified safe.
- `config/geofences.json` is empty by default. Populate it with
  `tools/build_geofences.py` (OpenStreetMap / Overpass), which copies only
  already-mapped features for a bounding box you provide. See README.

## Weather behaviour

- `WeatherContextProvider` calls Open-Meteo with the current coordinates; it
  maps WMO weather codes to a condition, computes a risk score from condition
  + precipitation + visibility + wind, and predicts "rain expected soon" only
  from hourly precipitation data (never from clouds alone).
- On any failure it reports **Unavailable** and (when fresh) falls back to a
  cached reading. No fake values are ever invented.

## Attention fusion

- `fuse_raw_risk` = `0.90 * fatigue + 0.38 * distraction` (distraction weighted
  by type + duration; a small bump when both are ≥ 40).
- `apply_context_boost` adds **soft** location/weather amplification onto the
  fused driver risk. Context can raise risk toward higher levels **only when
  there is already driver risk**; it never manufactures a fatigue diagnosis on
  its own.
- `context_urgency_from(location_risk, weather_risk, zone_type)` returns a
  **deterministic urgency modifier** in `[0.5, 1.0]` (1.0 = normal context).
  It is derived from the location risk axis, the weather risk axis, and the
  verified zone type; it returns to 1.0 as soon as the high-risk context
  disappears. The camera loop passes it to `AlertEscalationController`,
  where it scales the per-severity audio cooldown only (`cooldown *
  context_urgency`): a high-risk context lets an *existing* alert repeat
  sooner, but it never creates an alert or changes severity on its own.
- Night and driving duration already reach the score through the fatigue
  engine (bounded, small weights) — not as independent fatigue claims.
- The result is smoothed (asymmetric: rises fast, falls slowly), then mapped
  to `SAFE / MODERATE / HIGH / CRITICAL` and an alert severity.

## UI surfaces

- **ALERTNESS SCORE** (main metric), severity band, risk bar, and reason.
- **Risk-factor breakdown / trend** (Eye, Blink, Yawn, Night, Duration).
- **CONTEXT panel:** current time, night context, driving duration, location /
  zone (with honest wording + a transient zone entry/exit event line), weather
  (condition, temperature, rain, visibility, risk, and rain-expected text).
- **Bottom status bar:** STOP DRIVING and EXPORT PDF (fixed, always visible).

## Future simulation (not implemented)

The intended simulation path is a *different input mode* that implements the
same context API (`LocationContext` / `EnvironmentalContext` / time context)
so CARLA GNSS/weather/time could replace real GPS/weather/time without
rewriting the Attention Engine. No CARLA dependency or interface exists yet.
