# Dataset Evaluation Report

## UTA-RLDD

- **Dataset description used by evaluator:** The on-disk video packaging under `UTA-RLDD(videos)` contained Fold1, parts 1 and 2, 12 numbered subject folders and 36 videos. Labels were resolved from video filename stems (`0` = ALERT, `5` = LOW_VIGILANT, `10` = DROWSY); the binary evaluation task maps ALERT to ACTIVE and LOW_VIGILANT/DROWSY to FATIGUE. The evaluator's cited official dataset description is 60 subjects x 3 classes = 180 videos across five folds.
- **Total videos discovered:** 36
- **Evaluated videos:** 36
- **Skipped/unresolved videos:** 0
- **Failed videos:** 0
- **Frames processed:** 616,732 (all 616,732 decoded/source frames; no decode-skipped frames)
- **Ground-truth class distribution (binary):** {'ACTIVE': 12, 'FATIGUE': 24}
- **Source ground-truth distribution:** {'ALERT': 12, 'DROWSY': 12, 'LOW_VIGILANT': 12}
- **Predicted class distribution:** {'ACTIVE': 36}
- **Accuracy:** 33.33%
- **Precision (macro):** 16.67%
- **Recall (macro):** 50.00%
- **F1-score (macro):** 25.00%

**Per-class metrics** (precision, recall, F1):

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| ACTIVE | 33.33% | 100.00% | 50.00% |
| FATIGUE | 0.00% | 0.00% | 0.00% |

**Confusion matrix** (rows = actual, columns = predicted):

| Actual \ Predicted | ACTIVE | FATIGUE |
|---|---:|---:|
| ACTIVE | 12 | 0 |
| FATIGUE | 24 | 0 |


**Temporal-processing statistics:** The evaluator processed every frame sequentially at each video's reported container FPS, retaining temporal tracker state within each clip and initializing a fresh pipeline per video. Total source duration: 21558.97 s (598.86 s/video on average). Mean face-detection rate: 99.94%; valid face frames: 616,376. Assumed-FPS videos: 0. Recorded temporal diagnostics: 4,017 blinks, 7 yawns, 17 videos with prolonged eye closure, 9 videos with distraction, and 121 alert episodes. Processing time reported by the evaluator: 38927.4 s.

**Important evaluation limitations:** This local packaging covers only 12 subjects from Fold1 (36 clips), not all 180 videos in the cited official dataset description. Official source classes are collapsed into a binary task; thus low-vigilant and drowsy are combined. Video predictions use the evaluator's existing majority-vote rule over per-frame ACTIVE/FATIGUE labels; insufficient-data and unresolved labels are excluded (none occurred). Time context was fixed to neutral day (12:00); no CARLA, geo/weather context, or audio playback was used. This is a dataset-specific measurement, not a deployment/safety claim.

## YawDD

- **Dataset description used by evaluator:** Labeled videos were discovered under the YawDD `Mirror` directory and classified by filename condition. NORMAL/TALKING map to NON_YAWNING; YAWNING/TALKING_YAWNING map to YAWNING. The unlabeled/mixed `Dash` videos were excluded from metrics.
- **Total labeled videos:** 320
- **Evaluated videos:** 320
- **Skipped/failed videos:** 0 failed; 0 unlabeled Mirror videos excluded
- **Total source frames:** 223,581
- **Frames actually evaluated:** 223,581 (all source frames; no frames skipped)
- **Ground-truth class distribution:** {'NON_YAWNING': 205, 'YAWNING': 115}
- **Predicted class distribution:** {'NON_YAWNING': 222, 'YAWNING': 98}
- **Accuracy:** 94.69%
- **Precision (YAWNING positive class, as reported by evaluator):** 100.00%
- **Recall (YAWNING positive class, as reported by evaluator):** 85.22%
- **F1-score (YAWNING positive class, as reported by evaluator):** 92.02%

**Per-class metrics** (precision, recall, F1):

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| NON_YAWNING | 92.34% | 100.00% | 96.02% |
| YAWNING | 100.00% | 85.22% | 92.02% |

**Confusion matrix** (rows = actual, columns = predicted):

| Actual \ Predicted | NON_YAWNING | YAWNING |
|---|---:|---:|
| NON_YAWNING | 205 | 0 |
| YAWNING | 17 | 98 |


**Yawning detection statistics:** 98 videos had one or more detected yawns; 119 total yawn events and 5,664 frames in the yawn state were recorded. The sum of per-video maximum recorded yawn durations is 286.22 s; the largest per-video recorded yawn duration is 9.21 s. There were 0 insufficient-data videos and 29 Dash videos excluded. Face-valid frames: 222,035.

**Important evaluation limitations:** Ground-truth labels are condition-level filename labels, not frame-by-frame yawn annotations. The task is YAWNING vs NON_YAWNING and does not assess talking/singing classification. Video-level prediction is based on the existing temporal yawn detector's event count, with insufficient-data handling unchanged. Dash videos have no labels usable by this evaluator and were excluded. The reported precision/recall/F1 are for YAWNING as the positive class, matching the evaluator JSON.

## Comparison

| Metric | UTA-RLDD | YawDD |
|--------|----------|-------|
| Videos evaluated | 36 | 320 |
| Frames evaluated | 616,732 | 223,581 |
| Accuracy | 33.33% | 94.69% |
| Precision | 16.67% (macro) | 100.00% (YAWNING positive class) |
| Recall | 50.00% (macro) | 85.22% (YAWNING positive class) |
| F1-score | 25.00% (macro) | 92.02% (YAWNING positive class) |

These datasets measure different aspects of the system: UTA-RLDD measures video-level active/fatigue classification, while YawDD measures video-level yawning detection from filename-level condition labels. Their metric definitions and ground truth are not interchangeable; no ranking is implied.

## System Integrity

- Production code modified: **NO**
- FatigueEngine modified: **NO**
- AttentionEngine modified: **NO**
- Detection thresholds changed: **NO**
- Weights changed: **NO**
- Alert logic changed: **NO**
- Dataset modified: **NO**
- Dataset reorganized: **NO**
- CARLA used for evaluation: **NO**
- DROZY evaluated: **NO**
- UTA-RLDD sampled: **NO**
- YawDD sampled: **NO**

The YawDD evaluator received only an output-directory CLI option so its completed artifacts could be written to the requested dedicated folder without overwriting existing results; no production behavior or evaluation logic was changed.

**Worktree note:** During the long-running evaluations, unrelated modifications appeared in `src/app_ui.py`, `config/location_cache.json`, and `config/weather_cache.json`. I did not edit or revert them. The integrity confirmations above describe changes made for this evaluation task; those unrelated worktree modifications were preserved untouched.
