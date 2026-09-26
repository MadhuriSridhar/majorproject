#!/usr/bin/env python3
"""Headless VIDEO-level evaluation of the EXISTING driver-attention pipeline on the
video release of the UTA-RLDD (UTA Real-Life Drowsiness Dataset).

Why this file exists
--------------------
``scripts/evaluate_uta_rldd.py`` evaluated a *still-image* packaging of UTA-RLDD,
where every sample was a single frame. That is not a valid test of a temporal
system. This evaluator is a SEPARATE script for the *video* packaging
(``datasets/UTA-RLDD(videos)/Fold*_part*/<subject>/<clip>``) and it processes the
real frames of every video sequentially, in order, with the real video timing.

Production code is NOT touched by this file
-------------------------------------------
No module under ``src/`` is imported with intent to modify it, no threshold is
changed, no CARLA condition is invented and the dataset is never written to. The
evaluator only *consumes* the existing production components:

    facial_landmarks.detect_landmarks / create_face_landmarker
    eye_features.analyze_eyes                    (EAR)
    mouth_features.analyze_mouth                 (MAR)
    temporal_eye.EyeClosureTracker               (blinks, prolonged closure)
    temporal_mouth.YawnTracker                   (yawns)
    distraction_monitor.DistractionMonitor       (head pose / distraction)
    evidence_bridge.build_fatigue_evidence
    fatigue_engine.FatigueEngine
    attention_engine.AttentionEngine
    alert_escalation.AlertEscalationController   (alert/audio escalation state machine)
    driving_duration.DrivingDurationTracker      (neutral session clock, no CARLA)
    time_context.get_time_context                (day/neutral by default - see CLI)

Video-level aggregation rule
----------------------------
The rule is NOT invented here. It is imported from the previous evaluator
(``scripts/evaluate_uta_rldd.py``) so both evaluations use one definition:

    ``choose_video_prediction(frame_labels, mean_alertness)``
        - majority vote over the per-frame ACTIVE/FATIGUE labels produced by the
          existing pipeline for every analysed frame of the video,
        - exact tie -> ACTIVE when mean alertness >= 50 else FATIGUE,
    ``map_alertness_to_class(severity, alertness)``
        - existing project severities NORMAL/SAFE -> ACTIVE,
          LOW_FATIGUE/ALERT/WARNING/DROWSY -> FATIGUE,
    ``INSUFFICIENT_DATA`` when there are no valid frames or the face-detection
    rate is below ``MIN_FACE_DETECTION_RATE`` (0.15), identical to the previous
    evaluator. INSUFFICIENT_DATA is reported separately and is never counted as
    ACTIVE/FATIGUE.

Label discovery (no assumption that "01..06" are classes)
---------------------------------------------------------
The official UTA-RLDD project description states that the dataset contains 60
subjects x 3 videos = 180 videos, one per class, with the participant-provided
labels ``alert = 0``, ``low vigilant = 5`` and ``drowsy = 10``, and that the 60
subjects are split into five folds of 12 participants for cross-validation:
https://sites.google.com/view/utarldd/home

Because the on-disk layout must be verified instead of assumed, the evaluator
*derives* every label from the filesystem and prints a discovery report:

  1. an annotation / list file (CSV-ish) if one is present and usable,
  2. an ACTIVE/FATIGUE keyword inside the *video file name*,
  3. an ACTIVE/FATIGUE keyword inside a parent folder name (deepest first),
  4. the official numeric class value when it is the *video file stem*
     (0 / 5 / 10) - never a numbered *folder*, because those numbered folders
     are subject ids in this packaging and must not be read as class labels.

Anything that cannot be resolved is reported as UNKNOWN and excluded from the
metrics with an explicit reason; nothing is guessed. The binary task collapses
the three official classes to ACTIVE (alert) vs FATIGUE (low vigilant + drowsy);
the per-source-class prediction breakdown and the alternative reduction
(alert + low vigilant vs drowsy) are reported as well.

Usage (no GUI, no microphone, no CARLA, dataset opened read-only)
-----------------------------------------------------------------
    python scripts/evaluate_uta_rldd_videos.py --discover-only
    python scripts/evaluate_uta_rldd_videos.py --sanity 6
    python scripts/evaluate_uta_rldd_videos.py                  # full evaluation
    python scripts/evaluate_uta_rldd_videos.py --folds Fold1_part1,Fold2_part1
    python scripts/evaluate_uta_rldd_videos.py --sample-fps 10

``--sample-fps`` still decodes every frame sequentially from the container (no
frame skipping inside the tracker timeline); it only decides which decoded frames
are *analysed*, and every timestamp comes from the real container FPS. A full
frame-by-frame pass over 180 ten-minute videos takes many hours of CPU, so the
sampling flag exists for practical runs; the default is to analyse every frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for _path in (SRC, SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from alert_escalation import AlertEscalationController  # noqa: E402
from attention_engine import AttentionEngine  # noqa: E402
from distraction_monitor import DistractionMonitor  # noqa: E402
from driving_duration import DrivingDurationTracker  # noqa: E402
from evidence_bridge import build_fatigue_evidence  # noqa: E402
from eye_features import analyze_eyes  # noqa: E402
from facial_landmarks import create_face_landmarker, detect_landmarks  # noqa: E402
from fatigue_engine import FatigueEngine  # noqa: E402
from mouth_features import analyze_mouth  # noqa: E402
from temporal_eye import EyeClosureTracker  # noqa: E402
from temporal_mouth import YawnTracker  # noqa: E402
from time_context import get_time_context  # noqa: E402

try:
    # Single source of truth for the existing per-frame severity mapping, the
    # video-level aggregation rule and the confusion-matrix metrics.
    from evaluate_uta_rldd import (  # noqa: E402
        CLASS_NAMES,
        SYSTEM_MAPPING,
        choose_video_prediction,
        compute_metrics,
        map_alertness_to_class,
        safe_max,
        safe_mean,
        safe_min,
    )
except ImportError as exc:  # pragma: no cover - guards a broken checkout
    raise SystemExit(
        "scripts/evaluate_uta_rldd.py must remain in place: this evaluator reuses "
        f"its severity mapping and video-level aggregation rule ({exc})."
    )

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None

# ---------------------------------------------------------------------------
# Dataset locations (discovered at runtime; nothing is moved or renamed)
# ---------------------------------------------------------------------------

DATASET_CANDIDATES = (
    ROOT / "datasets" / "UTA-RLDD(videos)",
    ROOT.parent / "datasets" / "UTA-RLDD(videos)",
    ROOT / "datasets" / "UTA-RLDD" / "videos",
    ROOT.parent / "datasets" / "UTA-RLDD" / "videos",
    ROOT / "datasets",
    ROOT.parent / "datasets",
)

SEARCH_DIRS = (ROOT / "datasets", ROOT.parent / "datasets")

DEFAULT_RESULTS_DIR = ROOT / "results" / "uta_rldd_videos"

# Only real video containers are evaluated; the still-image packaging of the
# previous evaluation is deliberately NOT part of this task.
VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".m4v",
    ".webm",
    ".wmv",
    ".mpg",
    ".mpeg",
    ".3gp",
}

METADATA_EXTENSIONS = {".csv", ".json", ".txt", ".md", ".yaml", ".yml", ".lst"}

# ---------------------------------------------------------------------------
# Label space
# ---------------------------------------------------------------------------

BINARY_CLASSES = ("ACTIVE", "FATIGUE")          # == CLASS_NAMES of the previous evaluator
SOURCE_CLASSES = ("ALERT", "LOW_VIGILANT", "DROWSY", "FATIGUE_GENERIC", "UNKNOWN")

# Official UTA-RLDD participant labels (project page): alert = 0, low vigilant = 5,
# drowsy = 10. Accepted ONLY as a video file stem, never as a folder name, because
# the numbered folders in this packaging are subject ids.
OFFICIAL_CLASS_VALUES = {
    "0": "ALERT",
    "00": "ALERT",
    "5": "LOW_VIGILANT",
    "10": "DROWSY",
}

# Keyword -> source class. Tokens are compared after normalization
# (lowercase, non-alphanumeric removed), so "low_vigilant", "low-vigilant" and
# "lowVigilant" all become "lowvigilant".
SOURCE_CLASS_KEYWORDS = {
    "alert": "ALERT",
    "alertness": "ALERT",
    "active": "ACTIVE_ALIAS",          # alias resolved to ALERT (dataset ACTIVE class)
    "awake": "ALERT",
    "nondrowsy": "ALERT",
    "notdrowsy": "ALERT",
    "normal": "ALERT",
    "vigilant": "ALERT",
    "lowvigilant": "LOW_VIGILANT",
    "lowvigilance": "LOW_VIGILANT",
    "drowsy": "DROWSY",
    "drowsiness": "DROWSY",
    "drowsyness": "DROWSY",
    "sleepy": "DROWSY",
    "sleepiness": "DROWSY",
    "tired": "DROWSY",
    "microsleep": "DROWSY",
    "sleep": "DROWSY",
    "fatigue": "FATIGUE_GENERIC",
    "fatigued": "FATIGUE_GENERIC",
}

# Binary collapse used for the primary reported task (documented, not tuned):
# alert -> ACTIVE; low vigilant (early drowsiness) and drowsy -> FATIGUE.
SOURCE_CLASS_TO_BINARY = {
    "ALERT": "ACTIVE",
    "LOW_VIGILANT": "FATIGUE",
    "DROWSY": "FATIGUE",
    "FATIGUE_GENERIC": "FATIGUE",
}

FOLD_PATTERN = re.compile(r"^fold[\s_\-]*(\d+)", re.IGNORECASE)
PART_PATTERN = re.compile(r"part[\s_\-]*(\d+)", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Constants mirrored from production (copied, never changed)
# ---------------------------------------------------------------------------

# scripts/evaluate_uta_rldd.py: MIN_FACE_DETECTION_RATE (insufficient-data gate)
MIN_FACE_DETECTION_RATE = 0.15

# src/app_ui.py: FACE_LOSS_GRACE_S / SESSION_WARMUP_MIN_S / SESSION_WARMUP_MIN_FRAMES
FACE_LOSS_GRACE_S = 1.5
SESSION_WARMUP_MIN_S = 1.5
SESSION_WARMUP_MIN_FRAMES = 5

# src/app_ui.py: ALERT trust factors passed to AlertEscalationController ("fatigue"
# above the existing score, "stable" otherwise).
ALERT_FATIGUE_FACTOR_SCORE = 50.0

# Only used when a container reports fps <= 0. Recorded per video as
# ``fps_assumed = True`` so an assumed timeline can never pass unnoticed.
ASSUMED_FPS_FALLBACK = 30.0

# Deterministic day context for a dataset evaluation (the recordings' true
# time-of-day is unknown). Override with --use-real-time-context to reproduce the
# previous evaluator's wall-clock behaviour.
NEUTRAL_TIME_OF_DAY = "12:00:00"

# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------


@dataclass
class VideoItem:
    """One discovered video with its filesystem-derived label metadata."""

    path: Path
    relative_path: str
    fold: str = "UNKNOWN"
    part: str = ""
    subject: str = "UNKNOWN"
    source_class: str = "UNKNOWN"
    ground_truth: str = "UNKNOWN"
    label_source: str = "unresolved"

    @property
    def video_name(self) -> str:
        return self.path.name


def normalize_token(value: str) -> str:
    """Lowercase a name and drop every non-alphanumeric character."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def source_class_from_keywords(*names: str) -> tuple[str, str] | None:
    """Return (source_class, matched_name) for the first keyword hit, else None."""
    for name in names:
        token = normalize_token(name)
        if not token:
            continue
        for keyword, source_class in SOURCE_CLASS_KEYWORDS.items():
            if keyword and keyword in token:
                # "lowvigilant" must not be read as the ACTIVE keyword "vigilant".
                if keyword == "vigilant" and "lowvigilant" in token:
                    continue
                resolved = "ALERT" if source_class == "ACTIVE_ALIAS" else source_class
                return resolved, keyword
    return None


def source_class_from_stem_value(stem: str) -> str | None:
    """Official numeric class value from a *file* stem (0 / 5 / 10 only).

    Accepts an exact match, or the last numeric token of the stem (e.g.
    ``Fold1_part1_01_10`` -> 10). Folder names are never consulted this way,
    because the numbered folders in this packaging are subject ids.
    """
    token = normalize_token(stem)
    exact = OFFICIAL_CLASS_VALUES.get(token)
    if exact is not None:
        return exact
    tokens = [part for part in re.split(r"[^0-9a-zA-Z]+", str(stem)) if part]
    for candidate in reversed(tokens):
        if candidate.isdigit() and candidate in OFFICIAL_CLASS_VALUES:
            return OFFICIAL_CLASS_VALUES[candidate]
    return None


def iter_video_files(root: Path) -> list[Path]:
    """Recursively list video files without touching the dataset."""
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def resolve_dataset_root(explicit: str | None) -> Path | None:
    """Explicit --dataset-root wins; otherwise probe candidates/known parents."""
    if explicit:
        path = Path(explicit).expanduser()
        return path.resolve() if path.exists() else None

    for candidate in DATASET_CANDIDATES:
        if candidate.is_dir() and iter_video_files(candidate):
            return candidate

    for base in SEARCH_DIRS:
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            if "uta" in child.name.lower() and iter_video_files(child):
                return child
    return None


def describe_location(path: Path, root: Path) -> tuple[str, str, str]:
    """Return (fold, part, subject) derived from the relative path."""
    try:
        parts = list(path.relative_to(root).parts)
    except ValueError:
        parts = list(path.parts)
    folders = parts[:-1]  # drop the file name

    fold = "UNKNOWN"
    part = ""
    for name in folders:
        match = FOLD_PATTERN.match(name)
        if match and fold == "UNKNOWN":
            fold = f"Fold{int(match.group(1))}"
        if PART_PATTERN.search(name):
            part = name

    subject = "UNKNOWN"
    for name in reversed(folders):
        if re.fullmatch(r"\d{1,3}", name):
            subject = name
            break
    return fold, part, subject


def discover_metadata_files(root: Path) -> list[Path]:
    """List candidate annotation/documentation files (root and a few levels deep)."""
    found: list[Path] = []
    if not root.exists():
        return found
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in METADATA_EXTENSIONS:
            continue
        try:
            depth = len(path.relative_to(root).parts)
        except ValueError:
            depth = 99
        if depth <= 3:
            found.append(path)
    return found


def load_annotation_map(
    metadata_files: list[Path],
) -> tuple[dict[str, str], list[str]]:
    """Best-effort label table from a CSV-ish list file.

    A file is only used when it has a name-like column plus a label-like column
    and at least 90% of its label values resolve to a known keyword/value. An
    unusable file is reported, never silently guessed from.
    """
    mapping: dict[str, str] = {}
    notes: list[str] = []
    name_keys = ("file", "filename", "video", "videoname", "path", "clip", "name")
    label_keys = ("label", "class", "state", "category", "drowsiness", "level")

    for path in metadata_files:
        if path.suffix.lower() != ".csv":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            notes.append(f"{path.name}: unreadable ({exc})")
            continue
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            notes.append(f"{path.name}: empty")
            continue

        header = [normalize_token(cell) for cell in lines[0].split(",")]
        name_idx = next((i for i, cell in enumerate(header) if any(k in cell for k in name_keys)), None)
        label_idx = next((i for i, cell in enumerate(header) if any(k in cell for k in label_keys)), None)
        if name_idx is None or label_idx is None:
            notes.append(f"{path.name}: no name/label columns (header={lines[0][:70]!r})")
            continue

        file_map: dict[str, str] = {}
        rows = 0
        resolved = 0
        for line in lines[1:]:
            cells = [cell.strip().strip('"') for cell in line.split(",")]
            if max(name_idx, label_idx) >= len(cells):
                continue
            rows += 1
            hit = source_class_from_keywords(cells[label_idx])
            if hit is None:
                value_class = source_class_from_stem_value(cells[label_idx])
                hit = (value_class, "value") if value_class else None
            if hit is None:
                continue
            resolved += 1
            raw_name = cells[name_idx].replace("\\", "/")
            file_map[Path(raw_name).name] = hit[0]
            file_map[normalize_token(raw_name)] = hit[0]

        if rows and resolved / rows >= 0.90:
            mapping.update(file_map)
            notes.append(f"{path.name}: USED ({resolved}/{rows} rows resolved)")
        else:
            notes.append(f"{path.name}: ignored ({resolved}/{rows} rows resolved)")
    return mapping, notes


def build_dataset_items(root: Path) -> tuple[list[VideoItem], dict]:
    """Discover every video and derive labels/locations from the filesystem."""
    files = iter_video_files(root)
    metadata_files = discover_metadata_files(root)
    annotation_map, annotation_notes = load_annotation_map(metadata_files)

    items: list[VideoItem] = []
    for path in files:
        fold, part, subject = describe_location(path, root)
        rel = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        stem = path.stem
        source_class = "UNKNOWN"
        label_source = "unresolved"

        for key in (path.name, stem, normalize_token(stem), normalize_token(path.name)):
            if key in annotation_map:
                source_class = annotation_map[key]
                label_source = "annotation_file"
                break

        if source_class == "UNKNOWN":
            hit = source_class_from_keywords(stem)
            if hit is not None:
                source_class, keyword = hit
                label_source = f"filename_keyword:{keyword}"

        if source_class == "UNKNOWN":
            parts = list(path.relative_to(root).parts) if path.is_relative_to(root) else list(path.parts)
            for folder in reversed(parts[:-1]):
                hit = source_class_from_keywords(folder)
                if hit is not None:
                    source_class, keyword = hit
                    label_source = f"folder_keyword:{keyword}"
                    break

        if source_class == "UNKNOWN":
            value_class = source_class_from_stem_value(stem)
            if value_class is not None:
                source_class = value_class
                label_source = "filename_class_value"

        ground_truth = SOURCE_CLASS_TO_BINARY.get(source_class, "UNKNOWN")
        items.append(
            VideoItem(
                path=path,
                relative_path=rel,
                fold=fold,
                part=part,
                subject=subject,
                source_class=source_class,
                ground_truth=ground_truth,
                label_source=label_source,
            )
        )

    discovery = {
        "dataset_root": str(root),
        "dataset_exists": root.is_dir(),
        "video_extensions_found": dict(sorted(Counter(p.suffix.lower() for p in files).items())),
        "total_video_files": len(files),
        "label_source_counts": dict(sorted(Counter(item.label_source for item in items).items())),
        "source_class_counts": dict(sorted(Counter(item.source_class for item in items).items())),
        "binary_label_counts": dict(sorted(Counter(item.ground_truth for item in items).items())),
        "fold_structure": {},
        "part_structure": {},
        "distinct_subjects": len({item.subject for item in items}),
        "unresolved_examples": [item.relative_path for item in items if item.ground_truth == "UNKNOWN"][:10],
        "sample_video_paths": [item.relative_path for item in items[:12]],
        "filename_stem_counts": dict(
            sorted(Counter(item.path.stem for item in items).items())[:25]
        ),
        "metadata_files": [str(p.relative_to(root)) for p in metadata_files],
        "metadata_notes": annotation_notes,
        "annotation_map_entries": len(annotation_map),
        "folder_numeric_id_policy": (
            "numbered folders are subject ids; official numeric class values "
            "(0/5/10) are accepted only from a video file stem"
        ),
        "official_reference": (
            "UTA-RLDD: 180 videos, 60 subjects x 3 classes (alert=0, low vigilant=5, "
            "drowsy=10), 5 folds of 12 subjects (https://sites.google.com/view/utarldd/home)"
        ),
    }

    for item in items:
        bucket = discovery["fold_structure"].setdefault(
            item.fold, {"videos": 0, "parts": set(), "subjects": set()}
        )
        bucket["videos"] += 1
        bucket["subjects"].add(item.subject)
        if item.part:
            bucket["parts"].add(item.part)
        part_bucket = discovery["part_structure"].setdefault(
            item.part or "(none)", {"videos": 0, "classes": Counter()}
        )
        part_bucket["videos"] += 1
        part_bucket["classes"][item.source_class] += 1

    for bucket in discovery["fold_structure"].values():
        bucket["parts"] = sorted(bucket["parts"])
        bucket["subjects"] = sorted(bucket["subjects"])
    for part_bucket in discovery["part_structure"].values():
        part_bucket["classes"] = dict(sorted(part_bucket["classes"].items()))

    return items, discovery


def print_discovery_report(root: Path, discovery: dict) -> None:
    print("=" * 70)
    print("UTA-RLDD (VIDEOS) DATASET DISCOVERY")
    print("=" * 70)
    print(f"Root              : {root}")
    print(f"Video extensions  : {discovery['video_extensions_found']}")
    print(f"Video files       : {discovery['total_video_files']}")
    print(f"Distinct subjects : {discovery['distinct_subjects']}")
    print("Fold / part structure:")
    for fold, bucket in sorted(discovery["fold_structure"].items()):
        print(
            f"  {fold}: {bucket['videos']} videos | {len(bucket['subjects'])} subjects "
            f"| parts={bucket['parts']}"
        )
    for part, bucket in sorted(discovery["part_structure"].items()):
        print(f"  {part}: {bucket['videos']} videos | classes={bucket['classes']}")
    print("Label sources:")
    for source, count in discovery["label_source_counts"].items():
        print(f"  {source}: {count}")
    print("Source classes:")
    for name in SOURCE_CLASSES:
        print(f"  {name}: {discovery['source_class_counts'].get(name, 0)}")
    print("Binary ground truth:")
    for name in BINARY_CLASSES + ("UNKNOWN",):
        print(f"  {name}: {discovery['binary_label_counts'].get(name, 0)}")
    if discovery["unresolved_examples"]:
        print("  unresolved examples: " + ", ".join(discovery["unresolved_examples"]))
    print(f"Metadata files    : {discovery['metadata_files'] or 'none found'}")
    for note in discovery["metadata_notes"]:
        print(f"  - {note}")
    print(f"Filename stems    : {discovery.get('filename_stem_counts', {})}")
    print("Sample paths      :")
    for path in discovery.get("sample_video_paths", []):
        print(f"  - {path}")
    print(f"Policy            : {discovery['folder_numeric_id_policy']}")
    print(f"Reference         : {discovery['official_reference']}")
    print("=" * 70)


def folds_match(item: VideoItem, selectors: list[str]) -> bool:
    if not selectors:
        return True
    haystack = f"{item.fold}|{item.part}|{item.relative_path}".lower()
    return any(selector.lower() in haystack for selector in selectors)


def select_items(
    items: list[VideoItem],
    folds: list[str],
    limit_per_class: int = 0,
    max_videos: int = 0,
) -> list[VideoItem]:
    """Filter by fold/part and optionally take a class-balanced subset."""
    candidates = [item for item in items if folds_match(item, folds)]
    if limit_per_class > 0:
        buckets: dict[str, list[VideoItem]] = {}
        for item in candidates:
            if item.ground_truth == "UNKNOWN":
                continue
            buckets.setdefault(item.source_class, []).append(item)
        selected: list[VideoItem] = []
        index = 0
        while True:
            added = False
            for key in sorted(buckets):
                bucket = buckets[key]
                if index < len(bucket) and index < limit_per_class:
                    selected.append(bucket[index])
                    added = True
            if not added:
                break
            index += 1
        candidates = selected
    if max_videos > 0:
        candidates = candidates[:max_videos]
    return candidates


# ---------------------------------------------------------------------------
# Production behaviour mirrored locally (pure helpers, copied from src/app_ui.py)
# ---------------------------------------------------------------------------


def head_is_down(head) -> bool:
    """Mirror of src/app_ui.py head_is_down()."""
    if head is None:
        return False
    return bool(
        getattr(head, "looking_away", False)
        and getattr(head, "direction", "forward") == "down"
    )


def head_is_turned(head) -> bool:
    """Mirror of src/app_ui.py head_is_turned()."""
    if head is None:
        return False
    return bool(
        getattr(head, "looking_away", False)
        and getattr(head, "direction", "forward") in ("left", "right")
    )


def classify_face_state(face_detected: bool, lost_for_s: float | None, grace_s: float) -> str:
    """Mirror of src/app_ui.py classify_face_state()."""
    if face_detected:
        return "FACE_DETECTED"
    if lost_for_s is not None and lost_for_s < grace_s:
        return "FACE_TEMPORARILY_LOST"
    return "FACE_NOT_DETECTED"


class FaceStateTracker:
    """Mirror of the CameraWorker face-loss / startup-warmup bookkeeping.

    Same rules as src/app_ui.py: a short dropout keeps the last distraction
    observation (grace), a longer dropout feeds ``None`` into the eye/mouth
    trackers and resets the distraction monitor, and no decision is built before
    the startup warmup (SESSION_WARMUP_MIN_S seconds AND
    SESSION_WARMUP_MIN_FRAMES valid frames).
    """

    def __init__(self) -> None:
        self._face_lost_since: float | None = None
        self._valid_evidence_since: float | None = None
        self._valid_evidence_frames = 0
        self.last_distraction = None

    def _evidence_ready(self, now_s: float, face_valid: bool) -> bool:
        if not face_valid:
            self._valid_evidence_since = None
            self._valid_evidence_frames = 0
            return False
        if self._valid_evidence_since is None:
            self._valid_evidence_since = now_s
            self._valid_evidence_frames = 0
        self._valid_evidence_frames += 1
        return (
            self._valid_evidence_frames >= SESSION_WARMUP_MIN_FRAMES
            and now_s - self._valid_evidence_since >= SESSION_WARMUP_MIN_S
        )

    def update(self, num_faces: int, now_s: float) -> dict:
        face_detected = num_faces >= 1
        multiple_faces = num_faces > 1
        valid_face = num_faces == 1

        if face_detected:
            self._face_lost_since = None
        elif self._face_lost_since is None:
            self._face_lost_since = now_s
        lost_for = None if self._face_lost_since is None else now_s - self._face_lost_since

        face_state = classify_face_state(face_detected, lost_for, FACE_LOSS_GRACE_S)
        in_grace = face_state == "FACE_TEMPORARILY_LOST"
        warmup_ready = False if in_grace else self._evidence_ready(now_s, valid_face)
        return {
            "face_detected": face_detected,
            "multiple_faces": multiple_faces,
            "valid_face": valid_face,
            "in_grace": in_grace,
            "warmup_ready": warmup_ready,
            "face_state": face_state,
        }


# ---------------------------------------------------------------------------
# Per-video temporal evaluation
# ---------------------------------------------------------------------------

SEVERITY_PRIORITY = {
    "NORMAL": 0,
    "LOW_FATIGUE": 1,
    "ALERT": 2,
    "WARNING": 3,
    "DROWSY": 4,
}
SEVERITIES = ("NORMAL", "LOW_FATIGUE", "ALERT", "WARNING", "DROWSY")
ATTENTION_LEVELS = ("SAFE", "MODERATE", "HIGH", "CRITICAL")


def build_result_record(item: VideoItem, **stats) -> dict:
    """Assemble one result row (Part 8 fields) from collected frame statistics."""
    record = {
        "video_path": str(item.path),
        "video_name": item.video_name,
        "relative_path": item.relative_path,
        "fold": item.fold,
        "part": item.part,
        "subject": item.subject,
        "ground_truth": item.ground_truth,
        "source_class": item.source_class,
        "label_source": item.label_source,
        "predicted_class": stats.get("predicted_class", "INSUFFICIENT_DATA"),
        "correct": stats.get("correct", False),
        "evaluated": stats.get("evaluated", False),
        "skip_reason": stats.get("skip_reason", ""),
        "total_frames": stats.get("total_frames", 0),
        "frames_read": stats.get("frames_read", 0),
        "processed_frames": stats.get("processed_frames", 0),
        "skipped_frames": stats.get("skipped_frames", 0),
        "valid_face_frames": stats.get("valid_face_frames", 0),
        "face_missing_frames": stats.get("face_missing_frames", 0),
        "multiple_face_frames": stats.get("multiple_face_frames", 0),
        "face_detection_rate": stats.get("face_detection_rate", 0.0),
        "fps": stats.get("fps", 0.0),
        "fps_assumed": stats.get("fps_assumed", False),
        "duration_s": stats.get("duration_s", 0.0),
        "decision_frames": stats.get("decision_frames", 0),
        "warmup_gated_frames": stats.get("warmup_gated_frames", 0),
        "max_fatigue_risk": stats.get("max_fatigue_risk"),
        "mean_fatigue_risk": stats.get("mean_fatigue_risk"),
        "final_fatigue_risk": stats.get("final_fatigue_risk"),
        "max_distraction_risk": stats.get("max_distraction_risk"),
        "mean_distraction_risk": stats.get("mean_distraction_risk"),
        "max_attention_risk": stats.get("max_attention_risk"),
        "mean_attention_risk": stats.get("mean_attention_risk"),
        "final_attention_risk": stats.get("final_attention_risk"),
        "mean_alertness": stats.get("mean_alertness"),
        "final_alertness": stats.get("final_alertness"),
        "final_severity": stats.get("final_severity"),
        "final_attention_level": stats.get("final_attention_level"),
        "max_severity": stats.get("max_severity"),
        "alert_episodes": stats.get("alert_episodes", 0),
        "audio_alert_triggers": stats.get("audio_alert_triggers", 0),
        "blink_count": stats.get("blink_count", 0),
        "yawn_count": stats.get("yawn_count", 0),
        "prolonged_closure_frames": stats.get("prolonged_closure_frames", 0),
        "max_eye_closure_s": stats.get("max_eye_closure_s", 0.0),
        "distracted_frames": stats.get("distracted_frames", 0),
        "active_percentage": stats.get("active_percentage", 0.0),
        "fatigue_percentage": stats.get("fatigue_percentage", 0.0),
        "processing_seconds": stats.get("processing_seconds", 0.0),
        "processing_fps": stats.get("processing_fps", 0.0),
        "temporal_checks": stats.get("temporal_checks", {}),
        "error": stats.get("error", ""),
    }
    for name in SEVERITIES:
        record[f"reached_{name.lower()}"] = bool(stats.get("reached_severity", {}).get(name, False))
    for name in ATTENTION_LEVELS:
        record[f"reached_{name.lower()}_level"] = bool(stats.get("reached_level", {}).get(name, False))
    return record


def evaluate_video(
    item: VideoItem,
    sample_fps: float | None = None,
    use_real_time_context: bool = False,
    progress_index: int | None = None,
    total_videos: int = 0,
) -> dict:
    """Process one video frame-by-frame with the EXISTING production pipeline.

    All trackers/engines are created fresh for this video and only for this video:
    temporal state therefore persists across the frames of the video and is reset
    before the next video starts.
    """
    cap = cv2.VideoCapture(str(item.path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {item.path}")

    fps_raw = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps_assumed = fps_raw <= 0.0
    fps = ASSUMED_FPS_FALLBACK if fps_assumed else fps_raw
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Analysis sampling interval (None = analyse every decoded frame). A requested
    # sampling rate at/above the container FPS is ignored so the sampler can never
    # starve the pipeline of frames.
    sample_interval = (
        1.0 / float(sample_fps)
        if (sample_fps is not None and sample_fps > 0.0 and sample_fps < fps)
        else None
    )

    # Fresh state for every video (never shared between videos).
    landmarker = create_face_landmarker()
    eye_tracker = EyeClosureTracker()
    yawn_tracker = YawnTracker()
    fatigue_engine = FatigueEngine()
    attention_engine = AttentionEngine()
    distraction_monitor = DistractionMonitor()
    alert_controller = AlertEscalationController()
    driving_tracker = DrivingDurationTracker(start_s=0.0)  # neutral clock, no CARLA
    face_state = FaceStateTracker()

    tracker_ids = {
        "eye_tracker": id(eye_tracker),
        "yawn_tracker": id(yawn_tracker),
        "fatigue_engine": id(fatigue_engine),
        "attention_engine": id(attention_engine),
        "distraction_monitor": id(distraction_monitor),
        "alert_controller": id(alert_controller),
    }

    started = time.perf_counter()
    read_frames = 0
    processed_frames = 0
    frame_skip_count = 0
    valid_face_frames = 0
    face_missing_frames = 0
    multiple_face_frames = 0
    decision_frames = 0
    warmup_gated_frames = 0
    previous_now_s: float | None = None
    timestamps_increasing = True
    fresh_counts_zero = True
    first_tracker_update_seen = False
    # Cumulative tracker counters must never go backwards inside one video: that
    # is the direct evidence that temporal state persists across frames.
    counters_monotonic = True
    previous_blink_total = 0
    previous_yawn_total = 0

    frame_labels: list[str] = []
    frame_severities: list[str] = []
    frame_levels: list[str] = []
    alertness_values: list[float] = []
    fatigue_values: list[float] = []
    distraction_values: list[float] = []
    combined_values: list[float] = []
    eye_states: Counter = Counter()
    head_directions: Counter = Counter()
    reached_severity = {name: False for name in SEVERITIES}
    reached_level = {name: False for name in ATTENTION_LEVELS}
    blink_total = 0
    yawn_total = 0
    prolonged_closure_frames = 0
    max_eye_closure_s = 0.0
    distracted_frames = 0
    alert_episodes = 0
    audio_alert_triggers = 0
    previous_severity = "NORMAL"
    distraction_enabled = True  # production default (src/app_ui.py)

    try:
        read_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            read_index += 1
            read_frames += 1

            # Timestamp comes from the REAL video timeline (frame index / container FPS).
            now_s = float(read_index - 1) / fps
            if previous_now_s is not None and now_s <= previous_now_s:
                timestamps_increasing = False
            previous_now_s = now_s

            if sample_interval is not None:
                # Same sampling rule as scripts/evaluate_uta_rldd.py: every frame is
                # still decoded sequentially; only this decision skips *analysis*.
                process_frame = (now_s % sample_interval) < (1.0 / (fps * 100.0)) or read_index == 1
            else:
                process_frame = True
            if not process_frame:
                frame_skip_count += 1
                continue

            processed_frames += 1
            timestamp_ms = int(now_s * 1000.0)
            result = detect_landmarks(landmarker, frame, timestamp_ms)
            faces = result.face_landmarks
            num_faces = len(faces)
            state = face_state.update(num_faces, now_s)

            if not state["face_detected"]:
                face_missing_frames += 1
            elif state["multiple_faces"]:
                multiple_face_frames += 1
            if state["valid_face"]:
                valid_face_frames += 1

            h = int(frame.shape[0])
            w = int(frame.shape[1])
            eye_state = None
            mouth_state = None
            distraction_obs = None

            if state["face_detected"] and not state["multiple_faces"]:
                face = faces[0]
                eyes = analyze_eyes(face, w, h)
                mouth = analyze_mouth(face, w, h)
                eye_state = eye_tracker.update(eyes.average_ear, now_s)
                mouth_state = yawn_tracker.update(mouth.mar, now_s)
                if not first_tracker_update_seen:
                    first_tracker_update_seen = True
                    if eye_state.blink_count != 0 or mouth_state.yawn_count != 0:
                        fresh_counts_zero = False
                if state["valid_face"] and not state["warmup_ready"] and eye_state.eye == "OPEN":
                    distraction_monitor.head_tracker.calibrate(face, w, h)
                if state["warmup_ready"]:
                    distraction_monitor.head_tracker.finish_calibration()
                if distraction_enabled:
                    distraction_obs = distraction_monitor.update(face, w, h, now_s)
                else:
                    distraction_monitor.reset()
            elif state["in_grace"]:
                # Temporary dropout (src/app_ui.py): keep the last distraction state.
                distraction_obs = face_state.last_distraction
            else:
                eye_state = eye_tracker.update(None, now_s)
                mouth_state = yawn_tracker.update(None, now_s)
                distraction_monitor.reset()

            if distraction_obs is not None:
                face_state.last_distraction = distraction_obs

            if eye_state is not None:
                eye_states[eye_state.eye] += 1
                blink_total = int(eye_state.blink_count)
                if eye_state.prolonged_closure:
                    prolonged_closure_frames += 1
                max_eye_closure_s = max(max_eye_closure_s, float(eye_state.current_closure_s))
            if mouth_state is not None:
                yawn_total = int(mouth_state.yawn_count)
            if blink_total < previous_blink_total or yawn_total < previous_yawn_total:
                counters_monotonic = False
            previous_blink_total = blink_total
            previous_yawn_total = yawn_total

            head = distraction_obs.head if distraction_obs is not None else None
            if head is not None:
                head_directions[head.direction] += 1
            if distraction_obs is not None and bool(distraction_obs.active):
                distracted_frames += 1

            # Production builds a decision only with exactly one face and only
            # after the startup warmup; otherwise this frame stays a paused frame.
            if not (state["valid_face"] and state["warmup_ready"]):
                if state["valid_face"]:
                    warmup_gated_frames += 1
                continue

            head_down = head_is_down(head)
            head_turned = head_is_turned(head)
            # Neutral day context by default (the recordings' true time-of-day is
            # unknown); no CARLA and no location/weather context is injected.
            time_ctx = (
                get_time_context()
                if use_real_time_context
                else get_time_context(force_local_time=NEUTRAL_TIME_OF_DAY)
            )
            evidence = build_fatigue_evidence(
                eye=eye_state,
                mouth=mouth_state,
                time_ctx=time_ctx,
                driving=driving_tracker.update(now_s),
                face_detected=True,
                head_down=head_down,
                head_turned=head_turned,
                eyes_not_visible=bool(head_down or head_turned),
                mouth_not_visible=bool(head_down or head_turned),
            )
            decision = fatigue_engine.update(evidence, now_s)
            fatigue_risk = (
                float(decision.smoothed_risk)
                if decision is not None and decision.smoothed_risk is not None
                else float(decision.fatigue_score or 0)
            )
            attention = attention_engine.update(
                fatigue_risk=fatigue_risk,
                distraction=distraction_obs if distraction_enabled else None,
                now_s=now_s,
                face_ok=True,
            )

            severity = str(attention.alert_severity)
            level = str(attention.level)
            distraction_risk = (
                float(distraction_obs.distraction_risk) if distraction_obs is not None else 0.0
            )
            frame_labels.append(map_alertness_to_class(severity, attention.alertness_score))
            frame_severities.append(severity)
            frame_levels.append(level)
            alertness_values.append(float(attention.alertness_score))
            fatigue_values.append(float(fatigue_risk))
            distraction_values.append(float(distraction_risk))
            combined_values.append(float(attention.combined_risk))
            decision_frames += 1
            reached_severity[severity] = True
            reached_level[level] = True
            if (
                severity in ("ALERT", "WARNING", "DROWSY")
                and SEVERITY_PRIORITY.get(severity, 0) > SEVERITY_PRIORITY.get(previous_severity, 0)
            ):
                alert_episodes += 1
            previous_severity = severity

            # Existing alert/audio escalation state machine. No audio is played in
            # this headless evaluation; triggers are only counted (diagnostic).
            alert_result = alert_controller.update(
                severity,
                now_s,
                reason=attention.primary_reason,
                contributing_factors=(
                    attention.distraction_primary,
                    "fatigue"
                    if attention.fatigue_score >= ALERT_FATIGUE_FACTOR_SCORE
                    else "stable",
                ),
                alertness_score=attention.alertness_score,
                fatigue_score=attention.fatigue_score,
                distraction_score=attention.distraction_score,
                context_urgency=attention.context_urgency,
            )
            if alert_result.should_play_audio:
                audio_alert_triggers += 1
    finally:
        cap.release()
        try:
            landmarker.close()
        except Exception:  # pragma: no cover - close() must never hide results
            pass

    processing_seconds = max(1e-9, time.perf_counter() - started)
    duration_s = float(total_frames) / fps if total_frames > 0 else float(read_frames) / fps
    face_detection_rate = (valid_face_frames / processed_frames) if processed_frames else 0.0
    active_percentage = (
        frame_labels.count("ACTIVE") / len(frame_labels) * 100.0 if frame_labels else 0.0
    )
    fatigue_percentage = (
        frame_labels.count("FATIGUE") / len(frame_labels) * 100.0 if frame_labels else 0.0
    )
    mean_alertness = safe_mean(alertness_values)

    skip_reason = ""
    if processed_frames == 0:
        predicted_class = "INSUFFICIENT_DATA"
        skip_reason = "no_frames_processed"
    elif valid_face_frames == 0 or face_detection_rate < MIN_FACE_DETECTION_RATE:
        predicted_class = "INSUFFICIENT_DATA"
        skip_reason = (
            f"insufficient_face_data (valid_frames={valid_face_frames}, "
            f"face_detection_rate={face_detection_rate:.3f})"
        )
    else:
        # Existing video-level aggregation rule (imported, not reinvented).
        predicted_class = choose_video_prediction(frame_labels, mean_alertness)

    evaluated = predicted_class != "INSUFFICIENT_DATA" and item.ground_truth != "UNKNOWN"
    if item.ground_truth == "UNKNOWN" and not skip_reason:
        skip_reason = "label_unresolved"
    correct = bool(evaluated and predicted_class == item.ground_truth)
    max_severity = (
        max(frame_severities, key=lambda name: SEVERITY_PRIORITY.get(name, 0))
        if frame_severities
        else None
    )

    record = build_result_record(
        item,
        predicted_class=predicted_class,
        correct=correct,
        evaluated=evaluated,
        skip_reason=skip_reason,
        total_frames=total_frames,
        frames_read=read_frames,
        processed_frames=processed_frames,
        skipped_frames=frame_skip_count,
        valid_face_frames=valid_face_frames,
        face_missing_frames=face_missing_frames,
        multiple_face_frames=multiple_face_frames,
        face_detection_rate=face_detection_rate,
        fps=fps,
        fps_assumed=fps_assumed,
        duration_s=duration_s,
        decision_frames=decision_frames,
        warmup_gated_frames=warmup_gated_frames,
        max_fatigue_risk=safe_max(fatigue_values),
        mean_fatigue_risk=safe_mean(fatigue_values),
        final_fatigue_risk=fatigue_values[-1] if fatigue_values else None,
        max_distraction_risk=safe_max(distraction_values),
        mean_distraction_risk=safe_mean(distraction_values),
        max_attention_risk=safe_max(combined_values),
        mean_attention_risk=safe_mean(combined_values),
        final_attention_risk=combined_values[-1] if combined_values else None,
        mean_alertness=mean_alertness,
        final_alertness=alertness_values[-1] if alertness_values else None,
        final_severity=frame_severities[-1] if frame_severities else None,
        final_attention_level=frame_levels[-1] if frame_levels else None,
        max_severity=max_severity,
        alert_episodes=alert_episodes,
        audio_alert_triggers=audio_alert_triggers,
        blink_count=blink_total,
        yawn_count=yawn_total,
        prolonged_closure_frames=prolonged_closure_frames,
        max_eye_closure_s=max_eye_closure_s,
        distracted_frames=distracted_frames,
        active_percentage=active_percentage,
        fatigue_percentage=fatigue_percentage,
        processing_seconds=processing_seconds,
        processing_fps=(processed_frames / processing_seconds) if processed_frames else 0.0,
        reached_severity=reached_severity,
        reached_level=reached_level,
        temporal_checks={
            "multiple_frames_processed": bool(processed_frames > 1),
            "timestamps_strictly_increasing": bool(timestamps_increasing),
            "tracker_state_persisted_within_video": bool(counters_monotonic),
            "tracker_counts_zero_at_video_start": bool(fresh_counts_zero),
            "decision_frames": decision_frames,
            "valid_face_frames": valid_face_frames,
            "tracker_object_ids": tracker_ids,
        },
    )
    record["dominant_eye_state"] = eye_states.most_common(1)[0][0] if eye_states else "UNKNOWN"
    record["dominant_head_direction"] = (
        head_directions.most_common(1)[0][0] if head_directions else "UNKNOWN"
    )
    record["frame_label_counts"] = {
        "ACTIVE": frame_labels.count("ACTIVE"),
        "FATIGUE": frame_labels.count("FATIGUE"),
    }

    if progress_index is not None and total_videos:
        status = "CORRECT" if correct else ("WRONG" if evaluated else "SKIPPED")
        print(
            f"[{progress_index}/{total_videos}] {item.relative_path}\n"
            f"    ground truth={item.ground_truth} ({item.source_class}) "
            f"prediction={predicted_class} -> {status}\n"
            f"    frames={processed_frames}/{read_frames} valid={valid_face_frames} "
            f"face_rate={face_detection_rate * 100.0:.1f}% fps={fps:.2f}\n"
            f"    max_fatigue={record['max_fatigue_risk']} max_attention={record['max_attention_risk']} "
            f"final_severity={record['final_severity']} blinks={blink_total} yawns={yawn_total}"
        )
    return record


# ---------------------------------------------------------------------------
# Metrics, diagnostics and output files
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "video_path",
    "video_name",
    "relative_path",
    "fold",
    "part",
    "subject",
    "ground_truth",
    "source_class",
    "label_source",
    "predicted_class",
    "correct",
    "evaluated",
    "skip_reason",
    "error",
    "total_frames",
    "frames_read",
    "processed_frames",
    "skipped_frames",
    "valid_face_frames",
    "face_missing_frames",
    "multiple_face_frames",
    "face_detection_rate",
    "fps",
    "fps_assumed",
    "duration_s",
    "decision_frames",
    "warmup_gated_frames",
    "max_fatigue_risk",
    "mean_fatigue_risk",
    "final_fatigue_risk",
    "max_distraction_risk",
    "mean_distraction_risk",
    "max_attention_risk",
    "mean_attention_risk",
    "final_attention_risk",
    "mean_alertness",
    "final_alertness",
    "final_severity",
    "final_attention_level",
    "max_severity",
    "alert_episodes",
    "audio_alert_triggers",
    "blink_count",
    "yawn_count",
    "prolonged_closure_frames",
    "max_eye_closure_s",
    "distracted_frames",
    "dominant_eye_state",
    "dominant_head_direction",
    "active_percentage",
    "fatigue_percentage",
    "processing_seconds",
    "processing_fps",
    "temporal_checks",
]


def csv_cell(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    return value


def metrics_for(records: list[dict]) -> dict:
    """Binary metrics over the evaluated (non-skipped, label-resolved) videos."""
    usable = [r for r in records if r["evaluated"]]
    actual = [r["ground_truth"] for r in usable]
    predicted = [r["predicted_class"] for r in usable]
    metrics = compute_metrics(actual, predicted)
    per_class = metrics["per_class"]
    return {
        "videos_counted": len(usable),
        "videos_excluded_from_metrics": len(records) - len(usable),
        "accuracy": metrics["accuracy"],
        "macro_precision": (
            per_class["ACTIVE"]["precision"] + per_class["FATIGUE"]["precision"]
        ) / 2.0,
        "macro_recall": (per_class["ACTIVE"]["recall"] + per_class["FATIGUE"]["recall"]) / 2.0,
        "macro_f1": (per_class["ACTIVE"]["f1"] + per_class["FATIGUE"]["f1"]) / 2.0,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": list(BINARY_CLASSES),
            "matrix": metrics["confusion_matrix"],
        },
        "matrix_counts": metrics["matrix_counts"],
        "correctly_classified": sum(1 for r in usable if r["correct"]),
        "incorrectly_classified": sum(1 for r in usable if not r["correct"]),
    }


def alt_low_vigilant_metrics(records: list[dict]) -> dict:
    """Alternative reduction of the 3 official classes: alert + low vigilant vs drowsy.

    Reporting-only: the system's predictions are untouched; only the way the
    official 3-class ground truth is collapsed changes.
    """
    usable = [r for r in records if r["evaluated"]]
    actual = [
        "ACTIVE" if r["source_class"] in ("ALERT", "LOW_VIGILANT") else "FATIGUE" for r in usable
    ]
    predicted = [r["predicted_class"] for r in usable]
    metrics = compute_metrics(actual, predicted)
    per_class = metrics["per_class"]
    return {
        "definition": "ALERT+LOW_VIGILANT = ACTIVE, DROWSY = FATIGUE",
        "videos_counted": len(usable),
        "accuracy": metrics["accuracy"],
        "macro_precision": (
            per_class["ACTIVE"]["precision"] + per_class["FATIGUE"]["precision"]
        ) / 2.0,
        "macro_recall": (per_class["ACTIVE"]["recall"] + per_class["FATIGUE"]["recall"]) / 2.0,
        "macro_f1": (per_class["ACTIVE"]["f1"] + per_class["FATIGUE"]["f1"]) / 2.0,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": list(BINARY_CLASSES),
            "matrix": metrics["confusion_matrix"],
        },
    }


def per_source_class_breakdown(records: list[dict]) -> dict:
    """Prediction distribution per official source class (no fake 3-class model)."""
    breakdown: dict[str, dict] = {}
    for record in records:
        bucket = breakdown.setdefault(
            record["source_class"],
            {"videos": 0, "predicted_ACTIVE": 0, "predicted_FATIGUE": 0, "skipped": 0},
        )
        bucket["videos"] += 1
        if record["predicted_class"] == "ACTIVE":
            bucket["predicted_ACTIVE"] += 1
        elif record["predicted_class"] == "FATIGUE":
            bucket["predicted_FATIGUE"] += 1
        else:
            bucket["skipped"] += 1
    return dict(sorted(breakdown.items()))


def temporal_signal_summary(records: list[dict]) -> dict:
    """Diagnostic measurement only - nothing here changes a threshold."""
    usable = [r for r in records if r["evaluated"]]
    return {
        "videos_analysed": len(usable),
        "videos_with_blinks": sum(1 for r in usable if r["blink_count"] > 0),
        "videos_with_yawns": sum(1 for r in usable if r["yawn_count"] > 0),
        "videos_with_prolonged_eye_closure": sum(
            1 for r in usable if r["prolonged_closure_frames"] > 0
        ),
        "videos_with_distraction": sum(1 for r in usable if r["distracted_frames"] > 0),
        "videos_reaching_normal_severity": sum(1 for r in usable if r["reached_normal"]),
        "videos_reaching_low_fatigue": sum(1 for r in usable if r["reached_low_fatigue"]),
        "videos_reaching_alert": sum(1 for r in usable if r["reached_alert"]),
        "videos_reaching_warning": sum(1 for r in usable if r["reached_warning"]),
        "videos_reaching_drowsy": sum(1 for r in usable if r["reached_drowsy"]),
        "videos_reaching_safe_level": sum(1 for r in usable if r["reached_safe_level"]),
        "videos_reaching_moderate_level": sum(1 for r in usable if r["reached_moderate_level"]),
        "videos_reaching_high_level": sum(1 for r in usable if r["reached_high_level"]),
        "videos_reaching_critical_level": sum(1 for r in usable if r["reached_critical_level"]),
        "total_blinks": sum(r["blink_count"] for r in usable),
        "total_yawns": sum(r["yawn_count"] for r in usable),
        "total_alert_episodes": sum(r["alert_episodes"] for r in usable),
        "total_audio_alert_triggers": sum(r["audio_alert_triggers"] for r in usable),
        "total_processed_frames": sum(r["processed_frames"] for r in records),
        "total_valid_face_frames": sum(r["valid_face_frames"] for r in records),
        "mean_face_detection_rate": safe_mean([r["face_detection_rate"] for r in usable]),
    }


ERROR_FIELDS = [
    "error_type",
    "relative_path",
    "video_name",
    "fold",
    "part",
    "subject",
    "duration_s",
    "ground_truth",
    "predicted_class",
    "source_class",
    "max_fatigue_risk",
    "max_distraction_risk",
    "final_severity",
    "final_attention_level",
    "max_severity",
    "blink_count",
    "yawn_count",
    "prolonged_closure_frames",
    "distracted_frames",
    "face_detection_rate",
    "processed_frames",
    "skip_reason",
    "error",
]


def error_analysis(records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (false_positives, false_negatives, skipped_or_failed) records."""
    false_positives = [
        r
        for r in records
        if r["evaluated"] and r["ground_truth"] == "ACTIVE" and r["predicted_class"] == "FATIGUE"
    ]
    false_negatives = [
        r
        for r in records
        if r["evaluated"] and r["ground_truth"] == "FATIGUE" and r["predicted_class"] == "ACTIVE"
    ]
    skipped = [r for r in records if not r["evaluated"]]
    return false_positives, false_negatives, skipped


def write_confusion_png(path: Path, metrics: dict, title: str) -> None:
    if plt is None:
        return
    matrix = metrics["confusion_matrix"]["matrix"]
    labels = metrics["confusion_matrix"]["labels"]
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual (ground truth)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color="black", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def diagnostic_entries(error_type: str, records: list[dict]) -> list[dict]:
    entries = []
    for record in records:
        entry = {"error_type": error_type}
        for field in ERROR_FIELDS:
            if field == "error_type":
                continue
            entry[field] = record.get(field)
        entries.append(entry)
    return entries


def write_outputs(
    records: list[dict],
    dataset_discovery: dict,
    results_dir: Path,
    config: dict,
) -> tuple[Path, Path, Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "uta_rldd_videos_results.csv"
    errors_path = results_dir / "uta_rldd_videos_errors.csv"
    summary_path = results_dir / "uta_rldd_videos_summary.json"
    png_path = results_dir / "uta_rldd_videos_confusion_matrix.png"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = {field: csv_cell(record.get(field)) for field in CSV_FIELDS}
            checks = dict(record.get("temporal_checks") or {})
            checks.pop("tracker_object_ids", None)
            row["temporal_checks"] = json.dumps(checks, sort_keys=True)
            writer.writerow(row)

    false_positives, false_negatives, skipped = error_analysis(records)
    with errors_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ERROR_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for group in (
            diagnostic_entries("FALSE_POSITIVE", false_positives),
            diagnostic_entries("FALSE_NEGATIVE", false_negatives),
            diagnostic_entries("SKIPPED", skipped),
        ):
            for row in group:
                writer.writerow({field: csv_cell(row.get(field)) for field in ERROR_FIELDS})

    primary = metrics_for(records)
    skipped_reasons = Counter(r["skip_reason"] for r in records if not r["evaluated"])
    summary = {
        "dataset": dataset_discovery,
        "evaluation": {
            "videos_discovered": dataset_discovery["total_video_files"],
            "videos_selected": len(records),
            "videos_evaluated": sum(1 for r in records if r["evaluated"]),
            "videos_skipped_or_unresolved": sum(1 for r in records if not r["evaluated"]),
            "videos_failed_with_error": sum(1 for r in records if r.get("error")),
            "skipped_reasons": dict(sorted(skipped_reasons.items())),
            "total_frames_read": sum(r["frames_read"] for r in records),
            "total_frames_processed": sum(r["processed_frames"] for r in records),
            "total_decode_skipped_frames": sum(r["skipped_frames"] for r in records),
            "total_valid_face_frames": sum(r["valid_face_frames"] for r in records),
            "total_duration_s": sum(r["duration_s"] for r in records),
            "mean_duration_s": safe_mean([r["duration_s"] for r in records]),
            "mean_face_detection_rate": safe_mean(
                [r["face_detection_rate"] for r in records if r["processed_frames"] > 0]
            ),
            "videos_with_assumed_fps": sum(1 for r in records if r["fps_assumed"]),
            "total_processing_seconds": sum(r["processing_seconds"] for r in records),
            "mean_processing_fps": safe_mean(
                [r["processing_fps"] for r in records if r["processing_fps"] > 0]
            ),
        },
        "evaluation_task": "ACTIVE vs FATIGUE (video level, existing pipeline)",
        "metrics_binary_primary": primary,
        "metrics_binary_alt_low_vigilant_active": alt_low_vigilant_metrics(records),
        "per_source_class_prediction_breakdown": per_source_class_breakdown(records),
        "temporal_signals": temporal_signal_summary(records),
        "false_positives": diagnostic_entries("FALSE_POSITIVE", false_positives),
        "false_negatives": diagnostic_entries("FALSE_NEGATIVE", false_negatives),
        "skipped_videos": diagnostic_entries("SKIPPED", skipped),
        "config": config,
        "production_logic_changed": False,
        "notes": [
            "Video-level rule reused unchanged from scripts/evaluate_uta_rldd.py: "
            "majority vote over per-frame ACTIVE/FATIGUE labels; tie -> ACTIVE if mean "
            "alertness >= 50 else FATIGUE; INSUFFICIENT_DATA below a 0.15 face-detection rate.",
            "Per-frame labels come from the existing AttentionEngine alert_severity via the "
            "existing SYSTEM_MAPPING (NORMAL/SAFE -> ACTIVE; LOW_FATIGUE/ALERT/WARNING/DROWSY -> FATIGUE).",
            "Primary binary task: alert -> ACTIVE, low vigilant + drowsy -> FATIGUE. The "
            "reduction of the official 3 classes to 2 is a task definition, not a tuned threshold.",
            "No threshold, weight or rule was modified for this dataset; no training was performed "
            "and the test data was never used to change the system.",
            "Each video is processed as an independent continuous session: trackers/engines are "
            "created fresh per video and keep their state across the frames of that video.",
            "Neutral default context (no CARLA, no geo/weather, day time context) so the driver-state "
            "evaluation is reproducible; see config.neutral_time_context / use_real_time_context.",
            "Confusion matrix counts only label-resolved, sufficiently-observed videos; "
            "INSUFFICIENT_DATA and UNKNOWN-label videos are reported separately, never as a class.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_confusion_png(png_path, primary, "UTA-RLDD (videos) - existing pipeline")
    return csv_path, summary_path, png_path, errors_path


def run_sanity(
    items: list[VideoItem],
    sample_fps: float | None,
    use_real_time_context: bool,
) -> int:
    """Process a few videos and verify the temporal behaviour (no accuracy)."""
    print("=" * 70)
    print("SANITY CHECK - temporal behaviour only (NO accuracy is reported)")
    print("=" * 70)
    print(f"Videos selected: {len(items)}")
    print(f"Frame sampling : {'every frame' if not sample_fps else f'{sample_fps} fps'}")
    records: list[dict] = []
    for index, item in enumerate(items, start=1):
        try:
            record = evaluate_video(
                item, sample_fps=sample_fps, use_real_time_context=use_real_time_context
            )
        except Exception as exc:  # pragma: no cover - reported, never hidden
            print(f"[{index}/{len(items)}] {item.relative_path}: FAILED ({type(exc).__name__}: {exc})")
            continue
        records.append(record)
        checks = record["temporal_checks"]
        flags = (
            ("multiple frames analysed", checks["multiple_frames_processed"]),
            ("timestamps increase", checks["timestamps_strictly_increasing"]),
            ("tracker state persists across frames", checks["tracker_state_persisted_within_video"]),
            ("tracker counters start at zero (fresh per video)", checks["tracker_counts_zero_at_video_start"]),
        )
        print(f"\n[{index}/{len(items)}] {item.relative_path}  (label={item.ground_truth}/{item.source_class})")
        print(
            f"    frames read/analysed : {record['frames_read']}/{record['processed_frames']} "
            f"(decode-skipped {record['skipped_frames']})"
        )
        print(
            f"    valid face frames    : {record['valid_face_frames']} "
            f"({record['face_detection_rate'] * 100.0:.1f}%) | decision frames {record['decision_frames']}"
        )
        print(
            f"    duration / fps       : {record['duration_s']:.1f}s @ {record['fps']:.2f} fps "
            f"(assumed={record['fps_assumed']})"
        )
        print(f"    blinks / yawns       : {record['blink_count']} / {record['yawn_count']}")
        print(f"    max eye closure      : {record['max_eye_closure_s']:.2f}s")
        print(f"    max fatigue risk     : {record['max_fatigue_risk']}")
        print(f"    final severity/level : {record['final_severity']} / {record['final_attention_level']}")
        for label, passed in flags:
            print(f"    {label:45s}: {'PASS' if passed else 'FAIL'}")
        print(f"    tracker object ids   : {checks['tracker_object_ids']}")

    if not records:
        print("\nSANITY CHECK: FAIL - no video could be processed.")
        return 2

    summary = {
        "videos_processed": len(records),
        "multiple_frames_processed": all(
            r["temporal_checks"]["multiple_frames_processed"] for r in records
        ),
        "timestamps_strictly_increasing": all(
            r["temporal_checks"]["timestamps_strictly_increasing"] for r in records
        ),
        "tracker_state_persisted_within_video": all(
            r["temporal_checks"]["tracker_state_persisted_within_video"] for r in records
        ),
        "tracker_counts_zero_at_video_start": all(
            r["temporal_checks"]["tracker_counts_zero_at_video_start"] for r in records
        ),
    }
    print("\n" + "-" * 70)
    print("SANITY SUMMARY (temporal behaviour only - no accuracy by design)")
    print(f"  videos processed                     : {summary['videos_processed']}")
    for key in (
        "multiple_frames_processed",
        "timestamps_strictly_increasing",
        "tracker_state_persisted_within_video",
        "tracker_counts_zero_at_video_start",
    ):
        print(f"  {key:37s}: {'PASS' if summary[key] else 'FAIL'}")
    print(
        "  note: every video builds its own tracker objects, so a fresh (zero) "
        "cumulative count at the first frame of each video is the reset evidence."
    )
    print("-" * 70)
    return 0 if all(summary[key] for key in summary if key != "videos_processed") else 2


def print_final_report(records: list[dict], discovery: dict, paths: dict, config: dict) -> None:
    primary = metrics_for(records)
    alt = alt_low_vigilant_metrics(records)
    signals = temporal_signal_summary(records)
    false_positives, false_negatives, skipped = error_analysis(records)
    matrix = primary["confusion_matrix"]["matrix"]
    per_class = primary["per_class"]

    print("\n" + "=" * 70)
    print("UTA-RLDD (VIDEO) EVALUATION - FINAL REPORT")
    print("=" * 70)
    print("DATASET")
    print(f"  location             : {discovery['dataset_root']}")
    print(f"  videos on disk       : {discovery['total_video_files']}")
    print(f"  ACTIVE (ground truth): {discovery['binary_label_counts'].get('ACTIVE', 0)}")
    print(f"  FATIGUE (ground tr.) : {discovery['binary_label_counts'].get('FATIGUE', 0)}")
    print(f"  UNKNOWN labels       : {discovery['binary_label_counts'].get('UNKNOWN', 0)}")
    print(f"  folds discovered     : {sorted(discovery['fold_structure'].keys())}")
    print(f"  parts discovered     : {sorted(discovery['part_structure'].keys())}")
    print(f"  official split info  : {discovery['official_reference']}")

    evaluated = [r for r in records if r["evaluated"]]
    print("\nEVALUATION")
    print(f"  videos processed     : {len(records)}")
    print(f"  videos scored        : {primary['videos_counted']}")
    print(f"  videos skipped       : {len(skipped)}")
    print(f"  frames processed     : {signals['total_processed_frames']}")
    print(f"  valid face frames    : {signals['total_valid_face_frames']}")
    mean_duration = safe_mean([r["duration_s"] for r in evaluated])
    if mean_duration:
        print(f"  mean video duration  : {mean_duration:.1f}s")
    mean_rate = signals["mean_face_detection_rate"]
    if mean_rate:
        print(f"  mean face det. rate  : {mean_rate * 100.0:.1f}%")
    print(f"  sampling             : {config['frame_sampling']}")
    print(f"  aggregation rule     : {config['aggregation_rule']}")

    print("\nRESULTS (ACTIVE vs FATIGUE)")
    print(f"  accuracy             : {primary['accuracy'] * 100.0:.2f}%")
    print(f"  precision (macro)    : {primary['macro_precision'] * 100.0:.2f}%")
    print(f"  recall (macro)       : {primary['macro_recall'] * 100.0:.2f}%")
    print(f"  F1 (macro)           : {primary['macro_f1'] * 100.0:.2f}%")
    print(f"  correctly classified : {primary['correctly_classified']}")
    print(f"  incorrectly classif. : {primary['incorrectly_classified']}")
    print(f"  skipped / unresolved : {primary['videos_excluded_from_metrics']}")
    print(
        "  alt reduction (alert+low vigilant = ACTIVE) accuracy: "
        f"{alt['accuracy'] * 100.0:.2f}%"
    )

    print("\nPER-CLASS")
    for name in BINARY_CLASSES:
        values = per_class[name]
        print(
            f"  {name:7s} precision={values['precision'] * 100.0:6.2f}% "
            f"recall={values['recall'] * 100.0:6.2f}% f1={values['f1'] * 100.0:6.2f}%"
        )

    print("\nCONFUSION MATRIX (rows = actual, columns = predicted)")
    print("                 Pred ACTIVE   Pred FATIGUE")
    print(f"  Actual ACTIVE  {matrix[0][0]:11d}   {matrix[0][1]:12d}")
    print(f"  Actual FATIGUE {matrix[1][0]:11d}   {matrix[1][1]:12d}")
    print(f"  (INSUFFICIENT_DATA / unresolved reported separately: {len(skipped)})")

    print("\nTEMPORAL SIGNALS (diagnostic only - no threshold was tuned)")
    print(f"  videos with blinks                 : {signals['videos_with_blinks']}")
    print(f"  videos with yawns                  : {signals['videos_with_yawns']}")
    print(f"  videos with prolonged eye closure  : {signals['videos_with_prolonged_eye_closure']}")
    print(f"  videos with distraction            : {signals['videos_with_distraction']}")
    print(f"  videos reaching NORMAL             : {signals['videos_reaching_normal_severity']}")
    print(f"  videos reaching LOW_FATIGUE        : {signals['videos_reaching_low_fatigue']}")
    print(f"  videos reaching ALERT              : {signals['videos_reaching_alert']}")
    print(f"  videos reaching WARNING            : {signals['videos_reaching_warning']}")
    print(f"  videos reaching DROWSY             : {signals['videos_reaching_drowsy']}")
    print(
        "  attention levels SAFE/MODERATE/HIGH/CRITICAL: "
        f"{signals['videos_reaching_safe_level']}/"
        f"{signals['videos_reaching_moderate_level']}/"
        f"{signals['videos_reaching_high_level']}/"
        f"{signals['videos_reaching_critical_level']}"
    )
    print(f"  total blinks / yawns               : {signals['total_blinks']} / {signals['total_yawns']}")
    print(f"  total alert episodes               : {signals['total_alert_episodes']}")
    print(f"  total alert-audio triggers         : {signals['total_audio_alert_triggers']}")

    print("\nERRORS")
    print(f"  false positives (ACTIVE -> FATIGUE): {len(false_positives)}")
    print(f"  false negatives (FATIGUE -> ACTIVE): {len(false_negatives)}")
    for kind, group in (("FP", false_positives), ("FN", false_negatives)):
        for record in group[:15]:
            print(
                f"    {kind}: {record['relative_path']} | dur={record['duration_s']:.0f}s "
                f"max_fatigue={record['max_fatigue_risk']} "
                f"max_distraction={record['max_distraction_risk']} "
                f"final_severity={record['final_severity']} "
                f"final_level={record['final_attention_level']} "
                f"blinks={record['blink_count']} yawns={record['yawn_count']}"
            )
    print(f"  skipped / unresolved: {len(skipped)}")
    for reason, count in sorted(Counter(r["skip_reason"] for r in skipped).items()):
        print(f"    - {reason}: {count}")

    print("\nFILES")
    for label, path in paths.items():
        print(f"  {label:18s}: {path}")
    print("=" * 70)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the EXISTING driver-attention pipeline on the VIDEO UTA-RLDD "
            "dataset (frame-by-frame temporal processing)."
        )
    )
    parser.add_argument("--dataset-root", default=None, help="Override the dataset root.")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Output directory (default: results/uta_rldd_videos).",
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only inspect the dataset and its labels; process no frames.",
    )
    parser.add_argument(
        "--sanity",
        type=int,
        default=0,
        help="Process N videos and verify the temporal behaviour (no accuracy).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of videos per source class (0 = all).",
    )
    parser.add_argument("--max-videos", type=int, default=0, help="Hard cap on videos (0 = all).")
    parser.add_argument(
        "--folds",
        default="",
        help="Comma-separated fold/part selectors, e.g. Fold1_part1,Fold2_part2.",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=None,
        help=(
            "Optional analysis sampling rate. Every frame is still decoded in order and all "
            "timestamps come from the container FPS; the default analyses every frame."
        ),
    )
    parser.add_argument(
        "--use-real-time-context",
        action="store_true",
        help=(
            "Use the wall-clock day/night time context instead of the deterministic neutral "
            "day context (default)."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    folds = [item.strip() for item in str(args.folds).split(",") if item.strip()]

    root = resolve_dataset_root(args.dataset_root)
    if root is None:
        print("Dataset not found. Looked in:")
        for candidate in DATASET_CANDIDATES:
            print(f"  - {candidate}")
        print("Use --dataset-root to point at the 'UTA-RLDD(videos)' folder.")
        return 1

    items, discovery = build_dataset_items(root)
    print_discovery_report(root, discovery)
    if not items:
        print("No video files were found; nothing to evaluate.")
        return 1
    if args.discover_only:
        print("\nDiscovery-only run: no frames were processed and no accuracy was computed.")
        return 0

    if args.sanity and args.sanity > 0:
        classes_present = sorted({i.source_class for i in items if i.ground_truth != "UNKNOWN"})
        group_count = max(1, len(classes_present))
        per_class = max(1, (args.sanity + group_count - 1) // group_count)
        sanity_items = select_items(items, folds, limit_per_class=per_class, max_videos=args.sanity)
        return run_sanity(sanity_items, args.sample_fps, args.use_real_time_context)

    selected = select_items(items, folds, limit_per_class=args.limit, max_videos=args.max_videos)
    if not selected:
        print("No videos selected (check --folds/--limit/--max-videos).")
        return 1

    sampling_text = (
        "every decoded frame (container FPS)"
        if not args.sample_fps
        else f"{args.sample_fps} fps analysis sampling (all frames decoded in order)"
    )
    print(f"\nProcessing {len(selected)} video(s) frame-by-frame with the existing pipeline...")
    print(f"Frame sampling: {sampling_text}")
    print(f"Time context  : {'wall clock' if args.use_real_time_context else 'neutral day'}")

    started = time.perf_counter()
    records: list[dict] = []
    for index, item in enumerate(selected, start=1):
        try:
            record = evaluate_video(
                item,
                sample_fps=args.sample_fps,
                use_real_time_context=args.use_real_time_context,
                progress_index=index,
                total_videos=len(selected),
            )
        except Exception as exc:  # corrupted file / decoder error: report, never hide
            record = build_result_record(
                item,
                evaluated=False,
                skip_reason=f"evaluation_error:{type(exc).__name__}",
                error=str(exc),
            )
            print(f"[{index}/{len(selected)}] {item.relative_path}: FAILED ({type(exc).__name__}: {exc})")
        records.append(record)
    elapsed = time.perf_counter() - started

    results_dir = (
        Path(args.results_dir).expanduser().resolve() if args.results_dir else DEFAULT_RESULTS_DIR
    )
    config = {
        "dataset_root": str(root),
        "folds": folds,
        "frame_sampling": sampling_text,
        "sample_fps": args.sample_fps,
        "neutral_time_context": not args.use_real_time_context,
        "use_real_time_context": bool(args.use_real_time_context),
        "neutral_time_of_day": None if args.use_real_time_context else NEUTRAL_TIME_OF_DAY,
        "limit_per_source_class": args.limit,
        "max_videos": args.max_videos,
        "aggregation_rule": (
            "reused from scripts/evaluate_uta_rldd.py choose_video_prediction(): majority vote "
            "over per-frame ACTIVE/FATIGUE labels; exact tie -> ACTIVE when mean alertness >= 50 "
            "else FATIGUE; INSUFFICIENT_DATA when valid frames == 0 or face detection rate < "
            f"{MIN_FACE_DETECTION_RATE}"
        ),
        "system_mapping": dict(SYSTEM_MAPPING),
        "binary_collapse": dict(SOURCE_CLASS_TO_BINARY),
        "min_face_detection_rate": MIN_FACE_DETECTION_RATE,
        "face_loss_grace_s": FACE_LOSS_GRACE_S,
        "session_warmup_min_s": SESSION_WARMUP_MIN_S,
        "session_warmup_min_frames": SESSION_WARMUP_MIN_FRAMES,
        "assumed_fps_fallback": ASSUMED_FPS_FALLBACK,
        "carla_used": False,
        "location_weather_context_used": False,
        "audio_played": False,
        "dataset_modified": False,
        "production_modules_modified": False,
        "elapsed_wall_seconds": elapsed,
    }

    csv_path, summary_path, png_path, errors_path = write_outputs(
        records, discovery, results_dir, config
    )
    print_final_report(
        records,
        discovery,
        {
            "results CSV": csv_path,
            "summary JSON": summary_path,
            "confusion PNG": png_path,
            "errors CSV": errors_path,
        },
        config,
    )
    print(f"\nWall-clock time for {len(records)} video(s): {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

















