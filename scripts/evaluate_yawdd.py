#!/usr/bin/env python3
"""Evaluate the existing driver-monitoring pipeline on the labeled YawDD videos.

YawDD's Mirror videos have condition labels in their filenames.  The current
pipeline can measure yawning, but it does not contain a talking/singing
classifier, so the primary video-level task is YAWNING vs NON_YAWNING:
Yawning and Talking&Yawning are positive; Normal and Talking are negative.
The unlabeled Dash videos are reported as excluded rather than assigned a
fabricated ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}
CONDITION_LABELS = ("NORMAL", "TALKING", "YAWNING", "TALKING_YAWNING")
YAWN_CONDITIONS = {"YAWNING", "TALKING_YAWNING"}
NON_YAWN_CONDITIONS = {"NORMAL", "TALKING"}


def safe_mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def classify_condition(path: Path) -> str | None:
    name = path.name.lower()
    if "talking&yawning" in name or "talking&yawnin" in name:
        return "TALKING_YAWNING"
    if "-yawning" in name:
        return "YAWNING"
    if "-talking" in name:
        return "TALKING"
    if "-normal" in name:
        return "NORMAL"
    return None


def mirror_videos(dataset_root: Path) -> tuple[list[tuple[str, Path]], list[Path]]:
    mirror_root = dataset_root / "Mirror"
    labeled: list[tuple[str, Path]] = []
    unlabeled: list[Path] = []
    for path in sorted(mirror_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        condition = classify_condition(path)
        if condition is None:
            unlabeled.append(path)
        else:
            labeled.append((condition, path))
    return labeled, unlabeled


def dash_videos(dataset_root: Path) -> list[Path]:
    dash_root = dataset_root / "Dash"
    return [
        path
        for path in sorted(dash_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]


def binary_metrics(records: list[dict]) -> dict:
    usable = [r for r in records if r["predicted_class"] != "INSUFFICIENT_DATA"]
    actual = [r["ground_truth"] for r in usable]
    predicted = [r["predicted_class"] for r in usable]
    labels = ["NON_YAWNING", "YAWNING"]
    matrix = [[0, 0], [0, 0]]
    for truth, guess in zip(actual, predicted):
        matrix[labels.index(truth)][labels.index(guess)] += 1
    tp = matrix[1][1]
    fp = matrix[0][1]
    fn = matrix[1][0]
    correct = sum(truth == guess for truth, guess in zip(actual, predicted))
    total = len(actual)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "labels": labels,
        "matrix": matrix,
        "usable_videos": total,
        "correct": correct,
        "incorrect": total - correct,
        "insufficient_data": len(records) - total,
        "accuracy": (correct / total * 100.0) if total else 0.0,
        "precision": precision * 100.0,
        "recall": recall * 100.0,
        "f1_score": f1 * 100.0,
    }


def evaluate_video(
    path: Path,
    condition: str,
    sample_fps: float | None = None,
) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    processed_frames = 0
    skipped_frames = 0
    valid_face_frames = 0
    face_missing_frames = 0
    yawn_frame_count = 0
    yawn_states: list[bool] = []
    yawn_counts: list[int] = []
    yawn_durations: list[float] = []
    alertness_values: list[float] = []
    fatigue_values: list[float] = []
    severity_counts: Counter[str] = Counter()
    video_start = time.perf_counter()

    landmarker = create_face_landmarker()
    eye_tracker = EyeClosureTracker()
    yawn_tracker = YawnTracker()
    fatigue_engine = FatigueEngine()
    attention_engine = AttentionEngine()
    distraction_monitor = DistractionMonitor()
    driving_tracker = DrivingDurationTracker(start_s=0.0)

    try:
        read_index = 0
        next_sample_s = 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            read_index += 1
            now_s = (read_index - 1) / fps if fps > 0 else float(processed_frames)
            if sample_fps and fps > 0:
                interval = 1.0 / sample_fps
                if now_s + 1e-9 < next_sample_s:
                    skipped_frames += 1
                    continue
                next_sample_s += interval
            processed_frames += 1
            height, width = frame.shape[:2]
            result = detect_landmarks(landmarker, frame, int(now_s * 1000.0))
            faces = result.face_landmarks
            if len(faces) != 1:
                face_missing_frames += 1
                continue

            valid_face_frames += 1
            face = faces[0]
            eyes = analyze_eyes(face, width, height)
            mouth = analyze_mouth(face, width, height)
            eye_state = eye_tracker.update(eyes.average_ear, now_s)
            mouth_state = yawn_tracker.update(mouth.mar, now_s)
            distraction = distraction_monitor.update(face, width, height, now_s)
            yawn_states.append(bool(mouth_state.yawn))
            yawn_counts.append(mouth_state.yawn_count)
            if mouth_state.yawn:
                yawn_frame_count += 1
            if mouth_state.yawn_duration_s is not None:
                yawn_durations.append(float(mouth_state.yawn_duration_s))

            head_down = bool(
                distraction.head
                and distraction.head.direction == "down"
                and distraction.head.looking_away
            )
            head_turned = bool(
                distraction.head
                and distraction.head.direction in {"left", "right"}
                and distraction.head.looking_away
            )
            fatigue_evidence = build_fatigue_evidence(
                eye=eye_state,
                mouth=mouth_state,
                time_ctx=get_time_context(),
                driving=driving_tracker.update(now_s),
                face_detected=True,
                head_down=head_down,
                head_turned=head_turned,
                eyes_not_visible=head_down or head_turned,
                mouth_not_visible=head_down or head_turned,
            )
            fatigue = fatigue_engine.update(fatigue_evidence, now_s)
            fatigue_risk = (
                fatigue.smoothed_risk
                if fatigue.smoothed_risk is not None
                else float(fatigue.fatigue_score or 0)
            )
            attention = attention_engine.update(
                fatigue_risk=fatigue_risk,
                distraction=distraction,
                now_s=now_s,
                face_ok=True,
            )
            alertness_values.append(float(attention.alertness_score))
            fatigue_values.append(float(fatigue_risk))
            severity_counts[attention.alert_severity] += 1
    finally:
        cap.release()
        landmarker.close()

    detected_yawns = max(yawn_counts, default=0)
    predicted = "YAWNING" if detected_yawns > 0 else "NON_YAWNING"
    # When frame sampling is enabled, compare face detections with the frames
    # actually evaluated rather than the full source-video frame count.
    face_rate = valid_face_frames / processed_frames if processed_frames else 0.0
    if valid_face_frames == 0 or face_rate < 0.15:
        predicted = "INSUFFICIENT_DATA"
    ground_truth = "YAWNING" if condition in YAWN_CONDITIONS else "NON_YAWNING"
    elapsed = max(time.perf_counter() - video_start, 1e-9)
    return {
        "video_name": path.name,
        "video_path": str(path),
        "source_condition": condition,
        "ground_truth": ground_truth,
        "predicted_class": predicted,
        "correct": predicted != "INSUFFICIENT_DATA" and predicted == ground_truth,
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "skipped_frames": skipped_frames,
        "valid_face_frames": valid_face_frames,
        "face_missing_frames": face_missing_frames,
        "face_detection_rate": face_rate,
        "yawn_count": detected_yawns,
        "yawn_frame_count": yawn_frame_count,
        "yawn_duration": max(yawn_durations, default=0.0),
        "final_alertness": alertness_values[-1] if alertness_values else None,
        "mean_alertness": safe_mean(alertness_values),
        "final_fatigue_risk": fatigue_values[-1] if fatigue_values else None,
        "final_severity": severity_counts.most_common(1)[0][0] if severity_counts else "UNKNOWN",
        "processing_fps": processed_frames / elapsed,
    }


def write_outputs(
    records: list[dict],
    errors: list[dict],
    dataset_root: Path,
    results_dir: Path,
) -> tuple[Path, Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "yawdd_results.csv"
    error_path = results_dir / "yawdd_errors.csv"
    summary_path = results_dir / "yawdd_summary.json"
    fields = [
        "video_name", "video_path", "source_condition", "ground_truth",
        "predicted_class", "correct", "total_frames", "processed_frames",
        "skipped_frames", "valid_face_frames", "face_missing_frames",
        "face_detection_rate", "yawn_count", "yawn_frame_count",
        "yawn_duration", "final_alertness", "mean_alertness",
        "final_fatigue_risk", "final_severity", "processing_fps",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in records)
    with error_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["video_name", "video_path", "error"])
        writer.writeheader()
        writer.writerows(errors)

    metrics = binary_metrics(records)
    condition_counts = Counter(row["source_condition"] for row in records)
    summary = {
        "dataset": "YawDD",
        "dataset_path": str(dataset_root),
        "evaluation_task": "YAWNING vs NON_YAWNING",
        "label_mapping": {
            "YAWNING": "Yawning and Talking&Yawning filename conditions",
            "NON_YAWNING": "Normal and Talking filename conditions",
        },
        "videos_evaluated": len(records),
        "videos_failed": len(errors),
        "source_condition_counts": dict(sorted(condition_counts.items())),
        "total_frames": sum(row["total_frames"] for row in records),
        "processed_frames": sum(row["processed_frames"] for row in records),
        "valid_frames": sum(row["valid_face_frames"] for row in records),
        "invalid_or_face_missing_frames": sum(row["face_missing_frames"] for row in records),
        "metrics": metrics,
        "dash_videos_excluded_from_video_metrics": len(dash_videos(dataset_root)),
        "unlabeled_mirror_videos_excluded": 0,
        "errors": errors,
        "production_logic_changed": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path, summary_path, error_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the existing pipeline on labeled YawDD Mirror videos.")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--limit", type=int, default=0, help="Limit each source condition to N videos.")
    parser.add_argument("--sample-fps", type=float, default=None)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else ROOT / "datasets" / "YawDD"
    labeled, unlabeled = mirror_videos(dataset_root)
    if args.limit > 0:
        selected: list[tuple[str, Path]] = []
        counts: Counter[str] = Counter()
        for condition, path in labeled:
            if counts[condition] < args.limit:
                selected.append((condition, path))
                counts[condition] += 1
        labeled = selected
    if not labeled:
        print(f"No labeled Mirror videos found under {dataset_root / 'Mirror'}")
        return 1

    print("YAWDD DATASET EVALUATION")
    print(f"Dataset: {dataset_root}")
    print("Task: YAWNING vs NON_YAWNING")
    print(f"Labeled Mirror videos: {len(labeled)}")
    print(f"Unlabeled Mirror videos skipped: {len(unlabeled)}")
    print(f"Dash videos excluded from video-level metrics: {len(dash_videos(dataset_root))}")

    records: list[dict] = []
    errors: list[dict] = []
    for index, (condition, path) in enumerate(labeled, start=1):
        try:
            row = evaluate_video(path, condition, sample_fps=args.sample_fps)
            records.append(row)
            status = "CORRECT" if row["correct"] else row["predicted_class"]
            print(f"[{index}/{len(labeled)}] {path.name}: {status}")
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append({"video_name": path.name, "video_path": str(path), "error": str(exc)})
            print(f"[{index}/{len(labeled)}] {path.name}: FAILED: {exc}")

    results_dir = args.results_dir.expanduser().resolve()
    csv_path, summary_path, error_path = write_outputs(
        records, errors, dataset_root, results_dir
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps(summary["metrics"], indent=2))
    print(f"Results CSV: {csv_path}")
    print(f"Summary JSON: {summary_path}")
    print(f"Errors CSV: {error_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
