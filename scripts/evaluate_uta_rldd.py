#!/usr/bin/env python3
"""Headless evaluation script for the UTA-RLDD dataset.

This file intentionally reuses the project's existing detection and attention
logic rather than reimplementing it. It does not modify the live webcam system.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None

DATASET_CANDIDATES = [
    ROOT / "datasets" / "UTA-RLDD" / "test",
    ROOT / "datasets" / "UTA-RLDD",
    ROOT / "datasets",
]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".jpg", ".jpeg", ".png"}
CLASS_NAMES = ("ACTIVE", "FATIGUE")

# Existing project alertness states are documented in the live attention engine:
# NORMAL / LOW_FATIGUE / ALERT / WARNING / DROWSY. We map those to the binary
# dataset labels used by UTA-RLDD without changing production thresholds.
SYSTEM_MAPPING = {
    "NORMAL": "ACTIVE",
    "SAFE": "ACTIVE",
    "LOW_FATIGUE": "FATIGUE",
    "ALERT": "FATIGUE",
    "WARNING": "FATIGUE",
    "DROWSY": "FATIGUE",
}


def pretty_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    return f"{seconds:.2f}s"


def list_supported_media_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            files.append(path)
    return files


def detect_dataset_root(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if p.exists():
            return p
        return p

    for candidate in DATASET_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def discover_test_videos(dataset_root: Path, limit: int = 0) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]]]:
    active_root = dataset_root / "active"
    fatigue_root = dataset_root / "fatigue"

    active_files = list_supported_media_files(active_root)
    fatigue_files = list_supported_media_files(fatigue_root)

    if limit > 0:
        active_files = active_files[:limit]
        fatigue_files = fatigue_files[:limit]

    active = [("ACTIVE", path) for path in active_files]
    fatigue = [("FATIGUE", path) for path in fatigue_files]
    return active, fatigue


def map_alertness_to_class(alert_severity: str | None, alertness_score: float | None = None) -> str:
    key = (alert_severity or "NORMAL").upper()
    return SYSTEM_MAPPING.get(key, "ACTIVE")


def safe_mean(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return float(np.mean(items))


def safe_median(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return float(np.median(items))


def safe_min(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return float(np.min(items))


def safe_max(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return float(np.max(items))


def compute_metrics(actual: list[str], predicted: list[str]) -> dict:
    labels = ["ACTIVE", "FATIGUE"]
    matrix = {true: {pred: 0 for pred in labels} for true in labels}
    for truth, pred in zip(actual, predicted):
        if truth in matrix and pred in matrix[truth]:
            matrix[truth][pred] += 1

    tp = matrix["ACTIVE"]["ACTIVE"]
    fp = matrix["FATIGUE"]["ACTIVE"]
    fn = matrix["ACTIVE"]["FATIGUE"]
    tn = matrix["FATIGUE"]["FATIGUE"]

    total = len(actual)
    accuracy = (tp + tn) / total if total else 0.0

    precision_active = tp / (tp + fp) if (tp + fp) else 0.0
    recall_active = tp / (tp + fn) if (tp + fn) else 0.0
    f1_active = 2 * precision_active * recall_active / (precision_active + recall_active) if (precision_active + recall_active) else 0.0

    precision_fatigue = tn / (tn + fn) if (tn + fn) else 0.0
    recall_fatigue = tn / (tn + fp) if (tn + fp) else 0.0
    f1_fatigue = 2 * precision_fatigue * recall_fatigue / (precision_fatigue + recall_fatigue) if (precision_fatigue + recall_fatigue) else 0.0

    confusion_matrix = [
        [matrix["ACTIVE"]["ACTIVE"], matrix["ACTIVE"]["FATIGUE"]],
        [matrix["FATIGUE"]["ACTIVE"], matrix["FATIGUE"]["FATIGUE"]],
    ]

    return {
        "accuracy": accuracy,
        "precision": precision_active,
        "recall": recall_active,
        "f1_score": f1_active,
        "per_class": {
            "ACTIVE": {"precision": precision_active, "recall": recall_active, "f1": f1_active},
            "FATIGUE": {"precision": precision_fatigue, "recall": recall_fatigue, "f1": f1_fatigue},
        },
        "confusion_matrix": confusion_matrix,
        "matrix_counts": matrix,
    }


def write_confusion_matrix_png(path: Path, labels: list[str], matrix: list[list[int]]) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.3))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title("UTA-RLDD Confusion Matrix")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color="black", fontsize=11)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def choose_video_prediction(frame_labels: list[str], mean_alertness: float | None) -> str:
    if not frame_labels:
        return "INSUFFICIENT_DATA"
    active_count = frame_labels.count("ACTIVE")
    fatigue_count = frame_labels.count("FATIGUE")
    if active_count == fatigue_count:
        if mean_alertness is None:
            return "ACTIVE"
        return "ACTIVE" if mean_alertness >= 50.0 else "FATIGUE"
    return "ACTIVE" if active_count > fatigue_count else "FATIGUE"


def evaluate_video(
    media_path: Path,
    ground_truth: str,
    sample_fps: float | None = None,
    progress_index: int | None = None,
    total_videos: int = 0,
) -> dict:
    cap = cv2.VideoCapture(str(media_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {media_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0 and media_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        total_frames = 1

    video_start = time.perf_counter()
    frame_skip_count = 0
    processed_frames = 0
    valid_face_frames = 0
    no_face_frames = 0
    frame_labels: list[str] = []
    frame_severities: list[str] = []
    frame_levels: list[str] = []
    alertness_values: list[float] = []
    fatigue_values: list[float] = []
    distraction_values: list[float] = []
    raw_combined_values: list[float] = []
    eye_states: Counter[str] = Counter()
    head_dirs: Counter[str] = Counter()
    blink_total = 0
    yawn_total = 0

    landmarker = create_face_landmarker()
    eye_tracker = EyeClosureTracker()
    yawn_tracker = YawnTracker()
    fatigue_engine = FatigueEngine()
    attention_engine = AttentionEngine()
    distraction_monitor = DistractionMonitor()
    driving_tracker = DrivingDurationTracker(start_s=0.0)

    try:
        read_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            read_index += 1
            if sample_fps is not None and sample_fps > 0.0:
                if fps <= 0.0:
                    process_frame = True
                else:
                    interval = 1.0 / sample_fps
                    timestamp_sec = float(read_index - 1) / fps if fps > 0.0 else float(read_index - 1)
                    process_frame = (timestamp_sec % interval) < (1.0 / (fps * 100.0)) or read_index == 1
                    # Sample the video near the requested FPS while keeping the first frame.
            else:
                process_frame = True

            if not process_frame:
                frame_skip_count += 1
                continue

            processed_frames += 1
            h, w = frame.shape[:2]
            now_s = float(read_index - 1) / fps if fps > 0.0 else float(processed_frames - 1)
            timestamp_ms = int(now_s * 1000.0)
            result = detect_landmarks(landmarker, frame, timestamp_ms)
            faces = result.face_landmarks
            if not faces:
                no_face_frames += 1
                continue

            if len(faces) != 1:
                no_face_frames += 1
                continue

            valid_face_frames += 1
            face = faces[0]
            eyes = analyze_eyes(face, w, h)
            mouth = analyze_mouth(face, w, h)
            eye_state = eye_tracker.update(eyes.average_ear, now_s)
            mouth_state = yawn_tracker.update(mouth.mar, now_s)
            distraction = distraction_monitor.update(face, w, h, now_s)

            if eye_state is not None:
                eye_states[eye_state.eye] += 1
            if distraction and distraction.head is not None:
                head_dirs[distraction.head.direction] += 1

            if mouth_state is not None:
                # Trackers hold CUMULATIVE totals: the video total is the
                # latest count, not the sum of per-frame cumulative values.
                blink_total = int(eye_state.blink_count if eye_state else 0)
                yawn_total = int(mouth_state.yawn_count if mouth_state else 0)

            time_ctx = get_time_context()
            driving = driving_tracker.update(now_s)
            head_down = bool(distraction and distraction.head and distraction.head.direction == "down" and distraction.head.looking_away)
            head_turned = bool(distraction and distraction.head and distraction.head.direction in {"left", "right"} and distraction.head.looking_away)
            fatigue_evidence = build_fatigue_evidence(
                eye=eye_state,
                mouth=mouth_state,
                time_ctx=time_ctx,
                driving=driving,
                face_detected=True,
                head_down=head_down,
                head_turned=head_turned,
                eyes_not_visible=head_down or head_turned,
                mouth_not_visible=head_down or head_turned,
            )
            fatigue_decision = fatigue_engine.update(fatigue_evidence, now_s)
            fatigue_risk = (
                fatigue_decision.smoothed_risk
                if fatigue_decision is not None and fatigue_decision.smoothed_risk is not None
                else float(fatigue_decision.fatigue_score or 0)
            )
            attention = attention_engine.update(
                fatigue_risk=fatigue_risk,
                distraction=distraction,
                now_s=now_s,
                face_ok=True,
            )
            severity = attention.alert_severity
            class_label = map_alertness_to_class(severity, attention.alertness_score)
            frame_labels.append(class_label)
            frame_severities.append(severity)
            frame_levels.append(attention.level)
            alertness_values.append(float(attention.alertness_score))
            fatigue_values.append(float(fatigue_risk))
            distraction_values.append(float(distraction.distraction_risk) if distraction else 0.0)
            raw_combined_values.append(float(attention.raw_combined))

    finally:
        cap.release()
        landmarker.close()

    processing_elapsed = max(1e-9, time.perf_counter() - video_start)
    total_valid = valid_face_frames
    face_detection_rate = (valid_face_frames / total_frames) if total_frames else 0.0
    mean_alertness = safe_mean(alertness_values)
    median_alertness = safe_median(alertness_values)
    min_alertness = safe_min(alertness_values)
    max_alertness = safe_max(alertness_values)
    active_percentage = (frame_labels.count("ACTIVE") / len(frame_labels) * 100.0) if frame_labels else 0.0
    fatigue_percentage = (frame_labels.count("FATIGUE") / len(frame_labels) * 100.0) if frame_labels else 0.0

    if total_valid == 0 or face_detection_rate < 0.15:
        predicted_class = "INSUFFICIENT_DATA"
    else:
        predicted_class = choose_video_prediction(frame_labels, mean_alertness)

    result = {
        "video_name": media_path.name,
        "ground_truth": ground_truth,
        "predicted_class": predicted_class,
        "correct": bool(predicted_class != "INSUFFICIENT_DATA" and predicted_class == ground_truth),
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "skipped_frames": frame_skip_count,
        "valid_face_frames": valid_face_frames,
        "no_face_frames": no_face_frames,
        "face_detection_rate": face_detection_rate,
        "mean_alertness": mean_alertness,
        "median_alertness": median_alertness,
        "min_alertness": min_alertness,
        "max_alertness": max_alertness,
        "active_percentage": active_percentage,
        "fatigue_percentage": fatigue_percentage,
        "processing_fps": (processed_frames / processing_elapsed) if processed_frames else 0.0,
        "mean_fatigue_risk": safe_mean(fatigue_values),
        "max_fatigue_risk": safe_max(fatigue_values),
        "mean_distraction_risk": safe_mean(distraction_values),
        "max_distraction_risk": safe_max(distraction_values),
        "max_attention_raw": safe_max(raw_combined_values),
        "final_alertness": alertness_values[-1] if alertness_values else None,
        "final_fatigue_risk": fatigue_values[-1] if fatigue_values else None,
        "final_severity": frame_severities[-1] if frame_severities else None,
        "final_level": frame_levels[-1] if frame_levels else None,
        "blink_count": blink_total,
        "yawn_count": yawn_total,
        "dominant_eye_state": eye_states.most_common(1)[0][0] if eye_states else "UNKNOWN",
        "dominant_head_direction": head_dirs.most_common(1)[0][0] if head_dirs else "UNKNOWN",
        "sample_fps": sample_fps,
        "frame_labels": frame_labels,
        "label_counts": {"ACTIVE": frame_labels.count("ACTIVE"), "FATIGUE": frame_labels.count("FATIGUE")},
    }

    if progress_index is not None and total_videos:
        status_text = "CORRECT" if result['correct'] else ("WRONG" if predicted_class != 'INSUFFICIENT_DATA' else "INSUFFICIENT DATA")
        print(
            f"[{progress_index}/{total_videos}] {media_path.name}\n"
            f"  Ground truth: {ground_truth}\n"
            f"  Prediction:   {predicted_class}\n"
            f"  Total frames: {total_frames}\n"
            f"  Processed: {processed_frames}\n"
            f"  Skipped: {frame_skip_count}\n"
            f"  FPS: {result['processing_fps']:.2f}\n"
            f"  Mean alertness: {mean_alertness if mean_alertness is not None else 'N/A'}\n"
            f"  Face rate: {face_detection_rate * 100.0:.1f}%\n"
            f"  {status_text}"
        )

    return result


def summarize_results(records: list[dict], dataset_root: Path) -> dict:
    actual = [record["ground_truth"] for record in records if record["predicted_class"] != "INSUFFICIENT_DATA"]
    predicted = [record["predicted_class"] for record in records if record["predicted_class"] != "INSUFFICIENT_DATA"]
    metrics = compute_metrics(actual, predicted)

    correct = sum(1 for record in records if record["correct"])
    incorrect = sum(1 for record in records if not record["correct"] and record["predicted_class"] != "INSUFFICIENT_DATA")
    insufficient = sum(1 for record in records if record["predicted_class"] == "INSUFFICIENT_DATA")

    summary = {
        "dataset_path": str(dataset_root),
        "test_path": str(dataset_root),
        "total_videos": len(records),
        "active_videos": sum(1 for record in records if record["ground_truth"] == "ACTIVE"),
        "fatigue_videos": sum(1 for record in records if record["ground_truth"] == "FATIGUE"),
        "correctly_classified": correct,
        "incorrectly_classified": incorrect,
        "insufficient_data_videos": insufficient,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "confusion_matrix": {
            "labels": ["ACTIVE", "FATIGUE"],
            "matrix": metrics["confusion_matrix"],
        },
        "class_mapping": {
            "SYSTEM": SYSTEM_MAPPING,
            "dataset": {"ACTIVE": "ACTIVE", "FATIGUE": "FATIGUE"},
        },
        "per_class": metrics["per_class"],
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the UTA-RLDD test split against the existing attention pipeline.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Override dataset root (defaults to datasets/UTA-RLDD/test).")
    parser.add_argument("--video", type=str, default=None, help="Run a single video file instead of the full test set.")
    parser.add_argument("--limit", type=int, default=0, help="Limit each class to N videos (0 = all).")
    parser.add_argument("--sample-fps", type=float, default=None, help="Optional sample rate in fps for frame subsampling.")
    args = parser.parse_args()

    dataset_root = detect_dataset_root(args.dataset_root)
    if args.video:
        media_path = Path(args.video).expanduser().resolve()
        if not media_path.exists():
            print(f"Video not found: {media_path}")
            return 1
        ground_truth = "ACTIVE" if "active" in media_path.parts else "FATIGUE" if "fatigue" in media_path.parts else "UNKNOWN"
        if ground_truth == "UNKNOWN":
            print("Unable to infer ground truth from path; use an active/fatigue folder.")
            return 1
        records = [evaluate_video(media_path, ground_truth, sample_fps=args.sample_fps, progress_index=1, total_videos=1)]
        print("\nSingle video evaluation complete.")
        print(json.dumps({"video": str(media_path), "ground_truth": ground_truth, "prediction": records[0]["predicted_class"], "mean_alertness": records[0]["mean_alertness"]}, indent=2))
        return 0

    if dataset_root is None:
        print("Dataset not found. Expected one of:")
        for candidate in DATASET_CANDIDATES:
            print(f"  - {candidate}")
        print("\nCreate the dataset at the default path or specify --dataset-root.")
        return 1

    active_files, fatigue_files = discover_test_videos(dataset_root, limit=args.limit)
    videos = active_files + fatigue_files
    if not videos:
        print(f"No videos found under {dataset_root / 'active'} or {dataset_root / 'fatigue'}.")
        return 1

    print("=" * 55)
    print("UTA-RLDD DATASET EVALUATION")
    print("=" * 55)
    print(f"Dataset: {dataset_root}")
    print("Classes:")
    for label in CLASS_NAMES:
        print(f"  - {label}")
    print(f"Found active videos: {len(active_files)}")
    print(f"Found fatigue videos: {len(fatigue_files)}")
    print(f"Total videos: {len(videos)}")
    print("\nDataset mapping:")
    print("  ACTIVE ground truth -> active")
    print("  FATIGUE ground truth -> fatigue")
    print("\nSystem mapping (existing project alert states):")
    for k, v in SYSTEM_MAPPING.items():
        print(f"  {k:12s} -> {v}")
    print("\nVideo aggregation policy:")
    print("  - majority vote across valid face frames")
    print("  - if valid frames are few / face detection rate < 15%, mark INSUFFICIENT_DATA")
    print("  - if tied, prefer ACTIVE when mean alertness >= 50, otherwise FATIGUE")
    print("  - no audio, no GUI, no live webcam")
    print("=" * 55)

    results: list[dict] = []
    for idx, (ground_truth, path) in enumerate(videos, start=1):
        result = evaluate_video(path, ground_truth, sample_fps=args.sample_fps, progress_index=idx, total_videos=len(videos))
        results.append(result)

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    csv_path = results_dir / "uta_rldd_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "video_name",
            "ground_truth",
            "predicted_class",
            "correct",
            "total_frames",
            "processed_frames",
            "skipped_frames",
            "valid_face_frames",
            "no_face_frames",
            "face_detection_rate",
            "mean_alertness",
            "median_alertness",
            "min_alertness",
            "max_alertness",
            "active_percentage",
            "fatigue_percentage",
            "processing_fps",
            "mean_fatigue_risk",
            "max_fatigue_risk",
            "mean_distraction_risk",
            "max_distraction_risk",
            "max_attention_raw",
            "final_alertness",
            "final_fatigue_risk",
            "final_severity",
            "final_level",
            "blink_count",
            "yawn_count",
            "dominant_eye_state",
            "dominant_head_direction",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({
                "video_name": row["video_name"],
                "ground_truth": row["ground_truth"],
                "predicted_class": row["predicted_class"],
                "correct": row["correct"],
                "total_frames": row["total_frames"],
                "processed_frames": row["processed_frames"],
                "skipped_frames": row.get("skipped_frames", 0),
                "valid_face_frames": row["valid_face_frames"],
                "no_face_frames": row["no_face_frames"],
                "face_detection_rate": row["face_detection_rate"],
                "mean_alertness": row["mean_alertness"],
                "median_alertness": row["median_alertness"],
                "min_alertness": row["min_alertness"],
                "max_alertness": row["max_alertness"],
                "active_percentage": row["active_percentage"],
                "fatigue_percentage": row["fatigue_percentage"],
                "processing_fps": row["processing_fps"],
                "mean_fatigue_risk": row["mean_fatigue_risk"],
                "max_fatigue_risk": row.get("max_fatigue_risk"),
                "mean_distraction_risk": row["mean_distraction_risk"],
                "max_distraction_risk": row.get("max_distraction_risk"),
                "max_attention_raw": row.get("max_attention_raw"),
                "final_alertness": row.get("final_alertness"),
                "final_fatigue_risk": row.get("final_fatigue_risk"),
                "final_severity": row.get("final_severity"),
                "final_level": row.get("final_level"),
                "blink_count": row["blink_count"],
                "yawn_count": row["yawn_count"],
                "dominant_eye_state": row["dominant_eye_state"],
                "dominant_head_direction": row["dominant_head_direction"],
            })

    errors = [record for record in results if record["predicted_class"] != "INSUFFICIENT_DATA" and record["correct"] is False]
    errors_path = results_dir / "uta_rldd_errors.csv"
    with errors_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["video_name", "ground_truth", "predicted_class", "mean_alertness", "face_detection_rate", "active_percentage", "fatigue_percentage"])
        writer.writeheader()
        for row in errors:
            writer.writerow({
                "video_name": row["video_name"],
                "ground_truth": row["ground_truth"],
                "predicted_class": row["predicted_class"],
                "mean_alertness": row["mean_alertness"],
                "face_detection_rate": row["face_detection_rate"],
                "active_percentage": row["active_percentage"],
                "fatigue_percentage": row["fatigue_percentage"],
            })

    summary = summarize_results(results, dataset_root)
    summary_path = results_dir / "uta_rldd_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    confusion_path = results_dir / "uta_rldd_confusion_matrix.png"
    labels = ["ACTIVE", "FATIGUE"]
    matrix = summary["confusion_matrix"]["matrix"]
    if labels and matrix and len(matrix) == 2:
        write_confusion_matrix_png(confusion_path, labels, matrix)

    actual = [record["ground_truth"] for record in results if record["predicted_class"] != "INSUFFICIENT_DATA"]
    predicted = [record["predicted_class"] for record in results if record["predicted_class"] != "INSUFFICIENT_DATA"]
    metrics = compute_metrics(actual, predicted)

    print("\n" + "=" * 55)
    print("FINAL RESULTS")
    print("=" * 55)
    print(f"Accuracy : {metrics['accuracy'] * 100.0:.2f}%")
    print(f"Precision: {metrics['precision'] * 100.0:.2f}%")
    print(f"Recall   : {metrics['recall'] * 100.0:.2f}%")
    print(f"F1 Score : {metrics['f1_score'] * 100.0:.2f}%")
    print(f"Correct  : {sum(1 for record in results if record['correct'])}")
    print(f"Wrong    : {sum(1 for record in results if not record['correct'] and record['predicted_class'] != 'INSUFFICIENT_DATA')}")
    print(f"Skipped  : {sum(1 for record in results if record['predicted_class'] == 'INSUFFICIENT_DATA')}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {summary_path}")
    print(f"PNG:  {confusion_path}")
    print("=" * 55)
    return 0


DatasetRoot = None
if __name__ == "__main__":
    # Keep a module-level dataset root for summary serialization; the CLI resolves
    # the actual path before evaluation.
    DatasetRoot = detect_dataset_root(None)
    raise SystemExit(main())
