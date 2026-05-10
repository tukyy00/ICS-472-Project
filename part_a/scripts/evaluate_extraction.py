#!/usr/bin/env python3
"""Evaluate YOLOv8 amount extraction with IoU thresholds required by the project."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


CLASS_NAMES = {
    0: "legal_amount",
    1: "courtesy_amount",
}

DEFAULT_THRESHOLDS = (0.50, 0.75, 0.90)
IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Part-A YOLO extraction results.")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained YOLO weights, e.g. best.pt.")
    parser.add_argument("--data", type=Path, default=Path("yolo_dataset/data.yaml"))
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.70, help="NMS IoU threshold for YOLO prediction.")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLDS),
        help="IoU thresholds t used for Accuracy_t (e.g. --thresholds 0.5 0.75 0.9).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/part_a_eval"))
    return parser.parse_args()


def import_yolo():
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError("Ultralytics is not installed. Run: python -m pip install ultralytics") from exc
    return YOLO


def read_simple_data_yaml(data_yaml: Path) -> dict[str, str]:
    data = {}
    for raw_line in data_yaml.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip("'\"")
    return data


def resolve_split_dirs(data_yaml: Path, split: str) -> tuple[Path, Path]:
    data = read_simple_data_yaml(data_yaml)
    base = Path(data.get("path", data_yaml.parent)).expanduser()
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()
    split_value = data.get(split)
    if split_value is None:
        raise ValueError(f"{data_yaml} has no '{split}' entry")

    images_dir = Path(split_value)
    if not images_dir.is_absolute():
        images_dir = base / images_dir
    labels_dir = base / "labels" / split
    return images_dir.resolve(), labels_dir.resolve()


def yolo_to_xyxy(row: str, image_width: int, image_height: int) -> tuple[int, tuple[float, float, float, float]]:
    class_text, x_text, y_text, w_text, h_text = row.split()
    class_id = int(class_text)
    x_center = float(x_text) * image_width
    y_center = float(y_text) * image_height
    width = float(w_text) * image_width
    height = float(h_text) * image_height
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    return class_id, (x1, y1, x2, y2)


def read_ground_truth(label_path: Path, image_path: Path) -> dict[int, tuple[float, float, float, float]]:
    with Image.open(image_path) as image:
        width, height = image.size

    boxes = {}
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        class_id, box = yolo_to_xyxy(line, width, height)
        boxes[class_id] = box
    return boxes


def load_rgb_array(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        return np.asarray(image.convert("RGB"))


def box_iou(
    first: tuple[float, float, float, float] | None,
    second: tuple[float, float, float, float] | None,
) -> float:
    if first is None or second is None:
        return 0.0

    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def collect_images(images_dir: Path) -> list[Path]:
    return sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def best_predictions_by_class(result) -> dict[int, dict[str, object]]:
    best = {}
    boxes = result.boxes
    if boxes is None:
        return best

    for xyxy, cls, conf in zip(boxes.xyxy.cpu().tolist(), boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist()):
        class_id = int(cls)
        if class_id not in CLASS_NAMES:
            continue
        previous = best.get(class_id)
        if previous is None or conf > previous["conf"]:
            best[class_id] = {
                "conf": float(conf),
                "box": tuple(float(value) for value in xyxy),
            }
    return best


def evaluate(args: argparse.Namespace) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not args.data.exists():
        raise FileNotFoundError(f"data.yaml not found: {args.data}")
    if not args.model.exists():
        raise FileNotFoundError(f"model weights not found: {args.model}")
    thresholds = tuple(sorted(set(args.thresholds)))
    if not thresholds:
        raise ValueError("At least one IoU threshold is required.")
    if any(threshold < 0.0 or threshold > 1.0 for threshold in thresholds):
        raise ValueError("All IoU thresholds must be in [0, 1].")

    YOLO = import_yolo()
    model = YOLO(str(args.model))
    images_dir, labels_dir = resolve_split_dirs(args.data, args.split)
    image_paths = collect_images(images_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    rows = []
    ious_by_class = {class_id: [] for class_id in CLASS_NAMES}

    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        ground_truth = read_ground_truth(label_path, image_path)
        prediction = model.predict(
            source=load_rgb_array(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
            save=False,
        )[0]
        predictions = best_predictions_by_class(prediction)

        for class_id, class_name in CLASS_NAMES.items():
            pred_record = predictions.get(class_id, {})
            pred_box = pred_record.get("box")
            gt_box = ground_truth.get(class_id)
            iou_score = box_iou(pred_box, gt_box)
            ious_by_class[class_id].append(iou_score)
            rows.append(
                {
                    "check_file": image_path.name,
                    "class_id": class_id,
                    "class_name": class_name,
                    "iou": iou_score,
                    "pred_conf": pred_record.get("conf", ""),
                    "gt_box_xyxy": list(gt_box) if gt_box else "",
                    "pred_box_xyxy": list(pred_box) if pred_box else "",
                }
            )

    metrics = {}
    for class_id, class_name in CLASS_NAMES.items():
        class_ious = ious_by_class[class_id]
        total = len(class_ious)
        metrics[class_name] = {
            "samples": total,
            "mean_iou": sum(class_ious) / total if total else 0.0,
        }
        for threshold in thresholds:
            key = f"accuracy_iou_{int(threshold * 100)}"
            metrics[class_name][key] = (
                sum(1 for value in class_ious if value >= threshold) / total if total else 0.0
            )

    all_ious = [value for class_ious in ious_by_class.values() for value in class_ious]
    metrics["overall"] = {
        "samples": len(all_ious),
        "mean_iou": sum(all_ious) / len(all_ious) if all_ious else 0.0,
    }
    for threshold in thresholds:
        key = f"accuracy_iou_{int(threshold * 100)}"
        metrics["overall"][key] = (
            sum(1 for value in all_ious if value >= threshold) / len(all_ious) if all_ious else 0.0
        )

    return metrics, rows


def main() -> int:
    args = parse_args()
    try:
        metrics, rows = evaluate(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / f"{args.split}_metrics.json"
    rows_path = args.output_dir / f"{args.split}_per_image_iou.csv"

    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["check_file", "class_id", "class_name", "iou", "pred_conf", "gt_box_xyxy", "pred_box_xyxy"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved per-image IoU: {rows_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
