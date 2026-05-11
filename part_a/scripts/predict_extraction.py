#!/usr/bin/env python3
"""Run a trained YOLOv8 detector and export courtesy/legal amount boxes."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image


CLASS_NAMES = {
    0: "legal_amount",
    1: "courtesy_amount",
}

IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict courtesy and legal amount boxes from full check images.")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained YOLO weights, e.g. best.pt.")
    parser.add_argument("--source", type=Path, default=Path("CheckImages"), help="Image file or directory.")
    parser.add_argument("--output", type=Path, default=Path("outputs/part_a_extractions.csv"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--save-crops", action="store_true", help="Save cropped courtesy/legal patches.")
    parser.add_argument("--crops-dir", type=Path, default=Path("outputs/part_a_crops"))
    return parser.parse_args()


def import_yolo():
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError("Ultralytics is not installed. Run: python -m pip install ultralytics") from exc
    return YOLO


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    raise FileNotFoundError(f"source not found: {source}")


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
                "box": tuple(max(0, round(float(value))) for value in xyxy),
            }
    return best


def load_rgb_array(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        return np.asarray(image.convert("RGB"))


def crop_box(image_path: Path, box: tuple[int, int, int, int], output_path: Path) -> None:
    with Image.open(image_path) as image:
        width, height = image.size
        x1, y1, x2, y2 = box
        clipped = (
            max(0, min(width, x1)),
            max(0, min(height, y1)),
            max(0, min(width, x2)),
            max(0, min(height, y2)),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.crop(clipped).save(output_path)


def empty_box_fields(prefix: str) -> dict[str, object]:
    return {
        f"{prefix}_x1": "",
        f"{prefix}_y1": "",
        f"{prefix}_x2": "",
        f"{prefix}_y2": "",
        f"{prefix}_conf": "",
    }


def box_fields(prefix: str, prediction: dict[str, object] | None) -> dict[str, object]:
    if prediction is None:
        return empty_box_fields(prefix)
    x1, y1, x2, y2 = prediction["box"]
    return {
        f"{prefix}_x1": x1,
        f"{prefix}_y1": y1,
        f"{prefix}_x2": x2,
        f"{prefix}_y2": y2,
        f"{prefix}_conf": prediction["conf"],
    }


def predict(args: argparse.Namespace) -> list[dict[str, object]]:
    if not args.model.exists():
        raise FileNotFoundError(f"model weights not found: {args.model}")

    YOLO = import_yolo()
    model = YOLO(str(args.model))
    rows = []
    for image_path in collect_images(args.source):
        result = model.predict(
            source=load_rgb_array(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
            save=False,
        )[0]
        predictions = best_predictions_by_class(result)
        courtesy = predictions.get(1)
        legal = predictions.get(0)
        row = {"check_file_name": image_path.name}
        row.update(box_fields("courtesy", courtesy))
        row.update(box_fields("legal", legal))
        rows.append(row)

        if args.save_crops:
            if courtesy is not None:
                crop_box(image_path, courtesy["box"], args.crops_dir / "courtesy_amount" / f"{image_path.stem}.png")
            if legal is not None:
                crop_box(image_path, legal["box"], args.crops_dir / "legal_amount" / f"{image_path.stem}.png")

    return rows


def main() -> int:
    args = parse_args()
    try:
        rows = predict(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "check_file_name",
        "courtesy_x1",
        "courtesy_y1",
        "courtesy_x2",
        "courtesy_y2",
        "courtesy_conf",
        "legal_x1",
        "legal_y1",
        "legal_x2",
        "legal_y2",
        "legal_conf",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} extraction rows to {args.output}")
    if args.save_crops:
        print(f"Saved crops under {args.crops_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
