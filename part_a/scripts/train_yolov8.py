#!/usr/bin/env python3
"""Train YOLOv8 for Part-A legal/courtesy amount extraction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 detector for amount region extraction.")
    parser.add_argument("--data", type=Path, default=Path("yolo_dataset/data.yaml"))
    parser.add_argument("--model", default="yolov8s.pt", help="YOLOv8 model or local weights path.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Optional dataset fraction for quick smoke tests, e.g. 0.02.",
    )
    parser.add_argument("--device", default=None, help="Examples: cpu, 0, 0,1. Defaults to Ultralytics auto.")
    parser.add_argument("--project", default="runs/part_a_yolo")
    parser.add_argument("--name", default="extraction")
    return parser.parse_args()


def import_yolo():
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        message = (
            "Ultralytics is not installed. Install it in an environment supported by PyTorch, then rerun:\n"
            "  python -m pip install ultralytics\n"
            "If this machine's default Python is too new for PyTorch wheels, create a Python 3.10-3.12 "
            "environment first."
        )
        raise RuntimeError(message) from exc
    return YOLO


def main() -> int:
    args = parse_args()
    if not args.data.exists():
        print(f"ERROR: data.yaml not found: {args.data}", file=sys.stderr)
        print("Run: python scripts/prepare_yolo_dataset.py", file=sys.stderr)
        return 1

    try:
        YOLO = import_yolo()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "workers": args.workers,
        "patience": args.patience,
        "project": str(Path(args.project).resolve()),
        "name": args.name,
        "exist_ok": True,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    if args.fraction is not None:
        train_kwargs["fraction"] = args.fraction

    results = model.train(**train_kwargs)
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
