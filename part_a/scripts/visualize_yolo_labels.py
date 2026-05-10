#!/usr/bin/env python3
"""Draw sample YOLO labels over check images to verify class mapping."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw


CLASS_STYLES = {
    0: ("legal_amount", (34, 139, 230)),
    1: ("courtesy_amount", (220, 53, 69)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create visual overlays for YOLO amount labels.")
    parser.add_argument("--image-dir", type=Path, default=Path("CheckImages"))
    parser.add_argument("--label-dir", type=Path, default=Path("BoundingBoxes"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/label_overlays"))
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def yolo_to_xyxy(row: str, image_width: int, image_height: int) -> tuple[int, tuple[float, float, float, float]]:
    class_text, x_text, y_text, w_text, h_text = row.split()
    class_id = int(class_text)
    x_center = float(x_text) * image_width
    y_center = float(y_text) * image_height
    width = float(w_text) * image_width
    height = float(h_text) * image_height
    return (
        class_id,
        (
            x_center - width / 2,
            y_center - height / 2,
            x_center + width / 2,
            y_center + height / 2,
        ),
    )


def draw_overlay(image_path: Path, label_path: Path, output_path: Path) -> None:
    with Image.open(image_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size

    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        class_id, box = yolo_to_xyxy(line, width, height)
        label, color = CLASS_STYLES.get(class_id, (f"class_{class_id}", (0, 0, 0)))
        x1, y1, x2, y2 = box
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text_box = draw.textbbox((x1, y1), label)
        draw.rectangle(text_box, fill=color)
        draw.text((x1, y1), label, fill=(255, 255, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> int:
    args = parse_args()
    if not args.image_dir.is_dir():
        print(f"ERROR: image directory not found: {args.image_dir}", file=sys.stderr)
        return 1
    if not args.label_dir.is_dir():
        print(f"ERROR: label directory not found: {args.label_dir}", file=sys.stderr)
        return 1

    stems = sorted(path.stem for path in args.label_dir.glob("*.txt") if (args.image_dir / f"{path.stem}.tif").exists())
    if not stems:
        print("ERROR: no matching image/label pairs found", file=sys.stderr)
        return 1

    selected = random.Random(args.seed).sample(stems, min(args.samples, len(stems)))
    for stem in selected:
        draw_overlay(
            args.image_dir / f"{stem}.tif",
            args.label_dir / f"{stem}.txt",
            args.output_dir / f"{stem}.png",
        )

    print(f"Saved {len(selected)} overlays to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
