"""Generate YOLO-only courtesy amount crops and manifests."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# Keep Ultralytics settings/cache inside the project workspace. On this machine
# the default AppData settings path is locked, which breaks importing YOLO.
ULTRALYTICS_CONFIG_DIR = Path.cwd() / "arabic_check_pipeline" / "part_b_courtesy_ocr" / "outputs" / "ultralytics_config"
ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))

from ultralytics import YOLO

from .labels import load_labels_from_dir, load_labels_from_file, normalize_check_id


def find_images(source: Path) -> list[Path]:
    return sorted(
        path
        for path in source.rglob("*")
        if path.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    )


def load_labels(labels_dir: Path | None, labels_file: Path | None) -> dict[str, str]:
    if labels_file:
        return load_labels_from_file(labels_file)
    if labels_dir:
        return load_labels_from_dir(labels_dir)
    return {}


def iter_chunks(items: list[Path], chunk_size: int) -> list[list[Path]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def clamp(value: float, low: int, high: int) -> int:
    return max(low, min(high, int(round(value))))


def crop_with_padding(image_path: Path, xyxy: tuple[float, float, float, float], padding: float) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    x1, y1, x2, y2 = xyxy
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    pad_x = box_w * padding
    pad_y = box_h * padding
    left = clamp(x1 - pad_x, 0, width - 1)
    top = clamp(y1 - pad_y, 0, height - 1)
    right = clamp(x2 + pad_x, left + 1, width)
    bottom = clamp(y2 + pad_y, top + 1, height)
    return image.crop((left, top, right, bottom))


def select_candidate(
    candidates: list[tuple[float, tuple[float, float, float, float]]],
    strategy: str,
) -> tuple[float, tuple[float, float, float, float]] | None:
    if not candidates:
        return None
    if strategy == "confidence":
        return max(candidates, key=lambda item: item[0])
    if strategy == "rightmost":
        return max(candidates, key=lambda item: ((item[1][0] + item[1][2]) / 2.0, item[0]))
    if strategy == "leftmost":
        return min(candidates, key=lambda item: ((item[1][0] + item[1][2]) / 2.0, -item[0]))
    raise ValueError(f"Unsupported selection strategy: {strategy}")


def generate(args: argparse.Namespace) -> None:
    source = Path(args.source)
    output_dir = Path(args.output_dir)
    crops_dir = output_dir / "crops" / args.split
    crops_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(
        Path(args.labels_dir) if args.labels_dir else None,
        Path(args.labels_file) if args.labels_file else None,
    )
    images = find_images(source)
    if args.limit is not None:
        images = images[: args.limit]
    model = YOLO(args.weights)

    manifest_path = output_dir / f"{args.split}_manifest.csv"
    misses_path = output_dir / f"{args.split}_misses.csv"
    rows: list[dict[str, object]] = []
    misses: list[dict[str, object]] = []

    with tqdm(total=len(images), desc=f"YOLO crops ({args.split})") as progress:
        for chunk in iter_chunks(images, args.batch_size):
            predictions = model.predict(
                source=[str(path) for path in chunk],
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                stream=True,
                verbose=False,
            )
            for image_path, result in zip(chunk, predictions):
                check_id = normalize_check_id(image_path.name)
                candidates: list[tuple[float, tuple[float, float, float, float]]] = []
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i].item())
                        if cls_id != args.class_id:
                            continue
                        conf = float(boxes.conf[i].item())
                        xyxy = tuple(float(v) for v in boxes.xyxy[i].tolist())
                        candidates.append((conf, xyxy))
                selected = select_candidate(candidates, args.selection)

                label = labels.get(check_id, "")
                if selected is None:
                    row = {
                        "check_id": check_id,
                        "image_path": str(image_path),
                        "crop_path": "",
                        "label": label,
                        "confidence": "",
                        "x1": "",
                        "y1": "",
                        "x2": "",
                        "y2": "",
                        "status": "missed",
                    }
                    rows.append(row)
                    misses.append(row)
                    progress.update(1)
                    continue

                confidence, xyxy = selected
                crop = crop_with_padding(image_path, xyxy, args.padding)
                crop_path = crops_dir / f"{check_id}.png"
                crop.save(crop_path)
                if crop.size[0] <= 0 or crop.size[1] <= 0:
                    status = "empty"
                    misses.append({"check_id": check_id, "image_path": str(image_path), "status": status})
                else:
                    status = "ok"

                x1, y1, x2, y2 = xyxy
                rows.append(
                    {
                        "check_id": check_id,
                        "image_path": str(image_path),
                        "crop_path": str(crop_path),
                        "label": label,
                        "confidence": f"{confidence:.6f}",
                        "x1": f"{x1:.2f}",
                        "y1": f"{y1:.2f}",
                        "x2": f"{x2:.2f}",
                        "y2": f"{y2:.2f}",
                        "status": status,
                    }
                )
                progress.update(1)

    fieldnames = ["check_id", "image_path", "crop_path", "label", "confidence", "x1", "y1", "x2", "y2", "status"]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with misses_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(misses)
    print(f"Wrote {manifest_path}")
    print(f"Wrote {misses_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Image directory, searched recursively.")
    parser.add_argument("--weights", default="best.pt", help="Fine-tuned YOLOv8 weights.")
    parser.add_argument("--split", required=True, choices=["train", "test", "val"], help="Manifest/crop split name.")
    parser.add_argument("--output-dir", default="arabic_check_pipeline/part_b_courtesy_ocr/outputs", help="Output root.")
    parser.add_argument("--labels-dir", default=None, help="Directory of tokenized label txt files.")
    parser.add_argument("--labels-file", default=None, help="Single tokenized label txt file.")
    parser.add_argument("--class-id", type=int, default=1, help="YOLO class id for courtesy amount.")
    parser.add_argument("--conf", type=float, default=0.10, help="YOLO confidence threshold.")
    parser.add_argument(
        "--selection",
        choices=["rightmost", "confidence", "leftmost"],
        default="rightmost",
        help="How to choose among multiple detected courtesy candidates.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="YOLO device, e.g. 0 or cpu. None lets Ultralytics decide.")
    parser.add_argument("--padding", type=float, default=0.04, help="Relative padding around detected box.")
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N image limit for smoke tests.")
    parser.add_argument("--batch-size", type=int, default=4, help="YOLO prediction chunk size.")
    return parser


def main() -> None:
    generate(build_parser().parse_args())


if __name__ == "__main__":
    main()
