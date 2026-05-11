#!/usr/bin/env python3
"""Prepare the Part-A YOLOv8 dataset for Arabic check amount extraction."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


CLASS_NAMES = {
    0: "legal_amount",
    1: "courtesy_amount",
}

IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class Sample:
    stem: str
    image_path: Path
    label_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a YOLOv8 train/val split from check images and BoundingBoxes labels."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Directory with full check images. Auto-detects CheckImages or ../archive/CheckImages.",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=Path("BoundingBoxes"),
        help="Directory with YOLO-format amount-region labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("yolo_dataset"),
        help="Output YOLO dataset directory.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio. The project reference uses 0.15.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split.")
    parser.add_argument(
        "--image-mode",
        choices=("symlink", "copy", "png"),
        default="png",
        help="How to place images in the YOLO dataset. Use png if TIFF loading fails.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing generated yolo_dataset directory.",
    )
    parser.add_argument(
        "--allow-nonstandard-labels",
        action="store_true",
        help="Allow labels that do not contain exactly one legal and one courtesy box.",
    )
    return parser.parse_args()


def resolve_dir(path: Path | None, candidates: list[Path], description: str) -> Path:
    if path is not None:
        resolved = path.expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"{description} not found: {resolved}")
        return resolved

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_dir():
            return resolved

    formatted = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not auto-detect {description}. Checked: {formatted}")


def read_label(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number} must have 5 fields, got {len(parts)}")

        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = (float(value) for value in parts[1:])
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_number} has a non-numeric YOLO value") from exc

        if class_id not in CLASS_NAMES:
            raise ValueError(f"{label_path}:{line_number} has unknown class id {class_id}")

        values = (x_center, y_center, width, height)
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError(f"{label_path}:{line_number} has coordinates outside [0, 1]")

        rows.append((class_id, x_center, y_center, width, height))

    return rows


def find_image_paths(image_dir: Path) -> dict[str, Path]:
    images = {}
    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images[path.stem] = path
    return images


def find_label_paths(label_dir: Path) -> dict[str, Path]:
    return {path.stem: path for path in label_dir.glob("*.txt") if path.is_file()}


def collect_samples(
    image_dir: Path,
    label_dir: Path,
    allow_nonstandard_labels: bool,
) -> tuple[list[Sample], dict[str, object]]:
    image_paths = find_image_paths(image_dir)
    label_paths = find_label_paths(label_dir)

    missing_labels = sorted(set(image_paths) - set(label_paths))
    missing_images = sorted(set(label_paths) - set(image_paths))
    matched_stems = sorted(set(image_paths) & set(label_paths))

    samples = []
    label_warnings = []
    for stem in matched_stems:
        label_rows = read_label(label_paths[stem])
        classes = sorted(row[0] for row in label_rows)
        if classes != [0, 1]:
            message = f"{label_paths[stem].name} has classes {classes}, expected [0, 1]"
            if not allow_nonstandard_labels:
                raise ValueError(message)
            label_warnings.append(message)

        samples.append(Sample(stem=stem, image_path=image_paths[stem], label_path=label_paths[stem]))

    summary = {
        "source_image_dir": str(image_dir),
        "source_label_dir": str(label_dir),
        "total_images": len(image_paths),
        "total_labels": len(label_paths),
        "matched_samples": len(samples),
        "missing_labels": missing_labels,
        "missing_images": missing_images,
        "label_warnings": label_warnings,
    }
    return samples, summary


def ensure_empty_output(output_dir: Path, overwrite: bool) -> None:
    generated_paths = [
        output_dir / "images" / "train",
        output_dir / "images" / "val",
        output_dir / "labels" / "train",
        output_dir / "labels" / "val",
    ]
    metadata_paths = [
        output_dir / "data.yaml",
        output_dir / "split_manifest.csv",
        output_dir / "dataset_summary.json",
        output_dir / "labels" / "train.cache",
        output_dir / "labels" / "val.cache",
    ]

    if not output_dir.exists():
        return

    has_content = any(path.exists() for path in generated_paths + metadata_paths)
    if has_content and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains generated files. Re-run with --overwrite to replace them."
        )

    if overwrite:
        for path in generated_paths:
            if path.exists():
                shutil.rmtree(path)
        for path in metadata_paths:
            if path.exists():
                path.unlink()


def place_image(source: Path, destination_dir: Path, mode: str) -> Path:
    if mode == "png":
        destination = destination_dir / f"{source.stem}.png"
        with Image.open(source) as image:
            image.convert("RGB").save(destination)
        return destination

    destination = destination_dir / source.name
    if destination.exists() or destination.is_symlink():
        destination.unlink()

    if mode == "symlink":
        try:
            os.symlink(source.resolve(), destination)
            return destination
        except OSError:
            # Some filesystems disable symlinks. Fall back to copying so the run still succeeds.
            pass

    shutil.copy2(source, destination)
    return destination


def place_label(source: Path, destination_dir: Path) -> Path:
    destination = destination_dir / source.name
    shutil.copy2(source, destination)
    return destination


def write_data_yaml(output_dir: Path) -> Path:
    data_yaml = output_dir / "data.yaml"
    names = "\n".join(f"  {class_id}: {name}" for class_id, name in CLASS_NAMES.items())
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "nc: 2",
                "names:",
                names,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def split_samples(samples: list[Sample], val_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")

    shuffled = samples[:]
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_ratio))
    val_samples = sorted(shuffled[:val_count], key=lambda sample: sample.stem)
    train_samples = sorted(shuffled[val_count:], key=lambda sample: sample.stem)
    return train_samples, val_samples


def build_dataset(args: argparse.Namespace) -> dict[str, object]:
    image_dir = resolve_dir(
        args.image_dir,
        candidates=[
            Path("CheckImages"),
            Path("../archive/CheckImages"),
            Path("archive/CheckImages"),
        ],
        description="image directory",
    )
    label_dir = resolve_dir(args.label_dir, candidates=[Path("BoundingBoxes")], description="label directory")
    output_dir = args.output_dir.resolve()

    samples, summary = collect_samples(
        image_dir=image_dir,
        label_dir=label_dir,
        allow_nonstandard_labels=args.allow_nonstandard_labels,
    )
    if not samples:
        raise RuntimeError("No matched image/label pairs found.")

    ensure_empty_output(output_dir, overwrite=args.overwrite)

    train_samples, val_samples = split_samples(samples, args.val_ratio, args.seed)
    split_map = {
        "train": train_samples,
        "val": val_samples,
    }

    manifest_rows = []
    for split, split_samples_list in split_map.items():
        image_out_dir = output_dir / "images" / split
        label_out_dir = output_dir / "labels" / split
        image_out_dir.mkdir(parents=True, exist_ok=True)
        label_out_dir.mkdir(parents=True, exist_ok=True)

        for sample in split_samples_list:
            image_destination = place_image(sample.image_path, image_out_dir, args.image_mode)
            label_destination = place_label(sample.label_path, label_out_dir)
            manifest_rows.append(
                {
                    "split": split,
                    "stem": sample.stem,
                    "source_image": str(sample.image_path),
                    "source_label": str(sample.label_path),
                    "image": str(image_destination),
                    "label": str(label_destination),
                }
            )

    data_yaml = write_data_yaml(output_dir)

    summary.update(
        {
            "output_dir": str(output_dir),
            "data_yaml": str(data_yaml),
            "image_mode": args.image_mode,
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "class_names": CLASS_NAMES,
        }
    )

    with (output_dir / "split_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "stem", "source_image", "source_label", "image", "label"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return summary


def main() -> int:
    args = parse_args()
    try:
        summary = build_dataset(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("YOLO dataset prepared.")
    print(f"  output_dir: {summary['output_dir']}")
    print(f"  data_yaml: {summary['data_yaml']}")
    print(f"  matched_samples: {summary['matched_samples']}")
    print(f"  train_samples: {summary['train_samples']}")
    print(f"  val_samples: {summary['val_samples']}")
    if summary["missing_labels"]:
        print(f"  skipped_images_without_labels: {', '.join(summary['missing_labels'])}")
    if summary["missing_images"]:
        print(f"  labels_without_images: {', '.join(summary['missing_images'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
