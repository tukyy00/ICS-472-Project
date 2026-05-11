"""PyTorch dataset for cropped courtesy amount patches."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .ctc import encode_label


@dataclass(frozen=True)
class CropSample:
    check_id: str
    crop_path: Path
    label: str


def load_manifest(path: str | Path, require_labels: bool = True) -> list[CropSample]:
    samples: list[CropSample] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            crop_path = Path(row["crop_path"])
            label = row.get("label", "") or ""
            status = row.get("status", "ok")
            if status != "ok" or not crop_path.exists():
                continue
            if require_labels and not label:
                continue
            samples.append(CropSample(row["check_id"], crop_path, label))
    return samples


class CourtesyCropDataset(Dataset):
    def __init__(
        self,
        samples: list[CropSample],
        image_height: int = 64,
        image_width: int = 384,
        augment: bool = False,
    ) -> None:
        self.samples = samples
        self.image_height = image_height
        self.image_width = image_width
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        image = Image.open(sample.crop_path).convert("L")
        image = ImageOps.autocontrast(image)
        if self.augment:
            image = self._augment(image)
        tensor = self._resize_pad_to_tensor(image)
        target = torch.tensor(encode_label(sample.label), dtype=torch.long)
        return {
            "check_id": sample.check_id,
            "image": tensor,
            "target": target,
            "target_length": len(target),
            "label": sample.label,
        }

    def _augment(self, image: Image.Image) -> Image.Image:
        if random.random() < 0.7:
            angle = random.uniform(-2.5, 2.5)
            translate = (random.randint(-3, 3), random.randint(-2, 2))
            image = TF.affine(image, angle=angle, translate=translate, scale=1.0, shear=0)
        if random.random() < 0.5:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.35))
        if random.random() < 0.4:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.2))
        if random.random() < 0.2:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.7)))
        return image

    def _resize_pad_to_tensor(self, image: Image.Image) -> torch.Tensor:
        width, height = image.size
        scale = min(self.image_width / max(width, 1), self.image_height / max(height, 1))
        resized_width = max(1, min(self.image_width, int(round(width * scale))))
        resized_height = max(1, min(self.image_height, int(round(height * scale))))
        image = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        canvas = Image.new("L", (self.image_width, self.image_height), color=255)
        left = random.randint(0, max(0, self.image_width - resized_width)) if self.augment else 0
        top = (self.image_height - resized_height) // 2
        canvas.paste(image, (left, top))
        return TF.to_tensor(canvas)


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    images = torch.stack([item["image"] for item in batch])  # type: ignore[arg-type]
    targets = torch.cat([item["target"] for item in batch])  # type: ignore[arg-type]
    target_lengths = torch.tensor([item["target_length"] for item in batch], dtype=torch.long)
    return {
        "check_ids": [item["check_id"] for item in batch],
        "images": images,
        "targets": targets,
        "target_lengths": target_lengths,
        "labels": [item["label"] for item in batch],
    }

