"""Dataset loader for Paper15 legal amount crops."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .charset import Charset
from .normalize import canonicalize_legal_label


@dataclass(frozen=True)
class LegalSample:
    check_id: str
    crop_path: Path
    label: str


def load_manifest(path: str | Path, require_labels: bool = True, canonical: bool = True) -> list[LegalSample]:
    samples: list[LegalSample] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row = {key.lstrip("\ufeff").strip('"'): value for key, value in row.items()}
            if row.get("status", "ok") != "ok":
                continue
            crop_path = Path(row["crop_path"])
            label = row.get("label", "") or ""
            if canonical and label:
                label = canonicalize_legal_label(label)
            if not crop_path.exists() or (require_labels and not label):
                continue
            samples.append(LegalSample(row["check_id"], crop_path, label))
    return samples


class Paper15Dataset(Dataset):
    def __init__(self, samples: list[LegalSample], charset: Charset, augment: bool = False) -> None:
        self.samples = samples
        self.charset = charset
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        image = Image.open(sample.crop_path).convert("L")
        image = ImageOps.autocontrast(image)
        if self.augment:
            image = self._augment(image)
        image = image.resize((1340, 140), Image.Resampling.BILINEAR)
        image = ImageOps.mirror(image)
        tensor = TF.to_tensor(image).mul(2.0).sub(1.0)
        target = torch.tensor(self.charset.encode(sample.label), dtype=torch.long)
        return {
            "check_id": sample.check_id,
            "image": tensor,
            "target": target,
            "target_length": len(target),
            "label": sample.label,
        }

    def _augment(self, image: Image.Image) -> Image.Image:
        if random.random() < 0.6:
            image = TF.affine(
                image,
                angle=random.uniform(-1.5, 1.5),
                translate=(random.randint(-4, 4), random.randint(-2, 2)),
                scale=random.uniform(0.97, 1.03),
                shear=0,
            )
        if random.random() < 0.5:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.3))
        if random.random() < 0.35:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.88, 1.15))
        if random.random() < 0.15:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.15, 0.45)))
        return image


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "check_ids": [item["check_id"] for item in batch],
        "images": torch.stack([item["image"] for item in batch]),  # type: ignore[arg-type]
        "targets": torch.cat([item["target"] for item in batch]),  # type: ignore[arg-type]
        "target_lengths": torch.tensor([item["target_length"] for item in batch], dtype=torch.long),
        "labels": [item["label"] for item in batch],
    }

