"""Train the CNN + BiLSTM + CTC courtesy amount recognizer."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .ctc import decode_indices
from .dataset import CourtesyCropDataset, CropSample, collate_batch, load_manifest
from .metrics import evaluate_amounts
from .model import ResNetBiLSTMCTC


def split_samples(
    samples: list[CropSample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[CropSample], list[CropSample], list[CropSample]]:
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))
    test = shuffled[:n_test]
    val = shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]
    return train, val, test


def write_split_manifest(path: Path, samples: list[CropSample]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check_id", "crop_path", "label"])
        writer.writeheader()
        for sample in samples:
            writer.writerow({"check_id": sample.check_id, "crop_path": sample.crop_path, "label": sample.label})


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    epoch: int,
    epochs: int,
) -> float:
    model.train()
    total_loss = 0.0
    seen = 0
    progress = tqdm(loader, desc=f"epoch {epoch}/{epochs} train", leave=False, dynamic_ncols=True)
    for batch in progress:
        images = batch["images"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            log_probs = logits.log_softmax(dim=2)
            input_lengths = torch.full(
                size=(images.size(0),),
                fill_value=logits.size(0),
                dtype=torch.long,
                device=device,
            )
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * images.size(0)
        seen += images.size(0)
        progress.set_postfix(loss=f"{total_loss / max(1, seen):.4f}")
    return total_loss / max(1, len(loader.dataset))


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int | None = None,
    epochs: int | None = None,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    references: dict[str, str] = {}
    predictions: dict[str, str] = {}
    seen = 0
    desc = "eval" if epoch is None or epochs is None else f"epoch {epoch}/{epochs} val"
    progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for batch in progress:
        images = batch["images"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        logits = model(images)
        log_probs = logits.log_softmax(dim=2)
        input_lengths = torch.full(
            size=(images.size(0),),
            fill_value=logits.size(0),
            dtype=torch.long,
            device=device,
        )
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        total_loss += float(loss.item()) * images.size(0)
        best_indices = logits.argmax(dim=2).cpu().permute(1, 0).tolist()
        decoded = [decode_indices(indices) for indices in best_indices]
        for check_id, label, prediction in zip(batch["check_ids"], batch["labels"], decoded):
            references[check_id] = label
            predictions[check_id] = prediction
        seen += images.size(0)
        running_metrics = evaluate_amounts(references, predictions)
        progress.set_postfix(
            loss=f"{total_loss / max(1, seen):.4f}",
            acc=f"{float(running_metrics['digit_accuracy_percent']):.2f}%",
        )
    metrics = evaluate_amounts(references, predictions)
    metrics["loss"] = total_loss / max(1, len(loader.dataset))
    return metrics


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_manifest(args.manifest, require_labels=True)
    if not samples:
        raise RuntimeError(f"No labeled crop samples found in {args.manifest}")
    train_samples, val_samples, test_samples = split_samples(samples, args.val_ratio, args.test_ratio, args.seed)
    write_split_manifest(output_dir / "train_split.csv", train_samples)
    write_split_manifest(output_dir / "val_split.csv", val_samples)
    write_split_manifest(output_dir / "internal_test_split.csv", test_samples)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader = DataLoader(
        CourtesyCropDataset(train_samples, args.image_height, args.image_width, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        CourtesyCropDataset(val_samples, args.image_height, args.image_width, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )

    model = ResNetBiLSTMCTC(dropout=args.dropout).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(device=device.type, enabled=device.type == "cuda" and args.amp)

    best_accuracy = -1e9
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    checkpoint_path = output_dir / "best_crnn_ctc.pt"

    epoch_progress = tqdm(range(1, args.epochs + 1), desc="epochs", dynamic_ncols=True)
    for epoch in epoch_progress:
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            scaler.is_enabled(),
            epoch,
            args.epochs,
        )
        val_metrics = evaluate_loader(model, val_loader, criterion, device, epoch, args.epochs)
        val_accuracy = float(val_metrics["digit_accuracy_percent"])
        record = {"epoch": epoch, "train_loss": train_loss, "val": val_metrics}
        history.append(record)
        print(json.dumps(record, indent=2))
        epoch_progress.set_postfix(train_loss=f"{train_loss:.4f}", val_acc=f"{val_accuracy:.2f}%")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "image_height": args.image_height,
                    "image_width": args.image_width,
                    "dropout": args.dropout,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )
            print(f"Saved {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {args.patience} epochs without improvement.")
                break

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if test_samples:
        print(f"Internal test split saved at {output_dir / 'internal_test_split.csv'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="outputs/part_b/train_manifest.csv")
    parser.add_argument("--output-dir", default="outputs/part_b/checkpoints")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--image-width", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
