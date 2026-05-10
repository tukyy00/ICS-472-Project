"""Train the Paper15 legal amount CRNN."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .charset import build_charset
from .dataset import LegalSample, Paper15Dataset, collate_batch, load_manifest
from .metrics import evaluate_texts
from .model import Paper15CRNN


def split_samples(samples: list[LegalSample], seed: int) -> tuple[list[LegalSample], list[LegalSample], list[LegalSample]]:
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    n_test = round(len(shuffled) * 0.10)
    n_val = round(len(shuffled) * 0.10)
    return shuffled[n_test + n_val :], shuffled[n_test : n_test + n_val], shuffled[:n_test]


def write_split(path: Path, samples: list[LegalSample]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check_id", "crop_path", "label"])
        writer.writeheader()
        for sample in samples:
            writer.writerow({"check_id": sample.check_id, "crop_path": sample.crop_path, "label": sample.label})


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, charset, device: torch.device) -> dict[str, object]:
    model.eval()
    refs: dict[str, str] = {}
    hyps: dict[str, str] = {}
    loss_sum = 0.0
    for batch in tqdm(loader, desc="val", leave=False, dynamic_ncols=True):
        images = batch["images"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        logits = model(images)
        log_probs = logits.log_softmax(dim=2)
        input_lengths = torch.full((images.size(0),), logits.size(0), dtype=torch.long, device=device)
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        loss_sum += float(loss.item()) * images.size(0)
        decoded = [charset.decode(seq) for seq in logits.argmax(dim=2).cpu().permute(1, 0).tolist()]
        for check_id, label, pred in zip(batch["check_ids"], batch["labels"], decoded):
            refs[check_id] = label
            hyps[check_id] = pred
    metrics = evaluate_texts(refs, hyps)
    metrics["loss"] = loss_sum / max(1, len(loader.dataset))
    return metrics


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_manifest(args.manifest, require_labels=True, canonical=True)
    train_samples, val_samples, test_samples = split_samples(samples, args.seed)
    charset = build_charset([sample.label for sample in samples])
    charset.save(output_dir / "charset.json")
    write_split(output_dir / "train_split.csv", train_samples)
    write_split(output_dir / "val_split.csv", val_samples)
    write_split(output_dir / "internal_test_split.csv", test_samples)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader = DataLoader(
        Paper15Dataset(train_samples, charset, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        Paper15Dataset(val_samples, charset, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = Paper15CRNN(charset.num_classes, lstm_hidden=384, lstm_layers=2, dropout=0.1).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(device=device.type, enabled=device.type == "cuda" and args.amp)

    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    best_loss = float("inf")
    stale_epochs = 0
    global_step = 0
    history: list[dict[str, object]] = []
    for epoch in tqdm(range(1, args.epochs + 1), desc="epochs", dynamic_ncols=True):
        model.train()
        loss_sum = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch} train", leave=False, dynamic_ncols=True):
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=scaler.is_enabled()):
                logits = model(images)
                log_probs = logits.log_softmax(dim=2)
                input_lengths = torch.full((images.size(0),), logits.size(0), dtype=torch.long, device=device)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            if global_step % args.lr_decay_steps == 0:
                for group in optimizer.param_groups:
                    group["lr"] *= args.lr_decay_gamma
            loss_sum += float(loss.item()) * images.size(0)
        val_metrics = evaluate(model, val_loader, criterion, charset, device)
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": loss_sum / max(1, len(train_loader.dataset)),
            "lr": optimizer.param_groups[0]["lr"],
            "val": val_metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False, indent=2))
        val_loss = float(val_metrics["loss"])
        if val_loss < best_loss:
            best_loss = val_loss
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "arch": "paper15",
                    "unit": "char",
                    "codec_filename": "charset.json",
                    "label_mode": "canonical",
                    "image_height": 140,
                    "image_width": 1340,
                    "exact_resize": True,
                    "flip_image": True,
                    "normalize_minus_one_one": True,
                    "rtl": False,
                    "lstm_hidden": 384,
                    "lstm_layers": 2,
                    "dropout": 0.1,
                    "num_classes": charset.num_classes,
                    "checkpoint_metric": "loss",
                    "best_score": best_loss,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                output_dir / "best_legal_crnn_ctc.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="outputs/part_c/train_legal_manifest.csv")
    parser.add_argument("--output-dir", default="outputs/part_c/legal_crnn_paper15")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-decay-steps", type=int, default=10000)
    parser.add_argument("--lr-decay-gamma", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

