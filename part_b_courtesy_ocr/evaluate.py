"""Evaluate or predict with a trained Part B recognizer."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .ctc import ctc_prefix_beam_decode, decode_indices
from .dataset import CourtesyCropDataset, collate_batch, load_manifest
from .metrics import evaluate_amounts
from .model import build_model


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    image_height = int(checkpoint.get("image_height", args.image_height))
    image_width = int(checkpoint.get("image_width", args.image_width))
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    samples = load_manifest(args.manifest, require_labels=False)
    if not samples:
        raise RuntimeError(f"No usable crops found in {args.manifest}")
    dataset = CourtesyCropDataset(samples, image_height=image_height, image_width=image_width, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        arch=str(checkpoint.get("arch", "resnet18")),
        dropout=float(checkpoint.get("dropout", 0.1)),
        lstm_hidden=int(checkpoint.get("lstm_hidden", 256)),
        lstm_layers=int(checkpoint.get("lstm_layers", 2)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    predictions: dict[str, str] = {}
    references: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    progress = tqdm(loader, desc="predict", dynamic_ncols=True)
    for batch in progress:
        images = batch["images"].to(device)
        logits = model(images)
        if args.decoder == "beam":
            log_probs = logits.log_softmax(dim=2).cpu().permute(1, 0, 2)
            decoded = [
                ctc_prefix_beam_decode(
                    item,
                    beam_width=args.beam_width,
                    topk=args.beam_topk,
                )
                for item in log_probs
            ]
        else:
            best_indices = logits.argmax(dim=2).cpu().permute(1, 0).tolist()
            decoded = [decode_indices(indices) for indices in best_indices]
        for check_id, label, prediction in zip(batch["check_ids"], batch["labels"], decoded):
            predictions[check_id] = prediction
            if label:
                references[check_id] = label
            rows.append({"check_id": check_id, "label": label, "prediction": prediction})
        progress.set_postfix(amounts=len(predictions))

    prediction_txt = output_dir / "predictions.txt"
    prediction_csv = output_dir / "predictions.csv"
    with prediction_txt.open("w", encoding="utf-8") as handle:
        for check_id in sorted(predictions):
            handle.write(f"{check_id}.tif {predictions[check_id]}\n")
    with prediction_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check_id", "label", "prediction"])
        writer.writeheader()
        writer.writerows(rows)

    if references:
        metrics = evaluate_amounts(references, predictions)
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
    print(f"Wrote {prediction_txt}")
    print(f"Wrote {prediction_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", default="arabic_check_pipeline/part_b_courtesy_ocr/outputs/checkpoints/best_crnn_ctc.pt")
    parser.add_argument("--output-dir", default="arabic_check_pipeline/part_b_courtesy_ocr/outputs/eval")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--image-width", type=int, default=384)
    parser.add_argument("--device", default=None)
    parser.add_argument("--decoder", choices=["greedy", "beam"], default="greedy")
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--beam-topk", type=int, default=11)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
