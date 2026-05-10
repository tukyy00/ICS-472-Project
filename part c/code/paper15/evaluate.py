"""Evaluate the Paper15 legal amount CRNN."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .beam import ctc_prefix_beam_decode
from .charset import Charset
from .dataset import Paper15Dataset, collate_batch, load_manifest
from .metrics import evaluate_texts
from .model import Paper15CRNN
from .normalize import join_standalone_waw, normalize_legal_for_scoring


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    charset = Charset.load(args.charset)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    samples = load_manifest(args.manifest, require_labels=False, canonical=True)
    loader = DataLoader(
        Paper15Dataset(samples, charset, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = Paper15CRNN(
        num_classes=int(checkpoint["num_classes"]),
        lstm_hidden=int(checkpoint.get("lstm_hidden", 384)),
        lstm_layers=int(checkpoint.get("lstm_layers", 2)),
        dropout=float(checkpoint.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    references: dict[str, str] = {}
    predictions: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    for batch in tqdm(loader, desc="paper15 predict", dynamic_ncols=True):
        logits = model(batch["images"].to(device))
        if args.decoder == "beam":
            log_probs = logits.log_softmax(dim=2).cpu().permute(1, 0, 2)
            decoded = [
                ctc_prefix_beam_decode(item, charset, beam_width=args.beam_width, topk=args.beam_topk)
                for item in log_probs
            ]
        else:
            decoded = [charset.decode(seq) for seq in logits.argmax(dim=2).cpu().permute(1, 0).tolist()]
        for check_id, label, prediction in zip(batch["check_ids"], batch["labels"], decoded):
            predictions[check_id] = prediction
            if label:
                references[check_id] = label
            rows.append({"check_id": check_id, "label": label, "prediction": prediction})

    with (output_dir / "legal_crnn_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check_id", "label", "prediction"])
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "legal_crnn_predictions.txt").open("w", encoding="utf-8") as handle:
        for check_id in sorted(predictions):
            handle.write(f"{check_id}.tif\t{predictions[check_id]}\n")
    if references:
        norm_refs = {key: normalize_legal_for_scoring(value) for key, value in references.items()}
        norm_preds = {key: normalize_legal_for_scoring(value) for key, value in predictions.items()}
        waw_refs = {key: join_standalone_waw(value) for key, value in norm_refs.items()}
        waw_preds = {key: join_standalone_waw(value) for key, value in norm_preds.items()}
        report = {
            "raw": evaluate_texts(references, predictions),
            "normalized": evaluate_texts(norm_refs, norm_preds),
            "normalized_join_waw": evaluate_texts(waw_refs, waw_preds),
        }
        (output_dir / "legal_crnn_metrics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--charset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--decoder", choices=["greedy", "beam"], default="greedy")
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--beam-topk", type=int, default=16)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
