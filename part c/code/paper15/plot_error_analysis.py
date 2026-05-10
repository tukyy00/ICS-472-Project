"""Create report-ready plots from Paper15 error-analysis tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RUNS = ("validation_beam10", "test_beam10")
MODES = ("raw", "normalized", "normalized_join_waw")
MAIN_MODE = "normalized_join_waw"
COLORS = {
    "substitutions": "#2563eb",
    "deletions": "#dc2626",
    "insertions": "#16a34a",
    "cer": "#7c3aed",
    "wer": "#ea580c",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_count_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_label(text: str) -> str:
    if text == "":
        return "<empty>"
    return text


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(error_dir: Path, output_dir: Path) -> None:
    labels: list[str] = []
    cer_values: list[float] = []
    wer_values: list[float] = []

    for run in RUNS:
        for mode in MODES:
            summary_path = error_dir / run / mode / "summary.json"
            if not summary_path.exists():
                continue
            metrics = load_json(summary_path)["metrics"]
            labels.append(f"{run.replace('_beam10', '')}\n{mode.replace('_', ' ')}")
            cer_values.append(metrics["cer_percent"])
            wer_values.append(metrics["wer_percent"])

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - width / 2, cer_values, width, label="CER", color=COLORS["cer"])
    ax.bar(x + width / 2, wer_values, width, label="WER", color=COLORS["wer"])
    ax.set_title("Paper15 CER/WER by Evaluation Mode")
    ax.set_ylabel("Error rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, output_dir / "cer_wer_comparison.png")


def plot_operation_bars(error_dir: Path, output_dir: Path, level: str) -> None:
    labels: list[str] = []
    subs: list[int] = []
    dels: list[int] = []
    ins: list[int] = []

    op_key = "character_operations" if level == "character" else "word_operations"
    for run in RUNS:
        summary_path = error_dir / run / MAIN_MODE / "summary.json"
        if not summary_path.exists():
            continue
        ops = load_json(summary_path)[op_key]
        labels.append(run.replace("_beam10", ""))
        subs.append(ops["substitutions"])
        dels.append(ops["deletions"])
        ins.append(ops["insertions"])

    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, subs, width, label="Substitutions", color=COLORS["substitutions"])
    ax.bar(x, dels, width, label="Deletions", color=COLORS["deletions"])
    ax.bar(x + width, ins, width, label="Insertions", color=COLORS["insertions"])
    ax.set_title(f"{level.title()} Edit Operation Counts ({MAIN_MODE})")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, output_dir / f"{level}_operation_counts.png")


def plot_top_counts(
    csv_path: Path,
    output_path: Path,
    title: str,
    value_field: str = "value",
    top_k: int = 15,
    color: str = "#475569",
) -> None:
    rows = read_count_csv(csv_path)[:top_k]
    if not rows:
        return
    labels = [safe_label(row[value_field]) for row in rows][::-1]
    counts = [int(row["count"]) for row in rows][::-1]

    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.35 * len(labels) + 1.5)))
    ax.barh(labels, counts, color=color)
    ax.set_title(title)
    ax.set_xlabel("Count")
    ax.grid(axis="x", alpha=0.25)
    for i, count in enumerate(counts):
        ax.text(count + max(counts) * 0.01, i, str(count), va="center", fontsize=8)
    save_figure(fig, output_path)


def plot_top_substitutions(csv_path: Path, output_path: Path, title: str, top_k: int = 15) -> None:
    rows = read_count_csv(csv_path)[:top_k]
    if not rows:
        return
    labels = [f"{safe_label(row['reference'])} -> {safe_label(row['prediction'])}" for row in rows][::-1]
    counts = [int(row["count"]) for row in rows][::-1]

    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.35 * len(labels) + 1.5)))
    ax.barh(labels, counts, color=COLORS["substitutions"])
    ax.set_title(title)
    ax.set_xlabel("Count")
    ax.grid(axis="x", alpha=0.25)
    for i, count in enumerate(counts):
        ax.text(count + max(counts) * 0.01, i, str(count), va="center", fontsize=8)
    save_figure(fig, output_path)


def plot_confusion_heatmap(csv_path: Path, output_path: Path, title: str, top_k: int = 20) -> None:
    rows = read_count_csv(csv_path)[:top_k]
    if not rows:
        return

    refs = sorted({row["reference"] for row in rows})
    preds = sorted({row["prediction"] for row in rows})
    ref_to_idx = {value: idx for idx, value in enumerate(refs)}
    pred_to_idx = {value: idx for idx, value in enumerate(preds)}
    matrix = np.zeros((len(refs), len(preds)), dtype=int)
    for row in rows:
        matrix[ref_to_idx[row["reference"]], pred_to_idx[row["prediction"]]] += int(row["count"])

    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(preds) + 3), max(5, 0.45 * len(refs) + 2)))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted character")
    ax.set_ylabel("Reference character")
    ax.set_xticks(np.arange(len(preds)))
    ax.set_yticks(np.arange(len(refs)))
    ax.set_xticklabels([safe_label(value) for value in preds], rotation=45, ha="right")
    ax.set_yticklabels([safe_label(value) for value in refs])
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if value:
                ax.text(col_idx, row_idx, str(value), ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Count")
    save_figure(fig, output_path)


def make_plots(error_dir: Path, output_dir: Path) -> None:
    plot_metric_comparison(error_dir, output_dir)
    plot_operation_bars(error_dir, output_dir, "character")
    plot_operation_bars(error_dir, output_dir, "word")

    for run in RUNS:
        mode_dir = error_dir / run / MAIN_MODE
        run_output = output_dir / run
        label = run.replace("_", " ").title()
        plot_top_substitutions(
            mode_dir / "char_substitutions.csv",
            run_output / "top_character_substitutions.png",
            f"{label}: Top Character Substitutions",
        )
        plot_top_counts(
            mode_dir / "char_deletions.csv",
            run_output / "top_character_deletions.png",
            f"{label}: Top Character Deletions",
            color=COLORS["deletions"],
        )
        plot_top_counts(
            mode_dir / "char_insertions.csv",
            run_output / "top_character_insertions.png",
            f"{label}: Top Character Insertions",
            color=COLORS["insertions"],
        )
        plot_confusion_heatmap(
            mode_dir / "char_substitutions.csv",
            run_output / "character_substitution_heatmap.png",
            f"{label}: Character Substitution Heatmap",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--error-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_plots(args.error_dir, args.output_dir)
    print(f"Saved error-analysis images to {args.output_dir}")


if __name__ == "__main__":
    main()
