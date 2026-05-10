"""Create a paper-style character confusion matrix with insertion/deletion margins."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


INSERTION_LABEL = "Insertion"
DELETION_LABEL = "Deletion"
RAW_INSERTION = "<INS>"
RAW_DELETION = "<DEL>"


def read_confusion(path: Path) -> tuple[list[str], list[str], dict[tuple[str, str], int]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    refs = sorted({row["reference"] for row in rows if row["reference"] != RAW_INSERTION})
    preds = sorted({row["prediction"] for row in rows if row["prediction"] != RAW_DELETION})

    # Put digits first, then Arabic letters, then any remaining symbols. Keep space off
    # the matrix because it dominates neither the paper figure nor character OCR.
    refs = [value for value in refs if value != " "]
    preds = [value for value in preds if value != " "]

    def sort_key(value: str) -> tuple[int, str]:
        if value.isdigit():
            return (0, value)
        return (1, value)

    refs = sorted(refs, key=sort_key) + [INSERTION_LABEL]
    preds = sorted(preds, key=sort_key) + [DELETION_LABEL]

    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        ref = row["reference"]
        pred = row["prediction"]
        if ref == " " or pred == " ":
            continue
        if ref == RAW_INSERTION:
            ref = INSERTION_LABEL
        if pred == RAW_DELETION:
            pred = DELETION_LABEL
        counts[(ref, pred)] = counts.get((ref, pred), 0) + int(row["count"])
    return refs, preds, counts


def matrix_from_counts(
    refs: list[str],
    preds: list[str],
    counts: dict[tuple[str, str], int],
) -> np.ndarray:
    matrix = np.zeros((len(refs), len(preds)), dtype=int)
    ref_to_index = {value: idx for idx, value in enumerate(refs)}
    pred_to_index = {value: idx for idx, value in enumerate(preds)}
    for (ref, pred), count in counts.items():
        if ref in ref_to_index and pred in pred_to_index:
            matrix[ref_to_index[ref], pred_to_index[pred]] += count
    return matrix


def display_matrix(matrix: np.ndarray) -> np.ndarray:
    # Compress large diagonal counts visually while preserving the real numbers in cells.
    shown = np.sqrt(matrix.astype(float))
    shown[matrix == 0] = 0
    return shown


def plot_paper_matrix(
    confusion_csv: Path,
    output_png: Path,
    title: str,
    caption: str,
) -> None:
    refs, preds, counts = read_confusion(confusion_csv)
    matrix = matrix_from_counts(refs, preds, counts)

    cell_size = 0.42
    fig_width = max(11.5, len(preds) * cell_size)
    fig_height = max(10.5, len(refs) * cell_size + 1.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.imshow(display_matrix(matrix), cmap="Blues", interpolation="nearest")
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Predicted Labels", fontsize=11, labelpad=12)
    ax.set_ylabel("True Labels", fontsize=11, labelpad=12)
    ax.set_xticks(np.arange(len(preds)))
    ax.set_yticks(np.arange(len(refs)))
    ax.set_xticklabels(preds, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(refs, fontsize=8)

    ax.set_xticks(np.arange(-0.5, len(preds), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(refs), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if value == 0:
                text_color = "#111827"
                text = "0"
                fontweight = "normal"
            else:
                text_color = "white" if display_matrix(matrix)[row_idx, col_idx] > display_matrix(matrix).max() * 0.55 else "#111827"
                text = str(value)
                fontweight = "bold" if row_idx == col_idx else "normal"
            ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=6.5, color=text_color, fontweight=fontweight)

    deletion_col = len(preds) - 1
    insertion_row = len(refs) - 1
    ax.add_patch(
        patches.Rectangle(
            (deletion_col - 0.5, -0.5),
            1,
            len(refs),
            fill=False,
            edgecolor="#dc2626",
            linewidth=1.8,
        )
    )
    ax.add_patch(
        patches.Rectangle(
            (-0.5, insertion_row - 0.5),
            len(preds),
            1,
            fill=False,
            edgecolor="#16a34a",
            linewidth=1.8,
        )
    )

    fig.text(
        0.5,
        0.02,
        caption,
        ha="center",
        va="bottom",
        fontsize=11,
        fontfamily="serif",
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.98))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=260, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confusion-csv", required=True, type=Path)
    parser.add_argument("--output-png", required=True, type=Path)
    parser.add_argument("--title", default="Confusion Matrix")
    parser.add_argument(
        "--caption",
        default="Figure: Confusion matrix for character recognition using the CNN-BiLSTM-CTC model, including insertions and deletions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_paper_matrix(args.confusion_csv, args.output_png, args.title, args.caption)
    print(f"Saved paper-style confusion matrix to {args.output_png}")


if __name__ == "__main__":
    main()
