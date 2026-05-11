"""Digit-level error analysis for Part B courtesy amount recognition."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from .metrics import evaluate_amounts

INSERTION_LABEL = "<INS>"
DELETION_LABEL = "<DEL>"
DISPLAY_INSERTION = "Insertion"
DISPLAY_DELETION = "Deletion"
DIGITS = [str(i) for i in range(10)]

AlignmentOp = tuple[str, str, str]


def normalize_digits(value: str | int | None) -> str:
    if value is None:
        return ""
    return "".join(char for char in str(value) if char.isdigit())


def align(reference: str, prediction: str) -> list[AlignmentOp]:
    n, m = len(reference), len(prediction)
    dp: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)
    ]
    backtrace: list[list[str | None]] = [[None for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost, ins, dele, sub = dp[i - 1][0]
        dp[i][0] = (cost + 1, ins, dele + 1, sub)
        backtrace[i][0] = "D"
    for j in range(1, m + 1):
        cost, ins, dele, sub = dp[0][j - 1]
        dp[0][j] = (cost + 1, ins + 1, dele, sub)
        backtrace[0][j] = "I"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates: list[tuple[tuple[int, int, int, int], str]] = []
            if reference[i - 1] == prediction[j - 1]:
                candidates.append((dp[i - 1][j - 1], "M"))
            else:
                cost, ins, dele, sub = dp[i - 1][j - 1]
                candidates.append(((cost + 1, ins, dele, sub + 1), "S"))
            cost, ins, dele, sub = dp[i][j - 1]
            candidates.append(((cost + 1, ins + 1, dele, sub), "I"))
            cost, ins, dele, sub = dp[i - 1][j]
            candidates.append(((cost + 1, ins, dele + 1, sub), "D"))
            dp[i][j], backtrace[i][j] = min(
                candidates,
                key=lambda item: (item[0][0], item[0][3], item[0][1], item[0][2]),
            )

    ops: list[AlignmentOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = backtrace[i][j]
        if op in {"M", "S"}:
            ops.append((op, reference[i - 1], prediction[j - 1]))
            i -= 1
            j -= 1
        elif op == "D":
            ops.append((op, reference[i - 1], ""))
            i -= 1
        elif op == "I":
            ops.append((op, "", prediction[j - 1]))
            j -= 1
        else:
            raise RuntimeError("Invalid alignment backtrace")
    return list(reversed(ops))


def read_predictions(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_count_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_error_analysis(prediction_rows: list[dict[str, str]], output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    references: dict[str, str] = {}
    predictions: dict[str, str] = {}
    confusion: Counter[tuple[str, str]] = Counter()
    substitutions: Counter[tuple[str, str]] = Counter()
    deletions: Counter[str] = Counter()
    insertions: Counter[str] = Counter()
    per_sample: list[dict[str, object]] = []

    for row in prediction_rows:
        check_id = row["check_id"]
        reference = normalize_digits(row["label"])
        prediction = normalize_digits(row["prediction"])
        references[check_id] = reference
        predictions[check_id] = prediction
        ops = align(reference, prediction)
        counts = {
            "matches": sum(1 for op, _, _ in ops if op == "M"),
            "substitutions": sum(1 for op, _, _ in ops if op == "S"),
            "deletions": sum(1 for op, _, _ in ops if op == "D"),
            "insertions": sum(1 for op, _, _ in ops if op == "I"),
        }
        for op, ref, pred in ops:
            if op == "I":
                confusion[(INSERTION_LABEL, pred)] += 1
                insertions[pred] += 1
            elif op == "D":
                confusion[(ref, DELETION_LABEL)] += 1
                deletions[ref] += 1
            else:
                confusion[(ref, pred)] += 1
                if op == "S":
                    substitutions[(ref, pred)] += 1
        edits = counts["substitutions"] + counts["deletions"] + counts["insertions"]
        per_sample.append(
            {
                "check_id": check_id,
                "reference": reference,
                "prediction": prediction,
                "exact": int(edits == 0),
                "edit_count": edits,
                "substitutions": counts["substitutions"],
                "deletions": counts["deletions"],
                "insertions": counts["insertions"],
                "reference_length": len(reference),
                "prediction_length": len(prediction),
            }
        )

    metrics = evaluate_amounts(references, predictions)
    summary = {
        "metrics": metrics,
        "operation_share_of_digit_errors": {
            "substitutions": _share(metrics["substitutions"], _total_errors(metrics)),
            "deletions": _share(metrics["deletions"], _total_errors(metrics)),
            "insertions": _share(metrics["insertions"], _total_errors(metrics)),
        },
        "top_digit_substitutions": [
            {"reference": ref, "prediction": pred, "count": count}
            for (ref, pred), count in substitutions.most_common()
        ],
        "top_digit_deletions": [{"value": value, "count": count} for value, count in deletions.most_common()],
        "top_digit_insertions": [{"value": value, "count": count} for value, count in insertions.most_common()],
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_count_csv(
        output_dir / "digit_confusion_long.csv",
        [
            {"reference": ref, "prediction": pred, "count": count}
            for (ref, pred), count in sorted(confusion.items(), key=lambda item: (item[0][0], item[0][1]))
        ],
        ["reference", "prediction", "count"],
    )
    write_count_csv(
        output_dir / "digit_substitutions.csv",
        [
            {"reference": ref, "prediction": pred, "count": count}
            for (ref, pred), count in substitutions.most_common()
        ],
        ["reference", "prediction", "count"],
    )
    write_count_csv(
        output_dir / "digit_deletions.csv",
        [{"value": value, "count": count} for value, count in deletions.most_common()],
        ["value", "count"],
    )
    write_count_csv(
        output_dir / "digit_insertions.csv",
        [{"value": value, "count": count} for value, count in insertions.most_common()],
        ["value", "count"],
    )
    write_count_csv(
        output_dir / "per_sample_errors.csv",
        per_sample,
        [
            "check_id",
            "reference",
            "prediction",
            "exact",
            "edit_count",
            "substitutions",
            "deletions",
            "insertions",
            "reference_length",
            "prediction_length",
        ],
    )
    plot_confusion_matrix(confusion, output_dir / "images" / "digit_confusion_matrix.png")
    plot_operation_counts(metrics, output_dir / "images" / "digit_operation_counts.png")
    return summary


def _total_errors(metrics: dict[str, float | int]) -> int:
    return int(metrics["insertions"]) + int(metrics["deletions"]) + int(metrics["substitutions"])


def _share(value: float | int, total: int) -> float:
    return 0.0 if total == 0 else float(value) / total * 100.0


def plot_confusion_matrix(confusion: Counter[tuple[str, str]], output_path: Path) -> None:
    rows = DIGITS + [DISPLAY_INSERTION]
    cols = DIGITS + [DISPLAY_DELETION]
    matrix = np.zeros((len(rows), len(cols)), dtype=int)
    row_index = {value: index for index, value in enumerate(rows)}
    col_index = {value: index for index, value in enumerate(cols)}
    for (ref, pred), count in confusion.items():
        ref_display = DISPLAY_INSERTION if ref == INSERTION_LABEL else ref
        pred_display = DISPLAY_DELETION if pred == DELETION_LABEL else pred
        if ref_display in row_index and pred_display in col_index:
            matrix[row_index[ref_display], col_index[pred_display]] += count

    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    shown = np.sqrt(matrix.astype(float))
    image = ax.imshow(shown, cmap="Blues", interpolation="nearest")
    ax.set_title("Courtesy Amount Digit Confusion Matrix")
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(rows)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    max_value = shown.max() if shown.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            color = "white" if max_value and shown[row, col] > max_value * 0.55 else "#111827"
            ax.text(
                col,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=9,
                color=color,
                fontweight="bold" if row == col and rows[row] in DIGITS else "normal",
            )

    deletion_col = len(cols) - 1
    insertion_row = len(rows) - 1
    ax.add_patch(
        patches.Rectangle(
            (deletion_col - 0.5, -0.5),
            1,
            len(rows),
            fill=False,
            edgecolor="#dc2626",
            linewidth=2.0,
        )
    )
    ax.add_patch(
        patches.Rectangle(
            (-0.5, insertion_row - 0.5),
            len(cols),
            1,
            fill=False,
            edgecolor="#16a34a",
            linewidth=2.0,
        )
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="sqrt(count)")
    fig.text(
        0.5,
        0.02,
        "Figure: Confusion matrix for courtesy digit recognition, including insertions and deletions.",
        ha="center",
        fontsize=11,
        fontfamily="serif",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.98))
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_operation_counts(metrics: dict[str, float | int], output_path: Path) -> None:
    labels = ["Substitutions", "Deletions", "Insertions"]
    values = [int(metrics["substitutions"]), int(metrics["deletions"]), int(metrics["insertions"])]
    colors = ["#2563eb", "#dc2626", "#16a34a"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Courtesy Digit Edit Operation Counts")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, str(value), ha="center", va="bottom")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", default="arabic_check_pipeline/part_b_courtesy_ocr/outputs/test_eval/predictions.csv")
    parser.add_argument("--output-dir", default="arabic_check_pipeline/part_b_courtesy_ocr/outputs/error_analysis/test_eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_error_analysis(read_predictions(args.predictions_csv), Path(args.output_dir))
    print(
        "Part B error analysis: "
        f"{summary['metrics']['amounts_evaluated']} samples, "
        f"digit accuracy {summary['metrics']['digit_accuracy_percent']:.2f}%"
    )


if __name__ == "__main__":
    main()
