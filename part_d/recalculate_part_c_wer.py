"""Recalculate Part C scoring after converting legal text to numeric amounts.

This script is intentionally self-contained with the files in this folder:

- data/part_c_legal_predictions.csv
- data/part_b_courtesy_predictions.csv
- converter.py

It treats each converted amount as one word, so the converted WER is the
strict amount error rate: 0 for an exact numeric amount match, 1 otherwise.
Digit CER is also reported to show near misses.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from converter import convert_legal_amount, normalize_amount_digits


CONVERSION_STAGES = ("direct", "edit_distance", "edit_distance_split")


def read_csv_by_id(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["check_id"]: row for row in csv.DictReader(handle)}


def edit_counts(reference: str, prediction: str) -> tuple[int, int, int]:
    n, m = len(reference), len(prediction)
    dp = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, j, 0, 0)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            substitution_cost = 0 if reference[i - 1] == prediction[j - 1] else 1
            previous_cost, previous_i, previous_d, previous_s = dp[i - 1][j - 1]
            candidates = [
                (
                    previous_cost + substitution_cost,
                    previous_i,
                    previous_d,
                    previous_s + substitution_cost,
                ),
                (dp[i][j - 1][0] + 1, dp[i][j - 1][1] + 1, dp[i][j - 1][2], dp[i][j - 1][3]),
                (dp[i - 1][j][0] + 1, dp[i - 1][j][1], dp[i - 1][j][2] + 1, dp[i - 1][j][3]),
            ]
            dp[i][j] = min(candidates, key=lambda item: (item[0], item[1] + item[2], item[3]))

    _, insertions, deletions, substitutions = dp[n][m]
    return insertions, deletions, substitutions


def amount_metrics(rows: list[dict[str, str]], value_column: str) -> dict[str, float | int]:
    amount_errors = 0
    unresolved = 0
    digit_insertions = 0
    digit_deletions = 0
    digit_substitutions = 0
    digit_reference_count = 0

    for row in rows:
        reference = normalize_amount_digits(row["courtesy_true"])
        prediction = normalize_amount_digits(row[value_column])
        if not prediction:
            unresolved += 1
        amount_errors += int(reference != prediction)
        insertions, deletions, substitutions = edit_counts(reference, prediction)
        digit_insertions += insertions
        digit_deletions += deletions
        digit_substitutions += substitutions
        digit_reference_count += len(reference)

    count = len(rows)
    digit_errors = digit_insertions + digit_deletions + digit_substitutions
    return {
        "amounts_evaluated": count,
        "converted_wer_percent": 0.0 if count == 0 else amount_errors / count * 100.0,
        "exact_amount_percent": 0.0 if count == 0 else (count - amount_errors) / count * 100.0,
        "digit_cer_percent": 0.0 if digit_reference_count == 0 else digit_errors / digit_reference_count * 100.0,
        "digit_accuracy_percent": 0.0
        if digit_reference_count == 0
        else (1.0 - digit_errors / digit_reference_count) * 100.0,
        "amount_errors": amount_errors,
        "unresolved_count": unresolved,
        "digit_reference_count": digit_reference_count,
        "digit_insertions": digit_insertions,
        "digit_deletions": digit_deletions,
        "digit_substitutions": digit_substitutions,
    }


def build_rows(
    legal_rows: dict[str, dict[str, str]],
    courtesy_rows: dict[str, dict[str, str]],
) -> tuple[list[dict[str, str]], list[str], list[str]]:
    overlap = sorted(set(legal_rows) & set(courtesy_rows))
    legal_only = sorted(set(legal_rows) - set(courtesy_rows))
    courtesy_only = sorted(set(courtesy_rows) - set(legal_rows))

    rows: list[dict[str, str]] = []
    for check_id in overlap:
        legal = legal_rows[check_id]
        courtesy = courtesy_rows[check_id]
        row = {
            "check_id": check_id,
            "part_c_label_text": legal["label"],
            "part_c_prediction_text": legal["prediction"],
            "courtesy_true": normalize_amount_digits(courtesy["label"]),
            "courtesy_prediction": normalize_amount_digits(courtesy["prediction"]),
        }
        for stage in CONVERSION_STAGES:
            row[f"label_{stage}_value"] = convert_legal_amount(legal["label"], stage).value
            row[f"prediction_{stage}_value"] = convert_legal_amount(legal["prediction"], stage).value
        rows.append(row)

    return rows, legal_only, courtesy_only


def load_text_metrics(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    legal_rows = read_csv_by_id(args.part_c_predictions)
    courtesy_rows = read_csv_by_id(args.courtesy_predictions)
    rows, legal_only, courtesy_only = build_rows(legal_rows, courtesy_rows)

    metrics = {
        "inputs": {
            "part_c_predictions": str(args.part_c_predictions),
            "courtesy_predictions": str(args.courtesy_predictions),
            "part_c_rows": len(legal_rows),
            "courtesy_rows": len(courtesy_rows),
            "overlap_rows": len(rows),
            "part_c_only": legal_only,
            "courtesy_only": courtesy_only,
        },
        "original_part_c_text_metrics": load_text_metrics(args.part_c_text_metrics),
        "converted_part_c_prediction": {
            stage: amount_metrics(rows, f"prediction_{stage}_value") for stage in CONVERSION_STAGES
        },
        "converted_part_c_label_check": {
            stage: amount_metrics(rows, f"label_{stage}_value") for stage in CONVERSION_STAGES
        },
    }

    prediction_path = output_dir / "part_c_converted_predictions.csv"
    fieldnames = [
        "check_id",
        "courtesy_true",
        "courtesy_prediction",
        "part_c_label_text",
        "part_c_prediction_text",
    ]
    for prefix in ("label", "prediction"):
        for stage in CONVERSION_STAGES:
            fieldnames.append(f"{prefix}_{stage}_value")

    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metrics_path = output_dir / "converted_part_c_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_path = output_dir / "summary.md"
    best = metrics["converted_part_c_prediction"]["edit_distance_split"]
    original = metrics["original_part_c_text_metrics"] or {}
    normalized = original.get("normalized", {}) if isinstance(original, dict) else {}
    summary_path.write_text(
        "\n".join(
            [
                "# Converted Part C WER Check",
                "",
                f"- Overlap rows evaluated: `{len(rows)}`",
                f"- Original Part C normalized text WER: `{normalized.get('wer_percent', 'n/a')}`",
                f"- Converted Part C amount WER: `{best['converted_wer_percent']:.2f}%`",
                f"- Converted Part C exact amount accuracy: `{best['exact_amount_percent']:.2f}%`",
                f"- Converted Part C digit CER: `{best['digit_cer_percent']:.2f}%`",
                f"- Converted label consistency exact accuracy: "
                f"`{metrics['converted_part_c_label_check']['edit_distance_split']['exact_amount_percent']:.2f}%`",
                "",
                "The converted WER treats the full numeric amount as one word. This checks whether the Part C legal-text output represents the correct amount, even when its Arabic tokenization differs from the test labels.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Wrote {prediction_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part-c-predictions", type=Path, default=base_dir / "data" / "part_c_legal_predictions.csv")
    parser.add_argument("--courtesy-predictions", type=Path, default=base_dir / "data" / "part_b_courtesy_predictions.csv")
    parser.add_argument("--part-c-text-metrics", type=Path, default=base_dir / "data" / "part_c_text_metrics.json")
    parser.add_argument("--output-dir", type=Path, default=base_dir / "outputs")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
