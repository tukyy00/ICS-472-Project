"""Bonus mutual improvement for courtesy and legal amount predictions.

This script does not train models or run OCR. It reuses the saved Part B and
Part C/Part D CSV outputs and applies a conservative cross-check:

- keep the courtesy amount when courtesy and legal disagree weakly;
- use the legal amount only when all legal conversion stages agree and the
  legal value is exactly the courtesy value with one missing zero restored.

The final mutual amount is written as both the improved courtesy and improved
legal numeric amount, because the two recognizers have been reconciled to one
digit string.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from converter import normalize_amount_digits


LEGAL_STAGE_COLUMNS = {
    "direct": "prediction_direct_value",
    "edit_distance": "prediction_edit_distance_value",
    "edit_distance_split": "prediction_edit_distance_split_value",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def edit_counts(reference: str, prediction: str) -> tuple[int, int, int]:
    """Return insertions, deletions, substitutions from prediction to reference."""

    n, m = len(reference), len(prediction)
    dp = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, j, 0, 0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            substitution_cost = 0 if reference[i - 1] == prediction[j - 1] else 1
            prev_cost, prev_i, prev_d, prev_s = dp[i - 1][j - 1]
            candidates = [
                (prev_cost + substitution_cost, prev_i, prev_d, prev_s + substitution_cost),
                (dp[i][j - 1][0] + 1, dp[i][j - 1][1] + 1, dp[i][j - 1][2], dp[i][j - 1][3]),
                (dp[i - 1][j][0] + 1, dp[i - 1][j][1], dp[i - 1][j][2] + 1, dp[i - 1][j][3]),
            ]
            dp[i][j] = min(candidates, key=lambda item: (item[0], item[1] + item[2], item[3]))
    _, insertions, deletions, substitutions = dp[n][m]
    return insertions, deletions, substitutions


def amount_metrics(rows: list[dict[str, str]], value_column: str) -> dict[str, float | int]:
    total_digits = 0
    insertions = deletions = substitutions = 0
    exact = one_error = two_or_more = unresolved = 0
    for row in rows:
        reference = normalize_amount_digits(row["courtesy_true"])
        prediction = normalize_amount_digits(row[value_column])
        if not prediction:
            unresolved += 1
        ins, dels, subs = edit_counts(reference, prediction)
        edits = ins + dels + subs
        insertions += ins
        deletions += dels
        substitutions += subs
        total_digits += len(reference)
        if edits == 0:
            exact += 1
        elif edits == 1:
            one_error += 1
        else:
            two_or_more += 1

    count = len(rows)
    errors = insertions + deletions + substitutions
    return {
        "amounts_evaluated": count,
        "total_digits": total_digits,
        "insertions": insertions,
        "deletions": deletions,
        "substitutions": substitutions,
        "digit_accuracy_percent": 100.0 * (1.0 - errors / total_digits) if total_digits else 0.0,
        "exact_amount_count": exact,
        "exact_amount_percent": 100.0 * exact / count if count else 0.0,
        "one_error_amount_percent": 100.0 * one_error / count if count else 0.0,
        "two_or_more_errors_amount_percent": 100.0 * two_or_more / count if count else 0.0,
        "unresolved_count": unresolved,
    }


def consensus_value(values: dict[str, str]) -> str:
    normalized = [normalize_amount_digits(value) for value in values.values()]
    if all(normalized) and len(set(normalized)) == 1:
        return normalized[0]
    return ""


def restores_one_missing_zero(courtesy: str, legal: str) -> bool:
    """True when legal is courtesy with exactly one zero inserted."""

    courtesy = normalize_amount_digits(courtesy)
    legal = normalize_amount_digits(legal)
    if not courtesy or not legal or len(legal) != len(courtesy) + 1:
        return False
    for index, char in enumerate(legal):
        if char == "0" and legal[:index] + legal[index + 1 :] == courtesy:
            return True
    return False


def improve_row(row: dict[str, str]) -> dict[str, str]:
    courtesy = normalize_amount_digits(row["courtesy_prediction"])
    true_value = normalize_amount_digits(row["courtesy_true"])
    legal_values = {
        stage: normalize_amount_digits(row[column])
        for stage, column in LEGAL_STAGE_COLUMNS.items()
    }
    legal_split = legal_values["edit_distance_split"]
    legal_consensus = consensus_value(legal_values)

    if courtesy and courtesy == legal_split:
        mutual = courtesy
        decision = "verified_agreement"
        action = "unchanged"
    elif legal_consensus and restores_one_missing_zero(courtesy, legal_consensus):
        mutual = legal_consensus
        decision = "corrected_by_legal_consensus"
        action = "insert_missing_zero"
    elif courtesy:
        mutual = courtesy
        decision = "kept_courtesy"
        action = "unchanged"
    else:
        mutual = legal_consensus or legal_split
        decision = "used_legal_because_courtesy_empty"
        action = "courtesy_empty"

    return {
        "check_id": row["check_id"],
        "courtesy_true": true_value,
        "courtesy_prediction": courtesy,
        "legal_direct_value": legal_values["direct"],
        "legal_edit_distance_value": legal_values["edit_distance"],
        "legal_edit_distance_split_value": legal_split,
        "legal_consensus_value": legal_consensus,
        "mutual_amount": mutual,
        "improved_courtesy_amount": mutual,
        "improved_legal_amount": mutual,
        "decision": decision,
        "correction_action": action,
        "changed_from_courtesy": str(mutual != courtesy).lower(),
        "courtesy_was_correct": str(courtesy == true_value).lower(),
        "legal_split_was_correct": str(legal_split == true_value).lower(),
        "mutual_is_correct": str(mutual == true_value).lower(),
        "part_c_prediction_text": row["part_c_prediction_text"],
    }


def build_metrics(rows: list[dict[str, str]]) -> dict[str, object]:
    courtesy_exact = {row["check_id"] for row in rows if row["courtesy_prediction"] == row["courtesy_true"]}
    legal_exact = {
        row["check_id"]
        for row in rows
        if row["legal_edit_distance_split_value"] == row["courtesy_true"]
    }
    mutual_exact = {row["check_id"] for row in rows if row["mutual_amount"] == row["courtesy_true"]}
    changed = [row for row in rows if row["changed_from_courtesy"] == "true"]
    improved = [row for row in changed if row["courtesy_was_correct"] == "false" and row["mutual_is_correct"] == "true"]
    degraded = [row for row in changed if row["courtesy_was_correct"] == "true" and row["mutual_is_correct"] == "false"]

    return {
        "rule": (
            "Use legal to correct courtesy only when all legal conversion stages "
            "agree and the legal value restores exactly one missing zero."
        ),
        "baseline_courtesy": amount_metrics(rows, "courtesy_prediction"),
        "baseline_legal_edit_distance_split": amount_metrics(rows, "legal_edit_distance_split_value"),
        "mutual_improved_amount": amount_metrics(rows, "mutual_amount"),
        "delta_vs_courtesy": {
            "changed_count": len(changed),
            "improved_count": len(improved),
            "degraded_count": len(degraded),
            "net_exact_gain": len(mutual_exact) - len(courtesy_exact),
            "newly_correct_check_ids": sorted(mutual_exact - courtesy_exact),
            "newly_wrong_check_ids": sorted(courtesy_exact - mutual_exact),
        },
        "delta_vs_legal_split": {
            "net_exact_gain": len(mutual_exact) - len(legal_exact),
            "newly_correct_count": len(mutual_exact - legal_exact),
            "newly_wrong_count": len(legal_exact - mutual_exact),
        },
        "decisions": dict(Counter(row["decision"] for row in rows)),
        "correction_actions": dict(Counter(row["correction_action"] for row in rows)),
    }


def write_summary(path: Path, metrics: dict[str, object]) -> None:
    courtesy = metrics["baseline_courtesy"]
    legal = metrics["baseline_legal_edit_distance_split"]
    mutual = metrics["mutual_improved_amount"]
    delta = metrics["delta_vs_courtesy"]
    path.write_text(
        "\n".join(
            [
                "# Bonus Mutual Improvement",
                "",
                "This run reuses saved OCR/conversion outputs only. It does not train or rerun any model.",
                "",
                "## Rule",
                "",
                str(metrics["rule"]),
                "",
                "## Results",
                "",
                f"- Courtesy baseline exact amount: `{courtesy['exact_amount_count']}` / `{courtesy['amounts_evaluated']}` "
                f"(`{courtesy['exact_amount_percent']:.2f}%`)",
                f"- Legal converted baseline exact amount: `{legal['exact_amount_count']}` / `{legal['amounts_evaluated']}` "
                f"(`{legal['exact_amount_percent']:.2f}%`)",
                f"- Mutual improved exact amount: `{mutual['exact_amount_count']}` / `{mutual['amounts_evaluated']}` "
                f"(`{mutual['exact_amount_percent']:.2f}%`)",
                f"- Mutual improved digit accuracy: `{mutual['digit_accuracy_percent']:.2f}%`",
                f"- Changed from courtesy: `{delta['changed_count']}`",
                f"- Improved over courtesy: `{delta['improved_count']}`",
                f"- Degraded over courtesy: `{delta['degraded_count']}`",
                f"- Net exact gain over courtesy: `+{delta['net_exact_gain']}` checks",
                "",
                "## Corrected Checks",
                "",
                ", ".join(f"`{check_id}`" for check_id in delta["newly_correct_check_ids"]) or "None",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    base_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--converted-predictions",
        type=Path,
        default=base_dir / "outputs" / "part_c_converted_predictions.csv",
        help="Saved converted Part C predictions joined with courtesy predictions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "outputs" / "bonus_mutual_improvement",
    )
    args = parser.parse_args()

    rows = [improve_row(row) for row in read_rows(args.converted_predictions)]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = args.output_dir / "mutual_predictions.csv"
    fieldnames = [
        "check_id",
        "courtesy_true",
        "courtesy_prediction",
        "legal_direct_value",
        "legal_edit_distance_value",
        "legal_edit_distance_split_value",
        "legal_consensus_value",
        "mutual_amount",
        "improved_courtesy_amount",
        "improved_legal_amount",
        "decision",
        "correction_action",
        "changed_from_courtesy",
        "courtesy_was_correct",
        "legal_split_was_correct",
        "mutual_is_correct",
        "part_c_prediction_text",
    ]
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metrics = build_metrics(rows)
    (args.output_dir / "mutual_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_summary(args.output_dir / "summary.md", metrics)

    mutual = metrics["mutual_improved_amount"]
    delta = metrics["delta_vs_courtesy"]
    print(
        f"Mutual exact: {mutual['exact_amount_count']}/{mutual['amounts_evaluated']} "
        f"({mutual['exact_amount_percent']:.2f}%), "
        f"net gain vs courtesy: +{delta['net_exact_gain']} checks"
    )


if __name__ == "__main__":
    main()
