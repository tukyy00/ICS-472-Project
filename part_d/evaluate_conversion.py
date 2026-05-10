"""Evaluate Part D legal amount conversion on Part B/Part C OCR outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from .converter import convert_legal_amount, normalize_amount_digits, result_to_trace_dict


STAGES = ("direct", "edit_distance", "edit_distance_split", "fused")
LEGAL_STAGE_PRIORITY = {"edit_distance_split": 0, "edit_distance": 1, "direct": 2}


def read_csv_by_id(path: str | Path) -> dict[str, dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
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
            prev_cost, prev_i, prev_d, prev_s = dp[i - 1][j - 1]
            candidates = [
                (prev_cost + substitution_cost, prev_i, prev_d, prev_s + substitution_cost),
                (dp[i][j - 1][0] + 1, dp[i][j - 1][1] + 1, dp[i][j - 1][2], dp[i][j - 1][3]),
                (dp[i - 1][j][0] + 1, dp[i - 1][j][1], dp[i - 1][j][2] + 1, dp[i - 1][j][3]),
            ]
            dp[i][j] = min(candidates, key=lambda item: (item[0], item[1] + item[2], item[3]))
    _, insertions, deletions, substitutions = dp[n][m]
    return insertions, deletions, substitutions


def evaluate_amounts(rows: list[dict[str, str]], stage: str) -> dict[str, float | int]:
    total_digits = 0
    insertions = deletions = substitutions = 0
    exact = one_error = two_or_more = unresolved = 0
    for row in rows:
        reference = normalize_amount_digits(row["courtesy_true"])
        prediction = normalize_amount_digits(row[f"{stage}_value"])
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
        "exact_amount_percent": 100.0 * exact / count if count else 0.0,
        "one_error_amount_percent": 100.0 * one_error / count if count else 0.0,
        "two_or_more_errors_amount_percent": 100.0 * two_or_more / count if count else 0.0,
        "unresolved_count": unresolved,
    }


def choose_fused(candidates: dict[str, str], courtesy_prediction: str) -> tuple[str, str]:
    target = normalize_amount_digits(courtesy_prediction)
    scored: list[tuple[int, int, str, str]] = []
    for stage in ("edit_distance_split", "edit_distance", "direct"):
        value = normalize_amount_digits(candidates.get(stage, ""))
        if not value:
            continue
        edits = sum(edit_counts(target, value))
        scored.append((edits, LEGAL_STAGE_PRIORITY[stage], stage, value))
    if not scored:
        return "", "unresolved"
    _, _, stage, value = min(scored)
    return value, stage


def build_rows(
    courtesy_rows: dict[str, dict[str, str]],
    legal_rows: dict[str, dict[str, str]],
) -> tuple[list[dict[str, str]], list[str], list[str]]:
    overlap = sorted(set(courtesy_rows) & set(legal_rows))
    b_only = sorted(set(courtesy_rows) - set(legal_rows))
    c_only = sorted(set(legal_rows) - set(courtesy_rows))
    output_rows: list[dict[str, str]] = []
    iterator = tqdm(overlap, desc="Part D convert", dynamic_ncols=True) if tqdm else overlap
    for check_id in iterator:
        courtesy = courtesy_rows[check_id]
        legal = legal_rows[check_id]
        legal_prediction = legal["prediction"]
        results = {
            "direct": convert_legal_amount(legal_prediction, "direct"),
            "edit_distance": convert_legal_amount(legal_prediction, "edit_distance"),
            "edit_distance_split": convert_legal_amount(legal_prediction, "edit_distance_split"),
        }
        candidates = {stage: result.value for stage, result in results.items()}
        fused_value, selected_stage = choose_fused(candidates, courtesy["prediction"])
        row = {
            "check_id": check_id,
            "legal_prediction": legal_prediction,
            "courtesy_prediction": normalize_amount_digits(courtesy["prediction"]),
            "courtesy_true": normalize_amount_digits(courtesy["label"]),
            "direct_value": candidates["direct"],
            "edit_distance_value": candidates["edit_distance"],
            "edit_distance_split_value": candidates["edit_distance_split"],
            "fused_value": fused_value,
            "selected_stage": selected_stage,
        }
        output_rows.append(row)
    return output_rows, b_only, c_only


def write_outputs(
    rows: list[dict[str, str]],
    courtesy_rows: dict[str, dict[str, str]],
    legal_rows: dict[str, dict[str, str]],
    b_only: list[str],
    c_only: list[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "conversion_predictions.csv"
    fieldnames = [
        "check_id",
        "legal_prediction",
        "courtesy_prediction",
        "courtesy_true",
        "direct_value",
        "edit_distance_value",
        "edit_distance_split_value",
        "fused_value",
        "selected_stage",
    ]
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metrics = {stage: evaluate_amounts(rows, stage) for stage in STAGES}
    split_exact = {
        row["check_id"]
        for row in rows
        if normalize_amount_digits(row["edit_distance_split_value"]) == normalize_amount_digits(row["courtesy_true"])
    }
    fused_exact = {
        row["check_id"]
        for row in rows
        if normalize_amount_digits(row["fused_value"]) == normalize_amount_digits(row["courtesy_true"])
    }
    metrics["fusion_delta"] = {
        "improved_count": len(fused_exact - split_exact),
        "degraded_count": len(split_exact - fused_exact),
        "unchanged_count": len(rows) - len(fused_exact ^ split_exact),
    }
    metrics["join"] = {
        "part_b_rows": len(courtesy_rows),
        "part_c_rows": len(legal_rows),
        "overlap_rows": len(rows),
        "part_b_only": b_only,
        "part_c_only": c_only,
    }
    (output_dir / "conversion_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (output_dir / "stage_breakdown.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", *metrics["direct"].keys()])
        writer.writeheader()
        for stage in STAGES:
            writer.writerow({"stage": stage, **metrics[stage]})

    with (output_dir / "conversion_trace.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            trace = {}
            for stage, mode in (
                ("direct", "direct"),
                ("edit_distance", "edit_distance"),
                ("edit_distance_split", "edit_distance_split"),
            ):
                result = convert_legal_amount(row["legal_prediction"], mode)
                trace[stage] = result_to_trace_dict(result)
            handle.write(json.dumps({"check_id": row["check_id"], "trace": trace}, ensure_ascii=False) + "\n")

    readme = f"""# Part D Legal Amount Conversion

This folder contains the rule-based legal amount conversion results.

## Inputs

- Part B courtesy predictions: `outputs/part_b/test_eval/predictions.csv`
- Part C legal predictions: `paper15/metrics/test_beam10/legal_crnn_predictions.csv`

## Evaluation Set

- Part B rows: `{len(courtesy_rows)}`
- Part C rows: `{len(legal_rows)}`
- Overlap rows evaluated: `{len(rows)}`
- Part B-only checks: `{', '.join(b_only) if b_only else 'none'}`
- Part C-only checks: `{', '.join(c_only) if c_only else 'none'}`

## Outputs

- `conversion_predictions.csv`: per-check converted values.
- `conversion_metrics.json`: stage-level metrics and fusion delta.
- `conversion_trace.jsonl`: token-level match/edit/split traces.
- `stage_breakdown.csv`: compact stage metrics table.

## Main Result

Fused exact conversion accuracy: `{metrics['fused']['exact_amount_percent']:.2f}%`

The fused value is selected from legal conversion candidates using the Part B courtesy prediction as a guide. It is not a blind courtesy override.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--courtesy-predictions", default="outputs/part_b/test_eval/predictions.csv")
    parser.add_argument("--legal-predictions", default="paper15/metrics/test_beam10/legal_crnn_predictions.csv")
    parser.add_argument("--output-dir", default="outputs/part_d/legal_conversion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    courtesy_rows = read_csv_by_id(args.courtesy_predictions)
    legal_rows = read_csv_by_id(args.legal_predictions)
    rows, b_only, c_only = build_rows(courtesy_rows, legal_rows)
    write_outputs(rows, courtesy_rows, legal_rows, b_only, c_only, Path(args.output_dir))
    print(f"Wrote Part D conversion outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
