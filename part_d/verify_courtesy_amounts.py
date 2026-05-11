"""Verify courtesy amounts by matching them against converted legal amounts.

This script reuses outputs/part_c_converted_predictions.csv. It does not rerun
OCR or conversion. For each check:

- legal amount = converted Part C legal text digits
- courtesy amount = Part B courtesy amount digits
- verified = legal amount and courtesy amount match exactly
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "check_id",
        "legal_converted_amount",
        "courtesy_amount",
        "verification_status",
        "matches_ground_truth",
        "courtesy_true",
        "part_c_prediction_text",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]]) -> dict[str, float | int]:
    total = len(rows)
    verified = sum(1 for row in rows if row["verification_status"] == "verified")
    failed = total - verified
    correct_verified = sum(
        1
        for row in rows
        if row["verification_status"] == "verified" and row["matches_ground_truth"] == "true"
    )
    false_verified = verified - correct_verified
    correct_failed = sum(
        1
        for row in rows
        if row["verification_status"] == "failed" and row["matches_ground_truth"] == "false"
    )
    false_failed = failed - correct_failed
    return {
        "checks_evaluated": total,
        "verified_count": verified,
        "failed_count": failed,
        "verified_percent": 0.0 if total == 0 else verified / total * 100.0,
        "failed_percent": 0.0 if total == 0 else failed / total * 100.0,
        "correct_verified_count": correct_verified,
        "false_verified_count": false_verified,
        "correct_failed_count": correct_failed,
        "false_failed_count": false_failed,
    }


def run(args: argparse.Namespace) -> None:
    source_rows = read_rows(args.converted_predictions)
    legal_column = f"prediction_{args.stage}_value"
    if source_rows and legal_column not in source_rows[0]:
        raise ValueError(f"Missing legal conversion column: {legal_column}")

    verification_rows: list[dict[str, str]] = []
    for row in source_rows:
        legal_amount = row[legal_column]
        courtesy_amount = row["courtesy_prediction"]
        verified = bool(legal_amount) and legal_amount == courtesy_amount
        verification_rows.append(
            {
                "check_id": row["check_id"],
                "legal_converted_amount": legal_amount,
                "courtesy_amount": courtesy_amount,
                "verification_status": "verified" if verified else "failed",
                "matches_ground_truth": "true" if courtesy_amount == row["courtesy_true"] else "false",
                "courtesy_true": row["courtesy_true"],
                "part_c_prediction_text": row["part_c_prediction_text"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "courtesy_verification.csv"
    metrics_path = args.output_dir / "courtesy_verification_metrics.json"
    summary_path = args.output_dir / "courtesy_verification_summary.md"

    metrics = {
        "input": str(args.converted_predictions),
        "legal_conversion_stage": args.stage,
        "rule": "verified when converted legal amount exactly matches courtesy amount",
        **summarize(verification_rows),
    }

    write_csv(csv_path, verification_rows)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path.write_text(
        "\n".join(
            [
                "# Courtesy Amount Verification",
                "",
                f"- Legal conversion stage used: `{args.stage}`",
                f"- Checks evaluated: `{metrics['checks_evaluated']}`",
                f"- Verified: `{metrics['verified_count']}` (`{metrics['verified_percent']:.2f}%`)",
                f"- Failed: `{metrics['failed_count']}` (`{metrics['failed_percent']:.2f}%`)",
                f"- Verified and actually correct vs ground truth: `{metrics['correct_verified_count']}`",
                f"- Verified but courtesy was wrong vs ground truth: `{metrics['false_verified_count']}`",
                "",
                "Rule: convert the legal amount to digits, compare it with the courtesy amount digits, and mark the courtesy amount as verified only when both values match exactly.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Wrote {csv_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--converted-predictions",
        type=Path,
        default=base_dir / "outputs" / "part_c_converted_predictions.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=base_dir / "outputs")
    parser.add_argument(
        "--stage",
        choices=("direct", "edit_distance", "edit_distance_split"),
        default="edit_distance_split",
        help="Converted legal amount column to compare against the courtesy amount.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
