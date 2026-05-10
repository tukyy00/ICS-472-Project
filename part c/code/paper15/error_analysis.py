"""Generate character and word error-analysis tables for Paper15 predictions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from .metrics import evaluate_texts
from .normalize import join_standalone_waw, normalize_legal_for_scoring

AlignmentOp = tuple[str, str, str]


def align(reference: list[str], prediction: list[str]) -> list[AlignmentOp]:
    n, m = len(reference), len(prediction)
    dp = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
    backtrace: list[list[str | None]] = [[None for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        backtrace[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        backtrace[0][j] = "I"

    priority = {"M": 0, "S": 1, "D": 2, "I": 3}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_op = "M" if reference[i - 1] == prediction[j - 1] else "S"
            candidates = [
                (dp[i - 1][j - 1] + (0 if match_op == "M" else 1), match_op),
                (dp[i - 1][j] + 1, "D"),
                (dp[i][j - 1] + 1, "I"),
            ]
            dp[i][j], backtrace[i][j] = min(candidates, key=lambda item: (item[0], priority[item[1]]))

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


def counts_from_ops(ops: list[AlignmentOp]) -> dict[str, int]:
    return {
        "matches": sum(1 for op, _, _ in ops if op == "M"),
        "substitutions": sum(1 for op, _, _ in ops if op == "S"),
        "deletions": sum(1 for op, _, _ in ops if op == "D"),
        "insertions": sum(1 for op, _, _ in ops if op == "I"),
    }


def read_predictions(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def transform_text(text: str, mode: str) -> str:
    if mode == "raw":
        return text
    if mode == "normalized":
        return normalize_legal_for_scoring(text)
    if mode == "normalized_join_waw":
        return join_standalone_waw(normalize_legal_for_scoring(text))
    raise ValueError(f"Unknown analysis mode: {mode}")


def sorted_counter_rows(counter: Counter[tuple[str, str] | str], limit: int | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, count in counter.most_common(limit):
        if isinstance(key, tuple):
            rows.append({"reference": key[0], "prediction": key[1], "count": count})
        else:
            rows.append({"value": key, "count": count})
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze(rows: list[dict[str, str]], mode: str) -> tuple[dict[str, object], dict[str, list[dict[str, object]]]]:
    references: dict[str, str] = {}
    predictions: dict[str, str] = {}
    char_substitutions: Counter[tuple[str, str]] = Counter()
    char_deletions: Counter[str] = Counter()
    char_insertions: Counter[str] = Counter()
    char_confusions: Counter[tuple[str, str]] = Counter()
    word_substitutions: Counter[tuple[str, str]] = Counter()
    word_deletions: Counter[str] = Counter()
    word_insertions: Counter[str] = Counter()
    per_sample: list[dict[str, object]] = []

    for row in rows:
        check_id = row["check_id"]
        reference = transform_text(row.get("label", ""), mode)
        prediction = transform_text(row.get("prediction", ""), mode)
        references[check_id] = reference
        predictions[check_id] = prediction

        char_ops = align(list(reference.replace(" ", "")), list(prediction.replace(" ", "")))
        word_ops = align(reference.split(), prediction.split())
        char_counts = counts_from_ops(char_ops)
        word_counts = counts_from_ops(word_ops)

        for op, ref_value, pred_value in char_ops:
            if op == "M":
                char_confusions[(ref_value, pred_value)] += 1
            elif op == "S":
                char_substitutions[(ref_value, pred_value)] += 1
                char_confusions[(ref_value, pred_value)] += 1
            elif op == "D":
                char_deletions[ref_value] += 1
                char_confusions[(ref_value, "<DEL>")] += 1
            elif op == "I":
                char_insertions[pred_value] += 1
                char_confusions[("<INS>", pred_value)] += 1

        for op, ref_value, pred_value in word_ops:
            if op == "S":
                word_substitutions[(ref_value, pred_value)] += 1
            elif op == "D":
                word_deletions[ref_value] += 1
            elif op == "I":
                word_insertions[pred_value] += 1

        per_sample.append(
            {
                "check_id": check_id,
                "char_errors": char_counts["substitutions"] + char_counts["deletions"] + char_counts["insertions"],
                "char_substitutions": char_counts["substitutions"],
                "char_deletions": char_counts["deletions"],
                "char_insertions": char_counts["insertions"],
                "char_reference_count": len(reference.replace(" ", "")),
                "word_errors": word_counts["substitutions"] + word_counts["deletions"] + word_counts["insertions"],
                "word_substitutions": word_counts["substitutions"],
                "word_deletions": word_counts["deletions"],
                "word_insertions": word_counts["insertions"],
                "word_reference_count": len(reference.split()),
                "reference": reference,
                "prediction": prediction,
            }
        )

    metrics = evaluate_texts(references, predictions)
    char_error_count = metrics["char_errors"]
    word_error_count = metrics["word_errors"]
    summary = {
        "mode": mode,
        "metrics": metrics,
        "character_operations": {
            "substitutions": sum(char_substitutions.values()),
            "deletions": sum(char_deletions.values()),
            "insertions": sum(char_insertions.values()),
            "substitution_share_of_char_errors": _share(sum(char_substitutions.values()), int(char_error_count)),
            "deletion_share_of_char_errors": _share(sum(char_deletions.values()), int(char_error_count)),
            "insertion_share_of_char_errors": _share(sum(char_insertions.values()), int(char_error_count)),
        },
        "word_operations": {
            "substitutions": sum(word_substitutions.values()),
            "deletions": sum(word_deletions.values()),
            "insertions": sum(word_insertions.values()),
            "substitution_share_of_word_errors": _share(sum(word_substitutions.values()), int(word_error_count)),
            "deletion_share_of_word_errors": _share(sum(word_deletions.values()), int(word_error_count)),
            "insertion_share_of_word_errors": _share(sum(word_insertions.values()), int(word_error_count)),
        },
        "top_character_substitutions": sorted_counter_rows(char_substitutions, 20),
        "top_character_deletions": sorted_counter_rows(char_deletions, 20),
        "top_character_insertions": sorted_counter_rows(char_insertions, 20),
        "top_word_substitutions": sorted_counter_rows(word_substitutions, 20),
        "top_word_deletions": sorted_counter_rows(word_deletions, 20),
        "top_word_insertions": sorted_counter_rows(word_insertions, 20),
    }
    tables = {
        "char_confusion_long": sorted_counter_rows(char_confusions, None),
        "char_substitutions": sorted_counter_rows(char_substitutions, None),
        "char_deletions": sorted_counter_rows(char_deletions, None),
        "char_insertions": sorted_counter_rows(char_insertions, None),
        "word_substitutions": sorted_counter_rows(word_substitutions, None),
        "word_deletions": sorted_counter_rows(word_deletions, None),
        "word_insertions": sorted_counter_rows(word_insertions, None),
        "per_sample_errors": sorted(per_sample, key=lambda item: (item["word_errors"], item["char_errors"]), reverse=True),
    }
    return summary, tables


def _share(part: int, total: int) -> float:
    return 0.0 if total == 0 else part / total * 100.0


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_predictions(args.predictions_csv)
    package_summary: dict[str, object] = {}
    for mode in args.modes:
        mode_dir = output_dir / mode
        summary, tables = analyze(rows, mode)
        package_summary[mode] = summary
        mode_dir.mkdir(parents=True, exist_ok=True)
        (mode_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_csv(mode_dir / "char_confusion_long.csv", tables["char_confusion_long"], ["reference", "prediction", "count"])
        write_csv(mode_dir / "char_substitutions.csv", tables["char_substitutions"], ["reference", "prediction", "count"])
        write_csv(mode_dir / "char_deletions.csv", tables["char_deletions"], ["value", "count"])
        write_csv(mode_dir / "char_insertions.csv", tables["char_insertions"], ["value", "count"])
        write_csv(mode_dir / "word_substitutions.csv", tables["word_substitutions"], ["reference", "prediction", "count"])
        write_csv(mode_dir / "word_deletions.csv", tables["word_deletions"], ["value", "count"])
        write_csv(mode_dir / "word_insertions.csv", tables["word_insertions"], ["value", "count"])
        write_csv(
            mode_dir / "per_sample_errors.csv",
            tables["per_sample_errors"],
            [
                "check_id",
                "char_errors",
                "char_substitutions",
                "char_deletions",
                "char_insertions",
                "char_reference_count",
                "word_errors",
                "word_substitutions",
                "word_deletions",
                "word_insertions",
                "word_reference_count",
                "reference",
                "prediction",
            ],
        )
    (output_dir / "error_analysis_summary.json").write_text(
        json.dumps(package_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(package_summary, ensure_ascii=True, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["raw", "normalized", "normalized_join_waw"],
        choices=["raw", "normalized", "normalized_join_waw"],
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
