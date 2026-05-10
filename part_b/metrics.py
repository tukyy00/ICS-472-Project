"""Metrics required for Part B."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EditCounts:
    insertions: int
    deletions: int
    substitutions: int

    @property
    def total(self) -> int:
        return self.insertions + self.deletions + self.substitutions


def edit_counts(reference: str, prediction: str) -> EditCounts:
    """Return insertion/deletion/substitution counts via Levenshtein DP."""
    n, m = len(reference), len(prediction)
    dp: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)
    ]
    for i in range(1, n + 1):
        cost, ins, dele, sub = dp[i - 1][0]
        dp[i][0] = (cost + 1, ins, dele + 1, sub)
    for j in range(1, m + 1):
        cost, ins, dele, sub = dp[0][j - 1]
        dp[0][j] = (cost + 1, ins + 1, dele, sub)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates: list[tuple[int, int, int, int]] = []
            if reference[i - 1] == prediction[j - 1]:
                candidates.append(dp[i - 1][j - 1])
            else:
                cost, ins, dele, sub = dp[i - 1][j - 1]
                candidates.append((cost + 1, ins, dele, sub + 1))
            cost, ins, dele, sub = dp[i][j - 1]
            candidates.append((cost + 1, ins + 1, dele, sub))
            cost, ins, dele, sub = dp[i - 1][j]
            candidates.append((cost + 1, ins, dele + 1, sub))
            dp[i][j] = min(candidates, key=lambda item: (item[0], item[3], item[1], item[2]))

    _, insertions, deletions, substitutions = dp[n][m]
    return EditCounts(insertions, deletions, substitutions)


def evaluate_amounts(references: dict[str, str], predictions: dict[str, str]) -> dict[str, float | int]:
    """Compute project metrics for matching check ids."""
    ids = sorted(set(references) & set(predictions))
    total_digits = 0
    total_insertions = 0
    total_deletions = 0
    total_substitutions = 0
    exact = 0
    one_error = 0
    two_or_more = 0

    for check_id in ids:
        ref = references[check_id]
        pred = predictions[check_id]
        counts = edit_counts(ref, pred)
        total_digits += len(ref)
        total_insertions += counts.insertions
        total_deletions += counts.deletions
        total_substitutions += counts.substitutions
        if counts.total == 0:
            exact += 1
        elif counts.total == 1:
            one_error += 1
        else:
            two_or_more += 1

    total_errors = total_insertions + total_deletions + total_substitutions
    amount_count = len(ids)
    digit_accuracy = 0.0 if total_digits == 0 else (1.0 - total_errors / total_digits) * 100.0
    return {
        "amounts_evaluated": amount_count,
        "total_digits": total_digits,
        "insertions": total_insertions,
        "deletions": total_deletions,
        "substitutions": total_substitutions,
        "digit_accuracy_percent": digit_accuracy,
        "exact_amount_percent": _pct(exact, amount_count),
        "one_error_amount_percent": _pct(one_error, amount_count),
        "two_or_more_errors_amount_percent": _pct(two_or_more, amount_count),
    }


def _pct(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator * 100.0

