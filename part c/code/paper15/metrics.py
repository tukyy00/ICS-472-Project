"""CER/WER metrics for Paper15 legal OCR."""

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


def edit_counts(reference: list[str], prediction: list[str]) -> EditCounts:
    n, m = len(reference), len(prediction)
    dp: list[list[tuple[int, int, int, int]]] = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost, ins, dele, sub = dp[i - 1][0]
        dp[i][0] = (cost + 1, ins, dele + 1, sub)
    for j in range(1, m + 1):
        cost, ins, dele, sub = dp[0][j - 1]
        dp[0][j] = (cost + 1, ins + 1, dele, sub)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates = []
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


def evaluate_texts(references: dict[str, str], predictions: dict[str, str]) -> dict[str, float | int]:
    ids = sorted(set(references) & set(predictions))
    char_ref_count = char_errors = word_ref_count = word_errors = exact = 0
    for check_id in ids:
        ref = references[check_id]
        pred = predictions[check_id]
        char_counts = edit_counts(list(ref.replace(" ", "")), list(pred.replace(" ", "")))
        word_counts = edit_counts(ref.split(), pred.split())
        char_ref_count += len(ref.replace(" ", ""))
        char_errors += char_counts.total
        word_ref_count += len(ref.split())
        word_errors += word_counts.total
        exact += int(ref == pred)
    return {
        "amounts_evaluated": len(ids),
        "cer_percent": 0.0 if char_ref_count == 0 else char_errors / char_ref_count * 100.0,
        "wer_percent": 0.0 if word_ref_count == 0 else word_errors / word_ref_count * 100.0,
        "exact_text_percent": 0.0 if not ids else exact / len(ids) * 100.0,
        "char_errors": char_errors,
        "char_reference_count": char_ref_count,
        "word_errors": word_errors,
        "word_reference_count": word_ref_count,
    }

