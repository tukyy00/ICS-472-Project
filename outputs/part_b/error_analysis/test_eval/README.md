# Part B Courtesy Amount Error Analysis

This folder contains digit-level error analysis for the Part B courtesy amount recognizer on `outputs/part_b/test_eval/predictions.csv`.

## Main Metrics

- Amounts evaluated: `600`
- Digit accuracy: `93.21%`
- Exact amount match: `79.83%`
- One-edit amount rate: `15.83%`
- Two-or-more-edit amount rate: `4.33%`
- Digit substitutions: `56`
- Digit deletions: `83`
- Digit insertions: `38`

## Files

- `summary.json`: metrics, error-operation shares, and top digit errors.
- `digit_confusion_long.csv`: long confusion table, including `<DEL>` and `<INS>`.
- `digit_substitutions.csv`: digit substitution counts.
- `digit_deletions.csv`: deleted reference-digit counts.
- `digit_insertions.csv`: inserted prediction-digit counts.
- `per_sample_errors.csv`: per-check edit counts and exact-match flag.
- `images/digit_confusion_matrix.png`: report-ready confusion matrix with deletion column and insertion row.
- `images/digit_operation_counts.png`: operation-count bar chart.

## Regeneration Command

```powershell
$env:PYTHONPATH="."
.\.venv_part_b\Scripts\python.exe -m part_b.error_analysis --predictions-csv outputs\part_b\test_eval\predictions.csv --output-dir outputs\part_b\error_analysis\test_eval
```
