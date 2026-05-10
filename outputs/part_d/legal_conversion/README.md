# Part D Legal Amount Conversion

This folder contains the rule-based legal amount conversion results.

## Inputs

- Part B courtesy predictions: `outputs/part_b/test_eval/predictions.csv`
- Part C legal predictions: `paper15/metrics/test_beam10/legal_crnn_predictions.csv`

## Evaluation Set

- Part B rows: `600`
- Part C rows: `598`
- Overlap rows evaluated: `598`
- Part B-only checks: `ac03157, ac03397`
- Part C-only checks: `none`

## Outputs

- `conversion_predictions.csv`: per-check converted values.
- `conversion_metrics.json`: stage-level metrics and fusion delta.
- `conversion_trace.jsonl`: token-level match/edit/split traces.
- `stage_breakdown.csv`: compact stage metrics table.

## Main Result

Fused exact conversion accuracy: `78.43%`

The fused value is selected from legal conversion candidates using the Part B courtesy prediction as a guide. It is not a blind courtesy override.
