# Part D Self-Contained Conversion Check

This folder reruns the Part D conversion idea directly on the Part C legal OCR output, then recalculates the effective WER against the numeric check amount.

## Contents

- `converter.py`: local copy of the rule-based Arabic legal amount converter.
- `recalculate_part_c_wer.py`: standalone runner for the converted Part C check.
- `verify_courtesy_amounts.py`: verification stage that reuses converted legal digits and checks whether the courtesy amount matches.
- `data/part_c_legal_predictions.csv`: Part C test legal-text predictions.
- `data/part_c_text_metrics.json`: original Part C text CER/WER metrics.
- `data/part_b_courtesy_predictions.csv`: numeric ground truth labels and Part B courtesy predictions copied from `arabic_check_pipeline/part_b_courtesy_ocr/outputs/test_eval_deep16_beam/predictions.csv`.
- `outputs/`: generated converted predictions, metrics, and summary.

## Run

From this folder:

```powershell
python .\recalculate_part_c_wer.py
```

From the project root:

```powershell
python .\part_d_self_contained\recalculate_part_c_wer.py
```

Then run the verification stage from the project root:

```powershell
python .\part_d_self_contained\verify_courtesy_amounts.py
```

This reads `outputs/part_c_converted_predictions.csv`; it does not rerun OCR or conversion.

## Why This Check Is Fair

The original Part C WER compares Arabic legal amount text word by word. That is very sensitive to tokenization differences in the provided test labels, for example split Arabic word pieces versus natural full words.

This rerun converts the Part C legal-text prediction into a numeric amount first. The converted WER then treats the full amount as one word:

- correct amount: `0` word errors
- wrong amount: `1` word error

The output also reports digit CER, which is useful for near misses such as one missing digit.

## Courtesy Verification Rule

For each check:

1. Use the converted legal amount digits from Part C.
2. Use the courtesy amount digits from Part B.
3. If they match exactly, mark the courtesy amount as `verified`.
4. If they do not match, mark the verification as `failed`.

The generated files are:

- `outputs/courtesy_verification.csv`
- `outputs/courtesy_verification_metrics.json`
- `outputs/courtesy_verification_summary.md`
