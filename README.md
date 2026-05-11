# Arabic Bank Check Amount Extraction and Processing

This project implements an end-to-end system for processing Arabic bank checks. The system detects the two amount regions on a check image, recognizes the courtesy amount written as digits, recognizes the legal amount written in Arabic text, converts the legal text into a numeric value, and verifies whether both amounts match.

## Project Overview

Arabic bank checks usually contain two representations of the amount:

- **Courtesy amount**: the amount written as digits.
- **Legal amount**: the amount written as Arabic text.

The goal of this project is to automatically extract and process both fields from check images.

The pipeline is divided into four parts:

| Part | Task | Description |
|---|---|---|
| Part A | Amount region detection | Detect the legal amount and courtesy amount bounding boxes. |
| Part B | Courtesy amount recognition | Recognize the digit sequence from the courtesy amount crop. |
| Part C | Legal amount recognition | Recognize the Arabic text from the legal amount crop. |
| Part D | Legal conversion and verification | Convert Arabic legal text into digits and compare it with the courtesy amount. |

> **Note:** The dataset is licensed and not public. Dataset examples are treated as course-internal material only.

## Dataset Summary

### Detection Dataset

- Source images: **1,800**
- Matched image-label pairs: **1,799**
- Missing label: `ac00048`
- Training checks: **1,529**
- Validation checks: **270**
- Object classes:
  - `legal_amount`
  - `courtesy_amount`

### Courtesy Recognition Dataset

- Training rows: **1,439**
- Validation rows: **180**
- Internal-test rows: **180**
- Provided test checks: **600**

### Legal Recognition Dataset

- Training rows: **1,428**
- Validation rows: **179**
- Internal-test rows: **179**
- Provided test checks: **598**

Part D evaluates the **598-check overlap** between Part B and Part C. Two Part B rows, `ac03157` and `ac03397`, do not have matching Part C predictions.

## Methodology

## Part A: Amount Region Detection

Part A uses a **YOLOv8** detector to localize the two amount regions.

Main training configuration:

| Parameter | Value |
|---|---|
| Model | `yolov8s.pt` |
| Image size | 640 |
| Batch size | 16 |
| Optimizer | AdamW |
| Initial learning rate | 0.001 |
| Momentum | 0.9 |
| Weight decay | 0.0005 |
| Seed | 42 |
| Epochs | 100 |

The detection output format is:

```text
{check file name} {courtesy amount box} {legal amount box}
```

## Part B: Courtesy Amount Recognition

Part B uses a **CTC-based sequence recognizer** for courtesy amount digits.

Main model components:

- Grayscale ResNet18-style CNN feature extractor.
- 2-layer bidirectional LSTM.
- Hidden size: **256**.
- Dropout: **0.1**.
- Linear classifier over digit classes and the CTC blank.
- Input image size: **64 × 384**.
- Labels are normalized to digits only.

A stronger courtesy-recognition experiment also used:

- YOLO-generated courtesy crops only.
- Deep-16 CNN backbone with batch normalization and ReLU.
- Rectangular pooling to preserve horizontal sequence information.
- 4-layer bidirectional LSTM.
- Hidden size: **512**.
- CTC loss.
- AdamW optimizer.
- Learning rate: `1e-3`.
- Weight decay: `1e-4`.
- Up to 80 epochs.
- Early stopping with patience 12.
- Augmentations such as small rotations, translations, brightness shifts, contrast shifts, and random horizontal placement.

Digit accuracy is computed as:

```text
Accuracy (%) = (1 - (I + D + S) / N) × 100
```

where:

- `I` = insertions
- `D` = deletions
- `S` = substitutions
- `N` = total number of reference digits

## Part C: Legal Amount Recognition

Part C uses a **Paper15-style CRNN** for Arabic legal amount text recognition.

Main model components:

- Convolutional feature extractor.
- Adaptive pooling to sequence length 96.
- 2-layer bidirectional LSTM.
- Hidden size: **384**.
- Dropout: **0.1**.
- CTC classifier.
- Input image size: **140 × 1340**.
- Character set includes Arabic letters, digits, space, and the CTC blank.
- Total classes: **38**.

The main metrics are:

```text
CER/WER (%) = ((I + D + S) / N) × 100
```

where CER is character error rate and WER is word error rate.

## Part D: Legal Conversion and Verification

Part D converts the Arabic legal prediction into digits using a rule-based converter.

The converter uses:

- Direct matching.
- Edit-distance recovery.
- Edit-distance recovery with split handling.
- Fusion guided by the courtesy recognizer.

A check succeeds when the converted legal amount matches the courtesy amount.

## Results

## Part A Results: YOLO Detection

Final validation metrics from `outputs/part_a/results.csv`:

| Metric | Value |
|---|---:|
| Precision | 86.20% |
| Recall | 87.78% |
| mAP@50 | 88.93% |
| mAP@50:95 | 55.77% |

The repository includes YOLO validation artifacts such as:

- Training curves.
- Precision-recall curves.
- Confusion matrices.
- Validation label mosaics.
- Validation prediction mosaics.

The assignment-specific IoU-threshold output for IoU 50, 75, and 90 was not saved among the project artifacts, although the evaluator exists at:

```text
part_a/scripts/evaluate_extraction.py
```

## Part B Results: Courtesy Recognition

Saved test evaluation from `outputs/part_b/test_eval/metrics.json`:

| Metric | Value |
|---|---:|
| Amounts evaluated | 600 |
| Total reference digits | 2,606 |
| Insertions | 38 |
| Deletions | 83 |
| Substitutions | 56 |
| Digit accuracy | 93.21% |
| Amounts with no errors | 79.83% |
| Amounts with one error | 15.83% |
| Amounts with two or more errors | 4.33% |

A stronger supplementary courtesy-recognition run reported:

| Metric | Value |
|---|---:|
| Digit-level accuracy | 97.24% |
| Exact amount match | 90.33% |
| One-error amounts | 8.33% |
| Two-or-more-error amounts | 1.33% |

The 600-sample saved result is treated as the main reproducible score.

### Example Part B Predictions

| Check ID | Ground Truth | Prediction | Result |
|---|---:|---:|---|
| `ac03000` | 10000 | 10000 | Correct |
| `ac03001` | 5000 | 5000 | Correct |
| `ac03005` | 1733587 | 17387 | Error |
| `ac03007` | 12000 | 12000 | Correct |
| `ac03014` | 2800 | 2800 | Correct |

## Part C Results: Legal Recognition

Legal recognition results from `part c/metrics_summary.json`:

| Evaluation | Amounts | CER | WER | Exact Text |
|---|---:|---:|---:|---:|
| Validation greedy | 179 | 15.55% | 39.66% | 22.91% |
| Validation beam10 | 179 | 14.68% | 37.22% | 24.58% |
| Provided test greedy, raw | 598 | 5.36% | 81.65% | 0.00% |
| Provided test greedy, normalized | 598 | 5.36% | 54.83% | 9.70% |
| Provided test beam10, raw | 598 | 5.01% | 81.65% | 0.00% |
| Provided test beam10, normalized | 598 | 5.01% | 54.98% | 10.03% |

Beam search with width 10 improves validation WER and provided-test CER.

### Join-Waw Normalized Analysis

| Evaluation | CER | WER | Exact Text | Character Operations S/D/I |
|---|---:|---:|---:|---:|
| Validation beam10, raw | 14.68% | 37.22% | 24.58% | 171/512/77 |
| Validation beam10, normalized join-waw | 14.68% | 27.71% | 30.17% | 171/512/77 |
| Test beam10, raw | 5.01% | 81.65% | 0.00% | 130/438/160 |
| Test beam10, normalized join-waw | 4.90% | 52.83% | 7.53% | 130/405/175 |

CER is more stable than WER because the provided test labels use unusual token spacing, while the model often predicts normalized Arabic words.

## Part D Results: Legal Conversion and Verification

Metrics from `outputs/part_d/legal_conversion/conversion_metrics.json`:

| Stage | Digit Accuracy | Exact | One Error | Two or More Errors | I/D/S | Unresolved |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 88.34% | 71.74% | 15.89% | 12.37% | 2/151/150 | 16 |
| Edit distance | 91.11% | 78.26% | 11.20% | 10.54% | 5/86/140 | 4 |
| Edit distance split | 91.19% | 78.43% | 11.20% | 10.37% | 5/84/140 | 4 |
| Fused | 91.99% | 78.43% | 13.21% | 8.36% | 3/86/119 | 4 |

The fused stage evaluated:

- Amounts: **598**
- Reference digits: **2,598**
- Improved checks: **2**
- Degraded checks: **2**
- Unchanged checks: **594**

### Example Part D Predictions

| Check ID | Courtesy Prediction | True Amount | Fused Amount | Result |
|---|---:|---:|---:|---|
| `ac03000` | 10000 | 10000 | 10000 | Correct |
| `ac03001` | 5000 | 5000 | 5000 | Correct |
| `ac03005` | 17387 | 1733587 | 17395 | Error |
| `ac03007` | 12000 | 12000 | 2000 | Error |
| `ac03008` | 450 | 450 | 450 | Correct |

## Error Analysis

### Detection Errors

Part A performs well at localizing the two amount regions, but some courtesy amount annotations are larger than the tight digit regions predicted by the model. This creates an annotation-scale mismatch:

- The detector may correctly locate the visible digits.
- The predicted box may still receive a lower IoU score if the ground-truth box includes more whitespace or the full printed amount field.
- This issue becomes more severe at strict IoU thresholds such as 0.75 and 0.90.

This explains why mAP@50 is much higher than mAP@50:95.

### Courtesy Recognition Errors

Part B achieves high digit accuracy, but most digit errors are deletions.

Main error patterns:

| Error Type | Frequent Cases | Interpretation |
|---|---|---|
| Deletion | `0` deleted 46 times; `1` deleted 9 times | Missing digits are dominated by zeros. |
| Substitution | `0→1`, `0→2`, `0→5` | Weak marks and unclear zeros confuse the CTC decoder. |
| Insertion | `0` inserted 17 times; `1` inserted 6 times | Extra low-confidence strokes may be decoded as digits. |

The error in `ac03005`, where `1733587` was predicted as `17387`, is especially important because the wrong prediction is still a valid number.

### Legal Recognition Errors

Part C has low CER on the provided test set, but WER is high due to tokenization mismatch. The model often predicts natural Arabic amount phrases, while the labels may split words into smaller fragments.

Common character-level issues include:

- Deletion of frequent Arabic letters such as `ا`, `ي`, and `ه`.
- Insertions of letters common in words such as `ريال` and `الف`.
- Substitutions between visually or contextually similar characters.

Because of this, CER is a fairer metric than raw WER for the provided test labels.

### Legal Conversion Errors

Part D improves digit accuracy, but it is sensitive to missing or corrupted number words. For example, if a word such as `عشر` is missing, the parser may convert the amount to a completely different magnitude.

The fused method is conservative: it mostly keeps predictions unchanged, with only a small number of improved or degraded cases.

## Main Results Summary

| Component | Best Saved Result |
|---|---:|
| Part A detection | 88.93% mAP@50 |
| Part B courtesy recognition | 93.21% digit accuracy |
| Part C legal recognition | 5.01% CER with beam search |
| Part D legal conversion | 78.43% exact fused conversion accuracy |
| Part D fused digit accuracy | 91.99% |

## Limitations

The main limitations are:

- The Part A exact IoU-threshold output was not saved with the project artifacts.
- Some courtesy amount annotations are wider than the tight predicted digit boxes.
- Legal amount labels use unusual spacing and tokenization, which inflates WER.
- The legal-to-digit parser still struggles with missing or misspelled Arabic number words.
- Fusion helps, but it cannot fully recover from major legal recognition mistakes.

## Future Work

Possible improvements include:

- Save all Part A IoU evaluator outputs with the submitted artifacts.
- Standardize the annotation policy for courtesy amount boxes.
- Train recognizers with detection errors included in the pipeline.
- Improve Arabic amount parsing for thousands, hundreds, tens, and conjunctions with `و`.
- Add confidence-aware alternatives for common Arabic amount structures.
- Use stronger fusion between legal and courtesy predictions.

## Repository Structure

A possible repository organization is:

```text
.
├── part_a/
│   └── scripts/
│       └── evaluate_extraction.py
├── part_b/
├── part_c/
├── outputs/
│   ├── part_a/
│   ├── part_b/
│   └── part_d/
├── project_report.tex
└── README.md
```

## Requirements

The project uses deep learning and computer vision tools such as:

- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Pandas
- Matplotlib

Exact package versions should be added based on the final project environment.

## How to Run

The exact run commands depend on the repository scripts. A general workflow is:

```bash
# 1. Train or run YOLO detection for Part A
python part_a/scripts/train_yolo.py

# 2. Evaluate amount region extraction
python part_a/scripts/evaluate_extraction.py

# 3. Run courtesy amount recognition
python part_b/scripts/evaluate.py

# 4. Run legal amount recognition
python part_c/scripts/evaluate.py

# 5. Run legal conversion and verification
python part_d/scripts/convert_and_verify.py
```

Update these commands according to the actual script names in the repository.

## Conclusion

This project builds a complete Arabic bank check amount processing pipeline. The recognition models perform strongly, especially for courtesy digits and legal text CER. The most challenging part remains robust conversion of noisy Arabic legal amount predictions into exact numeric amounts. Better annotation consistency, stronger Arabic parsing, and confidence-aware fusion are the most important directions for improvement.
