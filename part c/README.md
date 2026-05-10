# Part C Paper15 Legal Amount CRNN Package

This package contains only the new `paper15` Part C legal amount recognition model and its evaluation artifacts.

## Model

- Architecture: paper-style 15-convolution CNN + 2-layer BiLSTM + CTC
- Input: grayscale legal amount crop resized to `1340 x 140`
- Preprocessing: horizontal image flip, exact resize, pixel normalization to `[-1, 1]`
- Output sequence length: `96`
- Label mode: canonical Arabic legal text
- Classes: `38`, including CTC blank
- Best checkpoint: epoch `25`, selected by validation loss
- Training reached: epoch `45` before interruption

## Contents

- `model/best_legal_crnn_ctc.pt`: PyTorch checkpoint
- `model/charset.json`: character set for CTC decoding
- `model/run_config.json`: training configuration
- `training/history.json`: training and validation history
- `training/train_split.csv`: training split
- `training/val_split.csv`: validation split
- `training/internal_test_split.csv`: internal held-out split
- `metrics/validation_greedy`: validation predictions and metrics using greedy CTC decode
- `metrics/validation_beam10`: validation predictions and metrics using beam search
- `metrics/test_greedy`: provided test predictions and metrics using greedy CTC decode
- `metrics/test_beam10`: provided test predictions and metrics using beam search
- `metrics/error_analysis`: character and word confusion/error tables for beam-search outputs
- `metrics/error_analysis/images`: report-ready PNG plots for the error-analysis metrics
- `code/paper15`: standalone Paper15-only source code
- `code/requirements-part-c.txt`: Python dependencies for the Part C code
- `metrics_summary.json`: compact report-ready metrics

## Reproduction Commands

Evaluate validation with beam search:

```powershell
$env:PYTHONPATH="paper15\code"
.\.venv_part_b\Scripts\python.exe -m paper15.evaluate --manifest paper15\training\val_split.csv --checkpoint paper15\model\best_legal_crnn_ctc.pt --charset paper15\model\charset.json --output-dir paper15\metrics\validation_beam10_rerun --batch-size 16 --device cuda --decoder beam --beam-width 10 --beam-topk 16
```

Train the same configuration from the project root:

```powershell
$env:PYTHONPATH="paper15\code"
.\.venv_part_b\Scripts\python.exe -m paper15.train --manifest outputs\part_c\train_legal_manifest.csv --output-dir outputs\part_c\legal_crnn_paper15_rerun --epochs 500 --patience 20 --batch-size 16 --lr 0.0001 --device cuda
```

Run error analysis on a prediction CSV:

```powershell
$env:PYTHONPATH="paper15\code"
.\.venv_part_b\Scripts\python.exe -m paper15.error_analysis --predictions-csv paper15\metrics\validation_beam10\predictions.csv --output-dir paper15\metrics\error_analysis\validation_beam10
```

Create report-ready error-analysis images:

```powershell
$env:PYTHONPATH="paper15\code"
.\.venv_part_b\Scripts\python.exe -m paper15.plot_error_analysis --error-dir paper15\metrics\error_analysis --output-dir paper15\metrics\error_analysis\images
```

Create the paper-style full confusion matrix with insertion/deletion margins:

```powershell
$env:PYTHONPATH="paper15\code"
.\.venv_part_b\Scripts\python.exe -m paper15.plot_paper_confusion --confusion-csv paper15\metrics\error_analysis\validation_beam10\normalized_join_waw\char_confusion_long.csv --output-png paper15\metrics\error_analysis\images\paper_style_confusion_matrix_validation.png --title "Confusion Matrix" --caption "Figure: Confusion matrix for character recognition using the Paper15 CNN-BiLSTM-CTC model, including insertions and deletions."
```

## Main Results

Validation greedy:

- CER: `15.55%`
- WER: `39.66%`
- Exact text: `22.91%`

Validation beam search:

- CER: `14.68%`
- WER: `37.22%`
- Exact text: `24.58%`
- WER with fair standalone Arabic waw boundary normalization: about `27.71%`

Provided test beam search:

- CER: `5.01%`
- Raw WER: `81.65%`
- Normalized WER: `54.98%`
- Normalized WER with fair standalone Arabic waw boundary handling: about `47.25%`

The provided test WER is inflated because the test labels are tokenized differently from the natural/raw legal labels used for training and validation. CER is therefore the more stable metric on the provided test set.

## Error Analysis Outputs

The error-analysis tables are saved for `validation_beam10` and `test_beam10`, each with `raw`, `normalized`, and `normalized_join_waw` scoring modes.

Each mode contains:

- `summary.json`: CER/WER plus edit-operation counts and top errors
- `char_confusion_long.csv`: character confusion table with `<DEL>` and `<INS>` rows
- `char_substitutions.csv`: character substitution counts
- `char_deletions.csv`: deleted reference-character counts
- `char_insertions.csv`: inserted prediction-character counts
- `word_substitutions.csv`: word substitution counts
- `word_deletions.csv`: deleted reference-word counts
- `word_insertions.csv`: inserted prediction-word counts
- `per_sample_errors.csv`: per-check CER/WER, exact-match flag, and edit counts

The `images` folder contains:

- `cer_wer_comparison.png`
- `character_operation_counts.png`
- `word_operation_counts.png`
- `validation_beam10/top_character_substitutions.png`
- `validation_beam10/top_character_deletions.png`
- `validation_beam10/top_character_insertions.png`
- `validation_beam10/character_substitution_heatmap.png`
- `paper_style_confusion_matrix_validation.png`
- `test_beam10/top_character_substitutions.png`
- `test_beam10/top_character_deletions.png`
- `test_beam10/top_character_insertions.png`
- `test_beam10/character_substitution_heatmap.png`

Key validation beam-search error profile:

- Raw/normalized CER: `14.68%`
- Raw WER: `37.22%`
- `normalized_join_waw` WER: `27.71%`
- Character errors: `171` substitutions, `512` deletions, `77` insertions
- Word errors after `normalized_join_waw`: `245` substitutions, `77` deletions, `15` insertions

Key provided-test beam-search error profile:

- Normalized CER: `4.90%`
- Raw WER: `81.65%`
- `normalized_join_waw` WER: `52.83%`
- Character errors after normalization: `130` substitutions, `405` deletions, `175` insertions
- Word errors after `normalized_join_waw`: `1175` substitutions, `1503` deletions, `5` insertions
