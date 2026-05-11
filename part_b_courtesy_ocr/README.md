# Part B Courtesy OCR

This folder is self-contained for courtesy amount recognition. It detects class `1` with `best.pt`, crops the courtesy field, trains a CNN-BiLSTM-CTC recognizer, and reports the required digit-level and amount-level metrics.

Useful commands are listed in the root `README.md`. The default outputs are under this folder's `outputs` directory.

## Deep 16-layer CNN training

The default Part B recognizer is now `deep16`: a 16-convolution CNN followed by a BiLSTM and CTC loss. It preserves more horizontal time steps than the old ResNet-style model, which is useful for digit strings.

```powershell
.\.venv_part_b\Scripts\python.exe -m arabic_check_pipeline.part_b_courtesy_ocr.train --manifest arabic_check_pipeline\part_b_courtesy_ocr\outputs\train_manifest.csv --output-dir arabic_check_pipeline\part_b_courtesy_ocr\outputs\checkpoints_deep16 --arch deep16 --epochs 120 --batch-size 16 --lr 0.0005 --weight-decay 0.0001 --patience 20 --image-height 64 --image-width 384 --dropout 0.15 --num-workers 0 --device cuda
```

Evaluate the resulting checkpoint with:

```powershell
.\.venv_part_b\Scripts\python.exe -m arabic_check_pipeline.part_b_courtesy_ocr.evaluate --manifest arabic_check_pipeline\part_b_courtesy_ocr\outputs\test_manifest.csv --checkpoint arabic_check_pipeline\part_b_courtesy_ocr\outputs\checkpoints_deep16\best_crnn_ctc.pt --output-dir arabic_check_pipeline\part_b_courtesy_ocr\outputs\test_eval_deep16 --batch-size 64 --num-workers 0 --device cuda
```

For beam-search decoding during evaluation, add `--decoder beam --beam-width 10 --beam-topk 11`:

```powershell
.\.venv_part_b\Scripts\python.exe -m arabic_check_pipeline.part_b_courtesy_ocr.evaluate --manifest arabic_check_pipeline\part_b_courtesy_ocr\outputs\test_manifest.csv --checkpoint arabic_check_pipeline\part_b_courtesy_ocr\outputs\checkpoints_deep16\best_crnn_ctc.pt --output-dir arabic_check_pipeline\part_b_courtesy_ocr\outputs\test_eval_deep16_beam --batch-size 64 --num-workers 0 --device cuda --decoder beam --beam-width 10 --beam-topk 11
```
