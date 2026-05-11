# Converted Part C WER Check

- Overlap rows evaluated: `598`
- Original Part C normalized text WER: `54.97875484943654`
- Converted Part C amount WER: `21.57%`
- Converted Part C exact amount accuracy: `78.43%`
- Converted Part C digit CER: `8.81%`
- Converted label consistency exact accuracy: `22.07%`

The converted WER treats the full numeric amount as one word. This checks whether the Part C legal-text output represents the correct amount, even when its Arabic tokenization differs from the test labels.
