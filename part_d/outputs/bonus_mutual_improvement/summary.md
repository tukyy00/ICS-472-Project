# Bonus Mutual Improvement

This run reuses saved OCR/conversion outputs only. It does not train or rerun any model.

## Rule

Use legal to correct courtesy only when all legal conversion stages agree and the legal value restores exactly one missing zero.

## Results

- Courtesy baseline exact amount: `540` / `598` (`90.30%`)
- Legal converted baseline exact amount: `469` / `598` (`78.43%`)
- Mutual improved exact amount: `551` / `598` (`92.14%`)
- Mutual improved digit accuracy: `97.65%`
- Changed from courtesy: `11`
- Improved over courtesy: `11`
- Degraded over courtesy: `0`
- Net exact gain over courtesy: `+11` checks

## Corrected Checks

`ac03001`, `ac03046`, `ac03204`, `ac03227`, `ac03238`, `ac03693`, `ac03866`, `ac03888`, `ac04022`, `ac04023`, `ac04185`
