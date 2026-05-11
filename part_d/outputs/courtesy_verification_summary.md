# Courtesy Amount Verification

- Legal conversion stage used: `edit_distance_split`
- Checks evaluated: `598`
- Verified: `426` (`71.24%`)
- Failed: `172` (`28.76%`)
- Verified and actually correct vs ground truth: `425`
- Verified but courtesy was wrong vs ground truth: `1`

Rule: convert the legal amount to digits, compare it with the courtesy amount digits, and mark the courtesy amount as verified only when both values match exactly.
