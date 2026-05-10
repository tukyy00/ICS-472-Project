"""CTC decoding helpers."""

from __future__ import annotations

DIGITS = "0123456789"
BLANK_INDEX = 0
INDEX_TO_CHAR = {index + 1: char for index, char in enumerate(DIGITS)}


def decode_indices(indices: list[int], blank_index: int = BLANK_INDEX) -> str:
    """Greedy CTC collapse: remove repeats and blanks."""
    output: list[str] = []
    previous: int | None = None
    for index in indices:
        if index != previous and index != blank_index:
            char = INDEX_TO_CHAR.get(index)
            if char is not None:
                output.append(char)
        previous = index
    return "".join(output)


def encode_label(label: str) -> list[int]:
    return [DIGITS.index(char) + 1 for char in label if char in DIGITS]

