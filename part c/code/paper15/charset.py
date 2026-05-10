"""Character vocabulary and CTC decoding for Paper15 legal OCR."""

from __future__ import annotations

import json
from pathlib import Path

BLANK_INDEX = 0


class Charset:
    def __init__(self, chars: list[str]) -> None:
        if " " not in chars:
            chars = [" "] + chars
        self.chars = chars
        self.char_to_index = {char: index + 1 for index, char in enumerate(chars)}
        self.index_to_char = {index + 1: char for index, char in enumerate(chars)}

    @property
    def num_classes(self) -> int:
        return len(self.chars) + 1

    def encode(self, text: str) -> list[int]:
        return [self.char_to_index[char] for char in text if char in self.char_to_index]

    def decode(self, indices: list[int]) -> str:
        output: list[str] = []
        previous: int | None = None
        for index in indices:
            if index != previous and index != BLANK_INDEX:
                char = self.index_to_char.get(index)
                if char is not None:
                    output.append(char)
            previous = index
        return " ".join("".join(output).split())

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"chars": self.chars}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Charset":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(list(data["chars"]))


def build_charset(labels: list[str]) -> Charset:
    return Charset(sorted({char for label in labels for char in label}))

