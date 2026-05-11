"""Label parsing utilities for courtesy amount recognition.

The project label files are not perfectly uniform:
- some names are prefixed with ``C`` (``Cac00000.tif``);
- a few use ``.tiff`` and omit a leading zero (``Cac1311.tiff``);
- tokenized labels include start/end markers and separator tokens.

This module normalizes those cases into ``acNNNNN -> digit string`` pairs.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

CHECK_ID_RE = re.compile(r"c?(ac)(\d+)", re.IGNORECASE)
DROP_TOKENS = {"10", "11", ".", "/", ",", " "}


def normalize_check_id(value: str | Path) -> str:
    """Return a canonical check id like ``ac00042`` from a label/image name."""
    name = Path(str(value)).name
    match = CHECK_ID_RE.search(name)
    if not match:
        raise ValueError(f"Could not extract check id from {value!r}")
    number = int(match.group(2))
    return f"ac{number:05d}"


def normalize_label_tokens(token_text: str) -> str:
    """Convert tokenized courtesy amount text to digits only.

    Examples:
        ``[10, 6, 2, 4, 2, '.', 6, 0, 10]`` -> ``624260``
        ``[10, 1, 7, 11, 3, 3, 5, 11, 8, 7, 10]`` -> ``1733587``
    """
    text = token_text.strip()
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = re.findall(r"'[^']*'|\d+|[./,]", text)

    digits: list[str] = []
    for token in parsed:
        value = str(token).strip().strip("'\"")
        if value in DROP_TOKENS:
            continue
        if value.isdigit() and len(value) == 1:
            digits.append(value)
    return "".join(digits)


def parse_label_line(line: str) -> tuple[str, str] | None:
    """Parse one label line into ``(check_id, digits)``.

    Blank lines and malformed lines without a target return ``None``.
    """
    stripped = line.strip()
    if not stripped:
        return None
    if "\t" in stripped:
        raw_name, token_text = stripped.split("\t", 1)
    else:
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2:
            return None
        raw_name, token_text = parts
    check_id = normalize_check_id(raw_name)
    return check_id, normalize_label_tokens(token_text)


def load_labels_from_file(path: str | Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parsed = parse_label_line(line)
        if parsed is None:
            continue
        check_id, digits = parsed
        labels[check_id] = digits
    return labels


def load_labels_from_dir(path: str | Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for file_path in sorted(Path(path).glob("*.txt")):
        labels.update(load_labels_from_file(file_path))
    return labels

