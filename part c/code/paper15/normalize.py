"""Arabic legal amount normalization used by the Paper15 run."""

from __future__ import annotations

import re
import unicodedata

BIDI_MARKS = {"\u202a", "\u202b", "\u202c", "\u202d", "\u202e", "\ufeff"}
DIACRITICS_RE = re.compile(r"[\u064b-\u065f\u0670]")
ARABIC_INDIC_DIGITS = str.maketrans(
    {
        "\u0660": "0",
        "\u0661": "1",
        "\u0662": "2",
        "\u0663": "3",
        "\u0664": "4",
        "\u0665": "5",
        "\u0666": "6",
        "\u0667": "7",
        "\u0668": "8",
        "\u0669": "9",
        "\u06f0": "0",
        "\u06f1": "1",
        "\u06f2": "2",
        "\u06f3": "3",
        "\u06f4": "4",
        "\u06f5": "5",
        "\u06f6": "6",
        "\u06f7": "7",
        "\u06f8": "8",
        "\u06f9": "9",
    }
)
CHAR_TRANSLATION = str.maketrans(
    {
        "\u0623": "\u0627",
        "\u0625": "\u0627",
        "\u0622": "\u0627",
        "\u0671": "\u0627",
        "\u0649": "\u064a",
        "\u06cc": "\u064a",
        "\u0626": "\u064a",
        "\u0624": "\u0648",
        "\u0629": "\u0647",
        "\u0640": "",
    }
)


def canonicalize_legal_label(text: str) -> str:
    for mark in BIDI_MARKS:
        text = text.replace(mark, "")
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(ARABIC_INDIC_DIGITS)
    text = DIACRITICS_RE.sub("", text)
    text = text.translate(CHAR_TRANSLATION).replace("\u0621", "")
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_legal_for_scoring(text: str) -> str:
    text = canonicalize_legal_label(text)
    replacements = [
        (r"\bا\s+لا\s+ف\b", "الاف"),
        (r"\bا\s+ل\s+ا\s+ف\b", "الاف"),
        (r"\bا\s+لف\b", "الف"),
        (r"\bا\s+ل\s+ف\b", "الف"),
        (r"\bر\s+يا\s+ل(?:ا)?\b", "ريال"),
        (r"\bر\s+ي\s+ا\s+ل(?:ا)?\b", "ريال"),
        (r"\bثلا\s+ث\b", "ثلاث"),
        (r"\bثما\s+ن\b", "ثمان"),
        (r"\bو\s+ن\b", "ون"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def join_standalone_waw(text: str) -> str:
    tokens = text.split()
    output: list[str] = []
    index = 0
    while index < len(tokens):
        if tokens[index] == "\u0648" and index + 1 < len(tokens):
            output.append("\u0648" + tokens[index + 1])
            index += 2
        else:
            output.append(tokens[index])
            index += 1
    return " ".join(output)

