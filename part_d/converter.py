"""Rule-based Arabic legal amount conversion for Part D."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable


BIDI_MARKS = {"\u202a", "\u202b", "\u202c", "\u202d", "\u202e", "\ufeff"}
DIACRITICS_RE = re.compile(r"[\u064b-\u065f\u0670]")
ARABIC_DIGITS = str.maketrans(
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
ARABIC_CHARS = str.maketrans(
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


IGNORE_WORDS = {
    "فقط",
    "لا",
    "غير",
    "غيرها",
    "وغير",
    "ريال",
    "ريالا",
    "ريالات",
    "ريالان",
    "هلله",
    "هللات",
    "هللة",
    "ه",
    "و",
}

UNITS = {
    "واحد": 1,
    "واحده": 1,
    "احد": 1,
    "احدى": 1,
    "اثنان": 2,
    "اثنين": 2,
    "اثنتان": 2,
    "اثنتين": 2,
    "اثنا": 2,
    "اثني": 2,
    "ثنين": 2,
    "ثنتين": 2,
    "ثلاث": 3,
    "ثلاثه": 3,
    "اربع": 4,
    "اربعه": 4,
    "خمس": 5,
    "خمسه": 5,
    "ست": 6,
    "سته": 6,
    "ستة": 6,
    "سبع": 7,
    "سبعه": 7,
    "ثمان": 8,
    "ثماني": 8,
    "ثمانيه": 8,
    "تسع": 9,
    "تسعه": 9,
}

TEENS_AND_TENS = {
    "عشر": 10,
    "عشره": 10,
    "احدعشر": 11,
    "احدىعشر": 11,
    "اثناعشر": 12,
    "اثنيعشر": 12,
    "ثلاثعشر": 13,
    "ثلاثهعشر": 13,
    "اربععشر": 14,
    "اربعهعشر": 14,
    "خمسعشر": 15,
    "خمسهعشر": 15,
    "ستعشر": 16,
    "ستهعشر": 16,
    "سبععشر": 17,
    "سبعهعشر": 17,
    "ثمانعشر": 18,
    "ثمانيهعشر": 18,
    "تسععشر": 19,
    "تسعهعشر": 19,
    "عشرون": 20,
    "عشرين": 20,
    "ثلاثون": 30,
    "ثلاثين": 30,
    "اربعون": 40,
    "اربعين": 40,
    "خمسون": 50,
    "خمسين": 50,
    "ستون": 60,
    "ستين": 60,
    "سبعون": 70,
    "سبعين": 70,
    "ثمانون": 80,
    "ثمانين": 80,
    "تسعون": 90,
    "تسعين": 90,
}

HUNDREDS = {
    "مايه": 100,
    "مايه": 100,
    "مائه": 100,
    "مئه": 100,
    "ميه": 100,
    "مئتان": 200,
    "مائتان": 200,
    "مايتان": 200,
    "ميتان": 200,
    "مئتين": 200,
    "مايتين": 200,
    "ثلاثمايه": 300,
    "ثلاثمائه": 300,
    "ثلثمايه": 300,
    "اربعمائه": 400,
    "اربعمايه": 400,
    "خمسمايه": 500,
    "خمسمائه": 500,
    "ستمايه": 600,
    "ستمائه": 600,
    "سبعمايه": 700,
    "سبعمائه": 700,
    "ثمانمايه": 800,
    "ثمانمائه": 800,
    "تسعمايه": 900,
    "تسعمائه": 900,
}

THOUSANDS = {
    "الف",
    "الاف",
    "الفا",
    "الفان",
    "الفي",
    "الفين",
    "لاف",
    "لف",
    "فالا",
    "فال",
}
THOUSANDS_FIXED = {"الفان": 2000, "الفي": 2000, "الفين": 2000}
MILLIONS = {"مليون", "مليونان", "مليونين", "ملايين"}
MILLIONS_FIXED = {"مليونان": 2_000_000, "مليونين": 2_000_000}

WORD_VALUES = {**UNITS, **TEENS_AND_TENS, **HUNDREDS}
LEXICON = set(WORD_VALUES) | IGNORE_WORDS | THOUSANDS | MILLIONS


@dataclass
class TokenTrace:
    token: str
    action: str
    value: int | None = None
    matched: str | None = None
    distance: int | None = None
    pieces: list[str] = field(default_factory=list)


@dataclass
class ConversionResult:
    value: str
    trace: list[TokenTrace]
    unresolved_tokens: list[str]


def normalize_arabic_text(text: str) -> str:
    for mark in BIDI_MARKS:
        text = text.replace(mark, "")
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(ARABIC_DIGITS)
    text = DIACRITICS_RE.sub("", text)
    text = text.translate(ARABIC_CHARS).replace("\u0621", "")
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
    text = re.sub(r"\s+", " ", text).strip()
    replacements = [
        (r"\bا\s+ل\s+ا\s+ف\b", "الاف"),
        (r"\bا\s+لا\s+ف\b", "الاف"),
        (r"\bا\s+لف\b", "الف"),
        (r"\bا\s+ل\s+ف\b", "الف"),
        (r"\bا\s+لفا\s+ن\b", "الفان"),
        (r"\bر\s+يا\s+ل(?:ا)?\b", "ريال"),
        (r"\bر\s+ي\s+ا\s+ل(?:ا)?\b", "ريال"),
        (r"\bه\s+لله\b", "هلله"),
        (r"\bثلا\s+ث\b", "ثلاث"),
        (r"\bثما\s+ن\b", "ثمان"),
        (r"\bو\s+ن\b", "ون"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token in normalize_arabic_text(text).split():
        if token.startswith("و") and len(token) > 1:
            rest = token[1:]
            if rest in LEXICON or rest.isdigit():
                tokens.append("و")
                tokens.append(rest)
                continue
        tokens.append(token)
    return tokens


def normalize_amount_digits(value: str | int | None) -> str:
    if value is None:
        return ""
    digits = re.sub(r"\D", "", str(value).translate(ARABIC_DIGITS))
    if not digits:
        return ""
    return digits.lstrip("0") or "0"


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


def _max_distance(token: str) -> int:
    if len(token) <= 3:
        return 1
    if len(token) <= 6:
        return 2
    return 3


def best_lexicon_match(token: str) -> tuple[str, int] | None:
    candidates: list[tuple[int, str]] = []
    forms = [token]
    if len(token) >= 4:
        forms.append(token[::-1])
    for form in forms:
        for word in LEXICON:
            distance = edit_distance(form, word)
            candidates.append((distance, word))
    if not candidates:
        return None
    distance, word = min(candidates, key=lambda item: (item[0], len(item[1]), item[1]))
    if distance <= _max_distance(token):
        return word, distance
    return None


def split_token(token: str) -> list[str] | None:
    if token in LEXICON or token.isdigit():
        return [token]
    n = len(token)
    best: list[str] | None = None
    memo: dict[int, list[str] | None] = {}

    def visit(index: int) -> list[str] | None:
        if index == n:
            return []
        if index in memo:
            return memo[index]
        local_best: list[str] | None = None
        for end in range(n, index + 1, -1):
            piece = token[index:end]
            if piece not in LEXICON and not piece.isdigit():
                continue
            rest = visit(end)
            if rest is None:
                continue
            candidate = [piece] + rest
            if local_best is None or len(candidate) < len(local_best):
                local_best = candidate
        memo[index] = local_best
        return local_best

    best = visit(0)
    if best and len(best) > 1:
        return best
    return None


def _resolve_tokens(tokens: Iterable[str], mode: str) -> tuple[list[str], list[TokenTrace]]:
    resolved: list[str] = []
    trace: list[TokenTrace] = []
    for token in tokens:
        if token.isdigit():
            trace.append(TokenTrace(token=token, action="numeric_ignored"))
            continue
        if token in LEXICON:
            resolved.append(token)
            trace.append(TokenTrace(token=token, action="direct", matched=token))
            continue
        if mode in {"edit_distance", "edit_distance_split"}:
            match = best_lexicon_match(token)
            if match is not None:
                word, distance = match
                resolved.append(word)
                trace.append(TokenTrace(token=token, action="edit_distance", matched=word, distance=distance))
                continue
        if mode == "edit_distance_split":
            pieces = split_token(token)
            if pieces is not None:
                resolved.extend(pieces)
                trace.append(TokenTrace(token=token, action="split", pieces=pieces))
                continue
        trace.append(TokenTrace(token=token, action="unresolved"))
    return resolved, trace


def _consume_token(token: str, current: int, total: int) -> tuple[int, int, int | None, str]:
    if token in IGNORE_WORDS:
        return current, total, None, "ignored"
    if token in THOUSANDS:
        if token in THOUSANDS_FIXED:
            value = THOUSANDS_FIXED[token]
        else:
            value = (current if current else 1) * 1000
        return 0, total + value, value, "thousand"
    if token in MILLIONS:
        if token in MILLIONS_FIXED:
            value = MILLIONS_FIXED[token]
        else:
            value = (current if current else 1) * 1_000_000
        return 0, total + value, value, "million"
    if token in HUNDREDS and HUNDREDS[token] == 100:
        value = (current if current else 1) * 100
        return 0, total + value, value, "hundred_multiplier"
    if token in WORD_VALUES:
        value = WORD_VALUES[token]
        return current + value, total, value, "number"
    return current, total, None, "unresolved"


def convert_legal_amount(text: str, mode: str = "direct") -> ConversionResult:
    if mode not in {"direct", "edit_distance", "edit_distance_split"}:
        raise ValueError(f"Unknown conversion mode: {mode}")
    raw_tokens = tokenize(text)
    resolved_tokens, resolution_trace = _resolve_tokens(raw_tokens, mode)

    total = 0
    current = 0
    has_numeric_value = False
    unresolved: list[str] = []
    conversion_trace: list[TokenTrace] = []

    for token in resolved_tokens:
        current, total, value, action = _consume_token(token, current, total)
        if action == "unresolved":
            unresolved.append(token)
        elif value is not None:
            has_numeric_value = True
        conversion_trace.append(TokenTrace(token=token, action=action, value=value, matched=token))

    total += current
    value = normalize_amount_digits(total) if has_numeric_value or total else ""
    return ConversionResult(value=value, trace=resolution_trace + conversion_trace, unresolved_tokens=unresolved)


def result_to_trace_dict(result: ConversionResult) -> list[dict[str, object]]:
    return [
        {
            "token": item.token,
            "action": item.action,
            "value": item.value,
            "matched": item.matched,
            "distance": item.distance,
            "pieces": item.pieces,
        }
        for item in result.trace
    ]
