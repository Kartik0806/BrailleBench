"""Text -> Unicode braille (Grade 1 / Grade 2), tables and source-text normalisation."""

import random
import re
import unicodedata
from typing import Optional
from .config import PipelineConfig
from .deps import HAS_LIBLOUIS, louis


# ===========================================================================
# TEXT -> UNICODE BRAILLE (Grade-1 / Grade-2)
# ===========================================================================

BRAILLE_SPACE = "\u2800"
CAPITAL_SIGN = "\u2820"  # dot 6
NUMBER_SIGN = "\u283c"  # dots 3456

# One-cell literary braille patterns used by the built-in Grade-1 translator.
LETTER_TO_BRAILLE = {
    "a": "⠁",
    "b": "⠃",
    "c": "⠉",
    "d": "⠙",
    "e": "⠑",
    "f": "⠋",
    "g": "⠛",
    "h": "⠓",
    "i": "⠊",
    "j": "⠚",
    "k": "⠅",
    "l": "⠇",
    "m": "⠍",
    "n": "⠝",
    "o": "⠕",
    "p": "⠏",
    "q": "⠟",
    "r": "⠗",
    "s": "⠎",
    "t": "⠞",
    "u": "⠥",
    "v": "⠧",
    "w": "⠺",
    "x": "⠭",
    "y": "⠽",
    "z": "⠵",
}
DIGIT_TO_BRAILLE = {
    "1": LETTER_TO_BRAILLE["a"],
    "2": LETTER_TO_BRAILLE["b"],
    "3": LETTER_TO_BRAILLE["c"],
    "4": LETTER_TO_BRAILLE["d"],
    "5": LETTER_TO_BRAILLE["e"],
    "6": LETTER_TO_BRAILLE["f"],
    "7": LETTER_TO_BRAILLE["g"],
    "8": LETTER_TO_BRAILLE["h"],
    "9": LETTER_TO_BRAILLE["i"],
    "0": LETTER_TO_BRAILLE["j"],
}
PUNCT_TO_BRAILLE = {
    ",": "⠂",
    ";": "⠆",
    ":": "⠒",
    ".": "⠲",
    "!": "⠖",
    "?": "⠦",
    "'": "⠄",
    "-": "⠤",
    '"': "⠦",
    "(": "⠐⠣",
    ")": "⠐⠜",
    "/": "⠸⠌",
    "&": "⠯",
    "+": "⠐⠖",
    "=": "⠐⠶",
    "%": "⠨⠴",
}

# Compatibility map retained for callers that used the old symbol name.
NABCC_TO_UNICODE = {" ": BRAILLE_SPACE, **LETTER_TO_BRAILLE, **PUNCT_TO_BRAILLE}
NABCC_TO_UNICODE.update(
    {k.upper(): CAPITAL_SIGN + v for k, v in LETTER_TO_BRAILLE.items()}
)
NABCC_TO_UNICODE.update({k: NUMBER_SIGN + v for k, v in DIGIT_TO_BRAILLE.items()})

SUPPORTED_ASCII_CHARS = set(LETTER_TO_BRAILLE) | set(
    k.upper() for k in LETTER_TO_BRAILLE
)
SUPPORTED_ASCII_CHARS |= set(DIGIT_TO_BRAILLE) | set(PUNCT_TO_BRAILLE) | {" ", "\n"}

# A compact UEB Grade-2 fallback. Full Grade-2 is preferably produced by
# liblouis; this table keeps the script operational when liblouis is absent.
G2_WHOLE_WORD = {
    "but": "⠃",
    "can": "⠉",
    "do": "⠙",
    "every": "⠑",
    "from": "⠋",
    "go": "⠛",
    "have": "⠓",
    "just": "⠚",
    "knowledge": "⠅",
    "like": "⠇",
    "more": "⠍",
    "not": "⠝",
    "people": "⠏",
    "quite": "⠟",
    "rather": "⠗",
    "so": "⠎",
    "that": "⠞",
    "us": "⠥",
    "very": "⠧",
    "will": "⠺",
    "it": "⠭",
    "you": "⠽",
    "as": "⠵",
    "and": "⠯",
    "for": "⠿",
    "of": "⠷",
    "the": "⠮",
    "with": "⠺",
    "child": "⠡",
    "shall": "⠩",
    "this": "⠹",
    "which": "⠱",
    "out": "⠳",
    "still": "⠌",
}
G2_GROUPS = [
    ("ing", "⠬"),
    ("ch", "⠡"),
    ("sh", "⠩"),
    ("th", "⠹"),
    ("wh", "⠱"),
    ("ou", "⠳"),
    ("st", "⠌"),
    ("ar", "⠜"),
    ("ed", "⠫"),
    ("er", "⠻"),
    ("ow", "⠪"),
    ("gh", "⠣"),
]


def is_unicode_braille_string(text: str, allow_newline: bool = True) -> bool:
    """True iff every non-newline codepoint is Unicode Braille U+2800..U+28FF."""
    return all(
        (allow_newline and ch == "\n") or (0x2800 <= ord(ch) <= 0x28FF) for ch in text
    )


def _normalize_source_text(text: str) -> str:
    """Normalize web text to a conservative ASCII subset with braille mappings."""
    replacements = {
        "“": '"',
        "”": '"',
        "„": '"',
        "’": "'",
        "‘": "'",
        "`": "'",
        "–": "-",
        "—": "-",
        "−": "-",
        "…": "...",
        "\u00a0": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    filtered = []
    for ch in text:
        if ch == "\n":
            filtered.append("\n")
        elif ch.isspace():
            filtered.append(" ")
        elif ch in SUPPORTED_ASCII_CHARS:
            filtered.append(ch)
        else:
            filtered.append(" ")
    cleaned = "\n".join(
        " ".join(part.split()) for part in "".join(filtered).split("\n")
    )
    return cleaned.strip()


def remove_unsupported_braille_chars(text: str) -> str:
    """Backward-compatible alias for strict source-text normalization."""
    return _normalize_source_text(text)


_ALNUM_RE = re.compile(r"[A-Za-z0-9]")


def strip_punct_edge_tokens(text: str) -> str:
    """Drop whitespace-separated tokens that contain no letter or digit from both ends of the text
    ("- and" -> "and", "UPC /" -> "UPC"); returns "" when no token has a letter or digit.
    Real labels never start or end with such a token; FineWeb windows often do (list dashes)."""
    words = text.split()
    while words and not _ALNUM_RE.search(words[0]):
        words.pop(0)
    while words and not _ALNUM_RE.search(words[-1]):
        words.pop()
    return " ".join(words)


def shuffle_source_text(
    text: str, cfg: PipelineConfig, rng: Optional[random.Random] = None
) -> tuple[str, dict]:
    """Perturb the English source BEFORE Braille translation and labeling."""
    rng = rng if rng is not None else random
    words = text.split()
    changed_order = False
    changed_character_words = 0

    if (
        cfg.p_shuffle_words > 0
        and len(set(words)) > 1
        and rng.random() < cfg.p_shuffle_words
    ):
        original = words[:]
        for _ in range(8):
            rng.shuffle(words)
            if words != original:
                break
        if words == original:
            words = words[1:] + words[:1]
        changed_order = True

    if cfg.p_shuffle_chars > 0 and rng.random() < cfg.p_shuffle_chars:
        def shuffle_letters(match: re.Match) -> str:
            token = match.group()
            letters = list(token.lower())
            if len(set(letters)) < 2:
                return token
            for _ in range(8):
                rng.shuffle(letters)
                if "".join(letters) != token.lower():
                    break
            if "".join(letters) == token.lower():
                letters = letters[1:] + letters[:1]
            return "".join(
                char.upper() if original.isupper() else char
                for char, original in zip(letters, token)
            )

        for idx, word in enumerate(words):
            if (
                re.search(r"[A-Za-z]{3,}", word)
                and rng.random() < cfg.shuffle_char_word_fraction
            ):
                changed = re.sub(r"[A-Za-z]{3,}", shuffle_letters, word)
                if changed != word:
                    words[idx] = changed
                    changed_character_words += 1

    if not changed_order and not changed_character_words:
        return text, {}
    return " ".join(words), {
        "word_order_shuffled": changed_order,
        "character_shuffled_words": changed_character_words,
    }


def _grade1_builtin(text: str) -> str:
    """Built-in uncontracted translator used only as a liblouis fallback."""
    out = []
    in_number = False
    for ch in text:
        if ch == "\n":
            out.append("\n")
            in_number = False
        elif ch.isspace():
            out.append(BRAILLE_SPACE)
            in_number = False
        elif ch.isdigit():
            if not in_number:
                out.append(NUMBER_SIGN)
                in_number = True
            out.append(DIGIT_TO_BRAILLE[ch])
        elif ch.isalpha():
            in_number = False
            if ch.isupper():
                out.append(CAPITAL_SIGN)
            out.append(LETTER_TO_BRAILLE[ch.lower()])
        else:
            in_number = False
            out.append(PUNCT_TO_BRAILLE[ch])
    result = "".join(out)
    if not is_unicode_braille_string(result):
        raise ValueError("Built-in Grade-1 translation emitted a non-braille codepoint")
    return result


def grade1_to_unicode(text: str) -> tuple[str, str]:
    """Return (unicode_braille, backend), preferring liblouis UEB Grade-1."""
    if HAS_LIBLOUIS:
        candidates = [
            ["braille-patterns.cti", "en-ueb-g1.ctb"],
            ["braille-patterns.cti", "en-us-g1.ctb"],
        ]
        for tables in candidates:
            try:
                translated = louis.translateString(tables, text)
                translated = translated.replace(" ", BRAILLE_SPACE)
                if is_unicode_braille_string(translated):
                    return translated, "liblouis:" + "+".join(tables)
            except Exception:
                pass

    return _grade1_builtin(text), "builtin-grade1-fallback"


def _grade2_builtin_word(word: str) -> str:
    """Small rule-based Grade-2 fallback used only when liblouis is unavailable."""
    # Preserve simple leading/trailing punctuation around the alphabetic core.
    left = 0
    right = len(word)
    while left < right and not word[left].isalnum():
        left += 1
    while right > left and not word[right - 1].isalnum():
        right -= 1
    prefix, core, suffix = word[:left], word[left:right], word[right:]
    if not core:
        return _grade1_builtin(word)

    lower = core.lower()
    cap = CAPITAL_SIGN if core[:1].isupper() else ""
    if lower in G2_WHOLE_WORD and lower.isalpha():
        middle = cap + G2_WHOLE_WORD[lower]
    elif lower.isalpha():
        cells = []
        i = 0
        while i < len(lower):
            matched = False
            for seq, cell in G2_GROUPS:
                if lower.startswith(seq, i):
                    cells.append(cell)
                    i += len(seq)
                    matched = True
                    break
            if not matched:
                cells.append(LETTER_TO_BRAILLE[lower[i]])
                i += 1
        middle = cap + "".join(cells)
    else:
        middle = _grade1_builtin(core)
    return _grade1_builtin(prefix) + middle + _grade1_builtin(suffix)


def _grade2_builtin(text: str) -> str:
    parts = []
    for pi, paragraph in enumerate(text.split("\n")):
        words = paragraph.split()
        parts.append(BRAILLE_SPACE.join(_grade2_builtin_word(w) for w in words))
        if pi != len(text.split("\n")) - 1:
            parts.append("\n")
    result = "".join(parts)
    if not is_unicode_braille_string(result):
        raise ValueError("Built-in Grade-2 translation emitted a non-braille codepoint")
    return result


def grade2_to_unicode(text: str) -> tuple[str, str]:
    """Return (unicode_braille, backend); liblouis is preferred for full UEB G2."""
    if HAS_LIBLOUIS:
        candidates = [
            ["braille-patterns.cti", "en-ueb-g2.ctb"],
            ["braille-patterns.cti", "en-us-g2.ctb"],
        ]
        for tables in candidates:
            try:
                translated = louis.translateString(tables, text)
                translated = translated.replace(" ", BRAILLE_SPACE)
                if is_unicode_braille_string(translated):
                    return translated, "liblouis:" + "+".join(tables)
            except Exception:
                pass
    return _grade2_builtin(text), "builtin-grade2-subset"


def text_to_braille(text: str, grade: int) -> tuple[str, str]:
    """Translate sanitized source text and guarantee Unicode-braille-only output."""
    text = _normalize_source_text(text)
    if grade == 2:
        braille, backend = grade2_to_unicode(text)
    else:
        braille, backend = grade1_to_unicode(text)
    if not is_unicode_braille_string(braille):
        bad = sorted(
            {
                f"U+{ord(ch):04X}"
                for ch in braille
                if ch != "\n" and not (0x2800 <= ord(ch) <= 0x28FF)
            }
        )
        raise ValueError(f"Invalid non-braille output codepoints: {bad}")
    return braille, backend
