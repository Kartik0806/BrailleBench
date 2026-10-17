"""
Stratified text plans for the braille OCR dataset generator.
============================================================
A plan (JSON, schema v1) lists strata; every stratum has a kind, a weight and a
list of sizes. ``build_items`` turns a plan into exactly ``n`` seeded
``{"text", "info"}`` items whose texts are already braille-safe, so the label
equals the rendered text.

    {"version": 1, "max_chars": 450,
     "strata": [{"name": "...", "kind": "...", "weight": 8,
                 "sizes": [1, 2, 3] | {"min": 1, "max": 12}, ...kind params}]}

Kinds (``size`` in brackets): chars (characters), digits (digit characters),
alnum (alphanumeric characters), words / shuffled_words / scrambled_chars /
word_salad / mixed_tokens (whitespace-separated words or tokens),
shuffled_sentences (minimum total words, whole sentences, at least two),
paragraphs (paragraph count).

Pure stdlib. The generator's ``_normalize_source_text`` and
``SUPPORTED_ASCII_CHARS`` are injected into ``build_items``; nothing is imported
from the generator. Documents are pulled lazily and never drained.

CLI (uses a built-in copy of the generator's normalizer):
    python src/braille_text_plan.py --plan P --n N --seed S --corpus C.jsonl --show K
"""

import argparse
import copy
import json
import math
import operator
import os
import random
import re
import sys
import unicodedata
from collections import Counter
from fractions import Fraction
from typing import Optional

PLAN_VERSION = 1
KINDS = (
    "chars",
    "digits",
    "alnum",
    "words",
    "shuffled_words",
    "scrambled_chars",
    "shuffled_sentences",
    "word_salad",
    "mixed_tokens",
    "paragraphs",
)

# Random window starts probed per document before the next document is taken.
PROBES = 16
# A duplicate text is re-drawn at most this often, then accepted and counted.
MAX_UNIQUE_RETRIES = 20
# Consecutive unusable documents one item may burn before the plan is declared
# unsatisfiable (guards against draining the stream on an infeasible plan).
MAX_DOCS_PER_ITEM = 200
# Consecutive documents that normalize to nothing before the stream is rejected.
MAX_EMPTY_DOCS = 1000
# Invariant violations tolerated per item before giving up.
MAX_BAD_DRAWS = 50
# Vocabulary pool: distinct word types, capped per document and overall.
POOL_MIN_TYPES = 3000
POOL_MAX_TYPES = 100_000
POOL_TOKENS_PER_DOC = 150
MAX_WORD_CHARS = 24
# Natural windows never contain a token longer than this (URLs, base64 junk).
MAX_NATURAL_TOKEN_CHARS = 30
MAX_SIZE_RANGE = 10_000

_LOWER = "abcdefghijklmnopqrstuvwxyz"
_UPPER = _LOWER.upper()
_DIGITS = "0123456789"
_PUNCT = ",;:.!?'-\"()/&+=%"
_TRAIL = ",.,.;:!?)\""
_CHARSETS = {
    "lower": _LOWER,
    "upper": _UPPER,
    "letters": _LOWER + _UPPER,
    "digits": _DIGITS,
    "alnum": _LOWER + _UPPER + _DIGITS,
    "alnum_punct": _LOWER + _UPPER + _DIGITS + _PUNCT,
}
DIGIT_FORMATS = ("plain", "grouped", "decimal", "dashed", "slash", "percent")
_DIGITS_MIN = {"plain": 1, "grouped": 4, "decimal": 3, "dashed": 4, "percent": 2}
ALNUM_STYLES = ("mixed", "upper_code", "code", "lower_hex")
TOKEN_TYPES = ("word", "digits", "alnum", "word_punct", "char")
DEFAULT_TOKEN_MIX = {
    "word": 5,
    "digits": 2,
    "alnum": 1.5,
    "word_punct": 1,
    "char": 0.5,
}
_TOKEN_DIGITS = {"plain": 6, "grouped": 1, "decimal": 1, "percent": 0.5}
_TOKEN_ALNUM = {"mixed": 5, "upper_code": 2, "code": 2, "lower_hex": 1}
DEFAULT_SUPPORTED_CHARS = frozenset(_LOWER + _UPPER + _DIGITS + _PUNCT + " \n")

_COMMON_KEYS = ("name", "kind", "weight", "sizes")
_NOTE_KEYS = ("description", "comment")
_KIND_KEYS = {
    "chars": ("charset", "vocab_fraction"),
    "digits": ("formats",),
    "alnum": ("styles",),
    "scrambled_chars": ("word_fraction",),
    "mixed_tokens": ("token_mix",),
    "paragraphs": ("words_per_paragraph",),
}
_MIN_SIZE = {"alnum": 2, "shuffled_words": 2, "shuffled_sentences": 2}
_SHUFFLE_OF = {
    "shuffled_words": "word_order",
    "shuffled_sentences": "sentence_order",
    "scrambled_chars": "char_scramble",
    "word_salad": "salad",
}
_HAS_BEFORE = ("shuffled_words", "shuffled_sentences", "scrambled_chars")
_SINGLE_DOC_KINDS = (
    "words",
    "shuffled_words",
    "scrambled_chars",
    "shuffled_sentences",
)
_NAME_RE = re.compile(r"[a-z0-9_]+")
_WORD_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")
_LETTER_RUN = re.compile(r"[A-Za-z]{3,}")
_ABBREV_GROUP = r"(?:[A-Za-z]|[A-Z][A-Za-z]|al|vs|pp|cf|ed|ca)"
_ABBREV_RE = re.compile(_ABBREV_GROUP + r"(?:\." + _ABBREV_GROUP + r")*")
_LIST_MARKER_RE = re.compile(r"[0-9]{1,2}")


# ===========================================================================
# PLAN VALIDATION
# ===========================================================================


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_num(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _weights(where, what, value, allowed, default, allow_name=False) -> dict:
    """Validate a {name: weight} mapping (or one bare name) into a new dict."""
    if value is None:
        value = default
    if allow_name and isinstance(value, str):
        value = {value: 1}
    if not isinstance(value, dict) or not value:
        shape = "a name or " if allow_name else ""
        raise ValueError(
            f"{where}: {what} must be {shape}a non-empty {{name: weight}} mapping"
        )
    out = {}
    for key, weight in value.items():
        if key not in allowed:
            raise ValueError(
                f"{where}: unknown {what} name {key!r} (allowed: {', '.join(allowed)})"
            )
        if not _is_num(weight) or weight <= 0:
            raise ValueError(
                f"{where}: {what} weight for {key!r} must be a number > 0 "
                f"(got {weight!r})"
            )
        out[key] = weight
    return out


def _expand_sizes(where, sizes) -> list:
    if isinstance(sizes, dict):
        if set(sizes) != {"min", "max"}:
            raise ValueError(f'{where}: sizes mapping needs exactly "min" and "max"')
        lo, hi = sizes["min"], sizes["max"]
        if not (_is_int(lo) and _is_int(hi) and 1 <= lo <= hi):
            raise ValueError(
                f"{where}: sizes min/max must be ints with 1 <= min <= max "
                f"(got {lo!r}, {hi!r})"
            )
        if hi - lo + 1 > MAX_SIZE_RANGE:
            raise ValueError(f"{where}: sizes range is larger than {MAX_SIZE_RANGE}")
        return list(range(lo, hi + 1))
    if not isinstance(sizes, list):
        raise ValueError(
            f"{where}: sizes must be a non-empty list of positive ints or a "
            f'{{"min", "max"}} mapping (got {type(sizes).__name__})'
        )
    if not sizes:
        raise ValueError(f"{where}: sizes must not be empty")
    for size in sizes:
        if not (_is_int(size) and size >= 1):
            raise ValueError(f"{where}: sizes must be positive ints (got {size!r})")
    if len(set(sizes)) != len(sizes):
        raise ValueError(f"{where}: sizes must be distinct (got {sizes})")
    return list(sizes)


def _min_chars(st: dict, size: int) -> int:
    """Fewest characters a text of this size can possibly have."""
    if st["kind"] in ("chars", "digits", "alnum"):
        return size
    if st["kind"] == "paragraphs":
        lo = st["words_per_paragraph"][0]
        return size * (2 * lo - 1) + size - 1
    return 2 * size - 1


def _validate_stratum(idx: int, raw, max_chars: int, seen: set) -> dict:
    if not isinstance(raw, dict):
        raise ValueError(
            f"plan stratum #{idx}: must be an object (got {type(raw).__name__})"
        )
    name = raw.get("name")
    where = (
        f"plan stratum {name!r}" if isinstance(name, str) else f"plan stratum #{idx}"
    )
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ValueError(f"{where}: name must match [a-z0-9_]+ (got {name!r})")
    if name in seen:
        raise ValueError(f"{where}: duplicate stratum name")
    seen.add(name)
    kind = raw.get("kind")
    if kind not in KINDS:
        raise ValueError(
            f"{where}: unknown kind {kind!r} (expected one of {', '.join(KINDS)})"
        )
    allowed = set(_COMMON_KEYS) | set(_NOTE_KEYS) | set(_KIND_KEYS.get(kind, ()))
    extra = sorted(str(key) for key in set(raw) - allowed)
    if extra:
        raise ValueError(
            f"{where}: unknown key(s) for kind {kind!r}: {', '.join(extra)}"
        )

    weight = raw.get("weight")
    if not _is_num(weight) or weight <= 0:
        raise ValueError(f"{where}: weight must be a number > 0 (got {weight!r})")
    out = {"name": name, "kind": kind, "weight": weight}
    out["sizes"] = _expand_sizes(where, raw.get("sizes"))
    for key in _NOTE_KEYS:
        if key in raw:
            out[key] = raw[key]

    min_size = _MIN_SIZE.get(kind, 1)
    if min(out["sizes"]) < min_size:
        raise ValueError(
            f"{where}: {kind} needs sizes >= {min_size} (got {min(out['sizes'])})"
        )

    if kind == "chars":
        out["charset"] = _weights(
            where,
            "charset",
            raw.get("charset"),
            tuple(_CHARSETS),
            "alnum",
            allow_name=True,
        )
        fraction = raw.get("vocab_fraction", 0)
        if not _is_num(fraction) or not 0 <= fraction <= 1:
            raise ValueError(
                f"{where}: vocab_fraction must be a number in [0, 1] "
                f"(got {fraction!r})"
            )
        out["vocab_fraction"] = fraction
    elif kind == "digits":
        out["formats"] = _weights(
            where, "formats", raw.get("formats"), DIGIT_FORMATS, {"plain": 1}
        )
    elif kind == "alnum":
        out["styles"] = _weights(
            where, "styles", raw.get("styles"), ALNUM_STYLES, {"mixed": 1}
        )
    elif kind == "scrambled_chars":
        fraction = raw.get("word_fraction", 0.6)
        if not _is_num(fraction) or not 0 < fraction <= 1:
            raise ValueError(
                f"{where}: word_fraction must be a number in (0, 1] "
                f"(got {fraction!r})"
            )
        out["word_fraction"] = fraction
    elif kind == "mixed_tokens":
        out["token_mix"] = _weights(
            where, "token_mix", raw.get("token_mix"), TOKEN_TYPES, DEFAULT_TOKEN_MIX
        )
    elif kind == "paragraphs":
        span = raw.get("words_per_paragraph", [6, 18])
        if not (
            isinstance(span, (list, tuple))
            and len(span) == 2
            and all(_is_int(v) for v in span)
            and 1 <= span[0] <= span[1]
        ):
            raise ValueError(
                f"{where}: words_per_paragraph must be [lo, hi] ints with "
                f"1 <= lo <= hi (got {span!r})"
            )
        out["words_per_paragraph"] = [int(span[0]), int(span[1])]

    for size in out["sizes"]:
        need = _min_chars(out, size)
        if need > max_chars:
            raise ValueError(
                f"{where}: size {size} needs at least {need} characters but "
                f"max_chars is {max_chars}"
            )
    return out


def load_plan(path_or_dict) -> dict:
    """Validate a plan (JSON path or dict) and return a normalized copy.

    Sizes are expanded to lists and every default is filled in. The result is a
    valid input again, so ``load_plan(load_plan(p)) == load_plan(p)``. Raises
    ``ValueError`` with a message that names the offending stratum.
    """
    if isinstance(path_or_dict, dict):
        raw = copy.deepcopy(path_or_dict)
    else:
        try:
            path = os.fspath(path_or_dict)
            with open(path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(
                f"plan {path_or_dict!r} is not readable JSON: {exc}"
            ) from exc
    if not isinstance(raw, dict):
        raise ValueError("plan must be a JSON object")
    known = {"version", "max_chars", "strata", *_NOTE_KEYS}
    extra = sorted(str(key) for key in set(raw) - known)
    if extra:
        raise ValueError(f"plan: unknown top-level key(s): {', '.join(extra)}")
    if raw.get("version", PLAN_VERSION) != PLAN_VERSION:
        raise ValueError(
            f"plan: unsupported version {raw.get('version')!r} "
            f"(expected {PLAN_VERSION})"
        )
    max_chars = raw.get("max_chars", 450)
    if not _is_int(max_chars) or max_chars < 1:
        raise ValueError(f"plan: max_chars must be a positive int (got {max_chars!r})")
    strata = raw.get("strata")
    if not isinstance(strata, list) or not strata:
        raise ValueError("plan: strata must be a non-empty list")
    seen: set = set()
    out = {"version": PLAN_VERSION, "max_chars": max_chars}
    for key in _NOTE_KEYS:
        if key in raw:
            out[key] = raw[key]
    out["strata"] = [
        _validate_stratum(idx, st, max_chars, seen) for idx, st in enumerate(strata)
    ]
    return out


# ===========================================================================
# ALLOCATION
# ===========================================================================


def _check_n(n) -> int:
    try:
        value = operator.index(n)
    except TypeError:
        raise ValueError(f"n must be a non-negative int (got {n!r})") from None
    if isinstance(n, bool) or value < 0:
        raise ValueError(f"n must be a non-negative int (got {n!r})")
    return value


def allocate(plan: dict, n: int) -> list[tuple[dict, int]]:
    """Largest-remainder split of ``n`` over the strata weights, in plan order.

    Exact rational arithmetic, ties go to the earlier stratum; the counts sum
    to ``n`` and strata may receive 0.
    """
    plan = load_plan(plan)
    n = _check_n(n)
    weights = [Fraction(st["weight"]) for st in plan["strata"]]
    total = sum(weights)
    quotas = [w * n / total for w in weights]
    counts = [int(q) for q in quotas]
    order = sorted(range(len(counts)), key=lambda j: (-(quotas[j] - counts[j]), j))
    for j in order[: n - sum(counts)]:
        counts[j] += 1
    return list(zip(plan["strata"], counts))


def _sizes_label(sizes: list) -> str:
    if len(sizes) > 1 and sizes == list(range(sizes[0], sizes[-1] + 1)):
        return f"{sizes[0]}..{sizes[-1]}"
    label = ",".join(str(s) for s in sizes)
    return label if len(label) <= 28 else label[:25] + "..."


def describe_allocation(plan: dict, n: int) -> str:
    """Printable stratum -> count table."""
    plan = load_plan(plan)
    rows = allocate(plan, n)
    header = ("stratum", "kind", "weight", "sizes", "count", "share")
    body = [
        (
            st["name"],
            st["kind"],
            f"{st['weight']:g}",
            _sizes_label(st["sizes"]),
            str(count),
            f"{100.0 * count / n:.1f}%" if n else "-",
        )
        for st, count in rows
    ]
    total_weight = sum(Fraction(st["weight"]) for st, _ in rows)
    share = "100.0%" if n else "-"
    body.append(("total", "", f"{float(total_weight):g}", "", str(n), share))
    widths = [max(len(r[c]) for r in [header] + body) for c in range(len(header))]

    def fmt(row):
        cells = [
            row[c].ljust(widths[c]) if c in (0, 1, 3) else row[c].rjust(widths[c])
            for c in range(len(header))
        ]
        return "  ".join(cells).rstrip()

    lines = [
        f"text plan v{plan['version']}: n={n} max_chars={plan['max_chars']} "
        f"strata={len(rows)}",
        fmt(header),
    ]
    lines += [fmt(r) for r in body]
    return "\n".join(lines)


# ===========================================================================
# SYNTHETIC STRINGS (chars / digits / alnum / tokens)
# ===========================================================================


def _wchoice(rng: random.Random, weights: dict) -> str:
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def _digit_string(rng: random.Random, n: int, lead: str = "some") -> str:
    """n random digits; the first is never 0 for lead="nonzero", rarely 0 otherwise."""
    if n == 1:
        first = rng.choice(_DIGITS)
    elif lead == "nonzero" or rng.random() < 0.8:
        first = rng.choice(_DIGITS[1:])
    else:
        first = rng.choice(_DIGITS)
    return first + "".join(rng.choice(_DIGITS) for _ in range(n - 1))


def _split_sizes(rng: random.Random, total: int, parts: int, minimum: int) -> list[int]:
    sizes = [minimum] * parts
    for _ in range(total - minimum * parts):
        sizes[rng.randrange(parts)] += 1
    return sizes


def _digits_feasible(fmt: str, size: int) -> bool:
    if fmt == "slash":
        return size in (6, 8)
    return size >= _DIGITS_MIN[fmt]


def _gen_digits(rng: random.Random, size: int, formats: dict, max_chars: int) -> str:
    fmt = _wchoice(rng, formats)
    if not _digits_feasible(fmt, size):
        fmt = "plain"
    if fmt == "grouped":
        digits = _digit_string(rng, size, "nonzero")
        head = size % 3 or 3
        parts = [digits[:head]] + [digits[i : i + 3] for i in range(head, size, 3)]
        text = ",".join(parts)
    elif fmt == "decimal":
        frac = 2 if rng.random() < 0.5 else rng.randint(1, min(4, size - 1))
        head = _digit_string(rng, size - frac, "nonzero")
        text = head + "." + "".join(rng.choice(_DIGITS) for _ in range(frac))
    elif fmt == "dashed":
        parts = rng.randint(2, min(4, size // 2))
        text = "-".join(
            "".join(rng.choice(_DIGITS) for _ in range(k))
            for k in _split_sizes(rng, size, parts, 2)
        )
    elif fmt == "slash":
        day = f"{rng.randint(1, 28):02d}"
        month = f"{rng.randint(1, 12):02d}"
        if size == 8:
            year = str(rng.randint(1900, 2099))
        else:
            year = f"{rng.randint(0, 99):02d}"
        order = rng.choice(("dmy", "mdy", "ymd"))
        text = "/".join({"d": day, "m": month, "y": year}[c] for c in order)
    elif fmt == "percent":
        text = _digit_string(rng, size, "nonzero") + "%"
    else:
        text = _digit_string(rng, size)
    if len(text) > max_chars:
        text = _digit_string(rng, size)
    return text


def _ensure_letter_and_digit(rng: random.Random, chars: list, letters: str) -> None:
    """Patch one random position so the string holds a letter and a digit (len >= 2)."""
    if not any(c.isalpha() for c in chars):
        chars[rng.randrange(len(chars))] = rng.choice(letters)
    elif not any(c.isdigit() for c in chars):
        chars[rng.randrange(len(chars))] = rng.choice(_DIGITS)


def _gen_alnum(rng: random.Random, size: int, styles: dict, max_chars: int) -> str:
    style = _wchoice(rng, styles)
    if style == "code":
        parts = 3 if size >= 6 and rng.random() < 0.4 else 2
        sizes = _split_sizes(rng, size, parts, 2 if size >= 4 else 1)
        chars: list[str] = []
        for k in sizes:
            alphabet = rng.choices(
                (_UPPER, _DIGITS, _UPPER + _DIGITS), weights=(4, 4, 2)
            )[0]
            chars.extend(rng.choice(alphabet) for _ in range(k))
        _ensure_letter_and_digit(rng, chars, _UPPER)
        groups = []
        pos = 0
        for k in sizes:
            groups.append("".join(chars[pos : pos + k]))
            pos += k
        text = "-".join(groups)
        if len(text) <= max_chars:
            return text
        style = "upper_code"
    alphabet, letters = {
        "mixed": (_LOWER + _UPPER + _DIGITS, _LOWER + _UPPER),
        "upper_code": (_UPPER + _DIGITS, _UPPER),
        "lower_hex": ("0123456789abcdef", "abcdef"),
    }[style]
    chars = [rng.choice(alphabet) for _ in range(size)]
    _ensure_letter_and_digit(rng, chars, letters)
    return "".join(chars)


def _gen_chars(rng: random.Random, size: int, charset: dict) -> str:
    alphabet = _CHARSETS[_wchoice(rng, charset)]
    return "".join(rng.choice(alphabet) for _ in range(size))


# ===========================================================================
# NATURAL TEXT (windows, scrambling, sentences)
# ===========================================================================


def _natural_ok(tokens: list) -> bool:
    """No over-long token, and at least one alphanumeric character."""
    if max(map(len, tokens)) > MAX_NATURAL_TOKEN_CHARS:
        return False
    return any(ch.isalnum() for tok in tokens for ch in tok)


def _find_window(words: list, k: int, rng: random.Random, limit: int, accept=None):
    """First acceptable window of exactly ``k`` words among random probed starts."""
    n_starts = len(words) - k + 1
    if n_starts <= 0:
        return None
    if n_starts <= PROBES:
        starts = list(range(n_starts))
        rng.shuffle(starts)
    else:
        starts = rng.sample(range(n_starts), PROBES)
    for start in starts:
        tokens = words[start : start + k]
        if sum(map(len, tokens)) + k - 1 > limit or not _natural_ok(tokens):
            continue
        if accept is None or accept(tokens):
            return tokens
    return None


def _eligible(word: str) -> bool:
    """A letter run of length >= 3 holding at least two distinct letters."""
    return any(len(set(m.group().lower())) >= 2 for m in _LETTER_RUN.finditer(word))


def _scramble_word(word: str, rng: random.Random) -> str:
    """Permute every eligible letter run; case is restored positionally.

    Mirrors ``shuffle_source_text`` of the generator: the lower-cased letters are
    shuffled (rotated if eight shuffles fail), then position i is upper-case iff
    the original character at i was, so the result always differs.
    """

    def scramble(match: re.Match) -> str:
        run = match.group()
        letters = list(run.lower())
        if len(set(letters)) < 2:
            return run
        shuffled = letters[:]
        for _ in range(8):
            rng.shuffle(shuffled)
            if shuffled != letters:
                break
        else:
            shuffled = letters[1:] + letters[:1]
        return "".join(
            c.upper() if o.isupper() else c for c, o in zip(shuffled, run)
        )

    return _LETTER_RUN.sub(scramble, word)


def _ends_sentence(token: str) -> bool:
    """Token ends in . ! ? and is not an abbreviation or a list number.

    Abbreviations are dot-separated groups of one letter or two letters with a
    capital initial (Mr. J. U.S. Ph.D. PM.), plus al/vs/pp/cf/ed/ca. Lowercase
    two-letter words (it. to. go.) are real words and do end a sentence; 1-2
    digit tokens (1. 12.) are list numbers, 3+ digit ones (2019.) are not.
    """
    last = token[-1]
    if last in "!?":
        return True
    if last != ".":
        return False
    core = token[:-1]
    return not (_ABBREV_RE.fullmatch(core) or _LIST_MARKER_RE.fullmatch(core))


def _reorder(items: list, rng: random.Random) -> list:
    """Shuffle until the order differs (a rotation if shuffling keeps failing)."""
    order = items[:]
    for _ in range(10):
        rng.shuffle(order)
        if order != items:
            return order
    return items[1:] + items[:1]


class _Doc:
    """A normalized document as a word list, with lazily split sentences."""

    __slots__ = ("words", "_sentences")

    def __init__(self, words: list):
        self.words = words
        self._sentences = None

    def sentences(self) -> list[tuple[int, int]]:
        """(start, end) spans of whole sentences; a trailing fragment is dropped."""
        if self._sentences is None:
            spans = []
            start = 0
            for idx, token in enumerate(self.words):
                if _ends_sentence(token):
                    spans.append((start, idx + 1))
                    start = idx + 1
            self._sentences = spans
        return self._sentences


class _Vocab:
    """Distinct word types of consumed documents, in deterministic insertion order."""

    def __init__(self):
        self.words: list[str] = []
        self.by_len: dict[int, list[str]] = {}
        self._seen: set[str] = set()

    def add_doc(
        self, words: list, rng: random.Random, supported, limit: int, normalize
    ) -> None:
        """Pool the word types of one document; tokens must satisfy the invariants."""
        if len(self.words) >= POOL_MAX_TYPES:
            return
        budget = 3 * POOL_TOKENS_PER_DOC
        sample = words if len(words) <= budget else rng.sample(words, budget)
        types = {}
        for token in sample:
            core = token.strip(_PUNCT)
            if (
                1 <= len(core) <= min(MAX_WORD_CHARS, limit)
                and _WORD_RE.fullmatch(core)
                and all(ch in supported for ch in core)
            ):
                types[core] = None
        candidates = list(types)
        if len(candidates) > POOL_TOKENS_PER_DOC:
            candidates = rng.sample(candidates, POOL_TOKENS_PER_DOC)
        joined = " ".join(candidates)
        if normalize(joined) != joined:  # unusual normalizer: keep fixed points only
            candidates = [w for w in candidates if normalize(w) == w]
        for word in candidates:
            if len(self.words) >= POOL_MAX_TYPES:
                break
            if word in self._seen:
                continue
            self._seen.add(word)
            self.words.append(word)
            if word.isalpha():
                self.by_len.setdefault(len(word), []).append(word)


# ===========================================================================
# ITEM BUILDER
# ===========================================================================


def _needs_vocab(st: dict) -> bool:
    return st["kind"] in ("word_salad", "mixed_tokens") or (
        st["kind"] == "chars" and st["vocab_fraction"] > 0
    )


def _assign_sizes(st: dict, count: int, seed) -> list[int]:
    """Concatenated seeded permutations of the sizes: per-size counts differ by <= 1."""
    rng = random.Random(f"{seed}/{st['name']}/sizes")
    out: list[int] = []
    while len(out) < count:
        perm = st["sizes"][:]
        rng.shuffle(perm)
        out.extend(perm)
    return out[:count]


class _Run:
    """State of one ``build_items`` call: document feed, vocabulary, uniqueness."""

    def __init__(self, plan: dict, seed, docs, normalize, supported_chars):
        self.max_chars = plan["max_chars"]
        self.seed = seed
        self.normalize = normalize
        self.supported = supported_chars
        self._docs = iter(docs)
        self.docs_consumed = 0
        self.docs_skipped = 0
        self.vocab = _Vocab()
        self.seen: set[str] = set()
        self.retry_hist: Counter = Counter()
        self.invariant_retries = 0
        self.dup_accepted = 0
        self.per_stratum: dict[str, dict] = {}
        self._builders = {
            "chars": self._b_chars,
            "digits": self._b_digits,
            "alnum": self._b_alnum,
            "words": self._b_words,
            "shuffled_words": self._b_shuffled_words,
            "scrambled_chars": self._b_scrambled_chars,
            "shuffled_sentences": self._b_shuffled_sentences,
            "word_salad": self._b_word_salad,
            "mixed_tokens": self._b_mixed_tokens,
            "paragraphs": self._b_paragraphs,
        }

    # -- documents ----------------------------------------------------------

    def next_doc(self, where: str) -> _Doc:
        """Next non-empty normalized document (feeds the vocabulary pool)."""
        empty_run = 0
        while True:
            try:
                raw = next(self._docs)
            except StopIteration:
                raise RuntimeError(
                    f"text plan stratum {where!r}: documents exhausted after "
                    f"{self.docs_consumed} documents (vocabulary types so far: "
                    f"{len(self.vocab.words)}); supply more documents or lower n"
                ) from None
            self.docs_consumed += 1
            words = []
            if isinstance(raw, str):
                words = self.normalize(raw).replace("\n", " ").split()
            if not words:
                self.docs_skipped += 1
                empty_run += 1
                if empty_run >= MAX_EMPTY_DOCS:
                    raise RuntimeError(
                        f"text plan stratum {where!r}: {empty_run} consecutive "
                        "documents were empty after normalization"
                    )
                continue
            pool_rng = random.Random(f"{self.seed}/pool/{self.docs_consumed}")
            self.vocab.add_doc(
                words, pool_rng, self.supported, self.max_chars, self.normalize
            )
            return _Doc(words)

    def fill_vocab(self, where: str) -> None:
        """Consume documents until the pool holds POOL_MIN_TYPES distinct types."""
        while len(self.vocab.words) < POOL_MIN_TYPES:
            self.next_doc(where)

    # -- invariants ---------------------------------------------------------

    def violation(self, text: str, kind: str, size: int) -> Optional[str]:
        """First violated invariant of section 4.4, or None."""
        if not text:
            return "empty text"
        if len(text) > self.max_chars:
            return f"{len(text)} characters > max_chars {self.max_chars}"
        if text != text.strip():
            return "leading or trailing whitespace"
        if "  " in text:
            return "double space"
        newlines = text.count("\n")
        if kind == "paragraphs":
            if newlines != size - 1 or not all(text.split("\n")):
                return "paragraph count or an empty paragraph"
        elif newlines:
            return "newline outside paragraphs"
        for ch in text:
            if ch not in self.supported:
                return f"unsupported character {ch!r}"
        if self.normalize(text) != text:
            return "text is not a fixed point of normalize"
        return None

    def make_item(self, st: dict, i: int, size: int) -> dict:
        name, kind = st["name"], st["kind"]
        rng = random.Random(f"{self.seed}/{name}/{i}")
        build = self._builders[kind]
        docs_before = self.docs_consumed
        doc = None
        docs_used = dups = bad = 0
        while True:
            if kind in _SINGLE_DOC_KINDS and doc is None:
                if docs_used >= MAX_DOCS_PER_ITEM:
                    raise RuntimeError(
                        f"text plan stratum {name!r}: no usable window of size "
                        f"{size} in {docs_used} consecutive documents "
                        f"(max_chars={self.max_chars})"
                    )
                doc = self.next_doc(name)
                docs_used += 1
            candidate = build(st, rng, size, doc)
            if candidate is None:
                doc = None
                continue
            text, before = candidate
            why = self.violation(text, kind, size)
            if why is not None:
                bad += 1
                self.invariant_retries += 1
                if bad > MAX_BAD_DRAWS:
                    raise RuntimeError(
                        f"text plan stratum {name!r}: cannot satisfy the "
                        f"invariants for size {size}: {why}"
                    )
                doc = None
                continue
            if text in self.seen and dups < MAX_UNIQUE_RETRIES:
                dups += 1
                continue
            break
        entry = self.per_stratum.setdefault(
            name, {"items": 0, "unique_retries": 0, "dup_accepted": 0, "docs": 0}
        )
        entry["items"] += 1
        entry["unique_retries"] += dups
        entry["docs"] += self.docs_consumed - docs_before
        if text in self.seen:
            self.dup_accepted += 1
            entry["dup_accepted"] += 1
        self.seen.add(text)
        self.retry_hist[dups] += 1
        info = {
            "stratum": name,
            "kind": kind,
            "size": size,
            "n_chars": len(text),
            "n_words": len(text.split()),
            "shuffle": _SHUFFLE_OF.get(kind, "none"),
        }
        if kind in _HAS_BEFORE:
            info["before_shuffle"] = before
        return {"text": text, "info": info}

    # -- builders: (stratum, rng, size, doc) -> (text, before) or None ------

    def _b_chars(self, st, rng, size, doc):
        fraction = st["vocab_fraction"]
        if fraction > 0 and rng.random() < fraction:
            words = self.vocab.by_len.get(size)
            if words:
                return rng.choice(words), None
        return _gen_chars(rng, size, st["charset"]), None

    def _b_digits(self, st, rng, size, doc):
        return _gen_digits(rng, size, st["formats"], self.max_chars), None

    def _b_alnum(self, st, rng, size, doc):
        return _gen_alnum(rng, size, st["styles"], self.max_chars), None

    def _b_words(self, st, rng, size, doc):
        tokens = _find_window(doc.words, size, rng, self.max_chars)
        return (" ".join(tokens), None) if tokens else None

    def _b_shuffled_words(self, st, rng, size, doc):
        tokens = _find_window(
            doc.words, size, rng, self.max_chars, accept=lambda t: len(set(t)) > 1
        )
        if tokens is None:
            return None
        return " ".join(_reorder(tokens, rng)), " ".join(tokens)

    def _b_scrambled_chars(self, st, rng, size, doc):
        tokens = _find_window(
            doc.words,
            size,
            rng,
            self.max_chars,
            accept=lambda t: any(map(_eligible, t)),
        )
        if tokens is None:
            return None
        eligible = [j for j, word in enumerate(tokens) if _eligible(word)]
        take = int(len(eligible) * st["word_fraction"] + 0.5)
        chosen = set(rng.sample(eligible, max(1, min(len(eligible), take))))
        out = [
            _scramble_word(word, rng) if j in chosen else word
            for j, word in enumerate(tokens)
        ]
        return " ".join(out), " ".join(tokens)

    def _b_shuffled_sentences(self, st, rng, size, doc):
        spans = doc.sentences()
        if len(spans) < 2:
            return None
        tight = max(8, size // 2)
        best = None
        best_over = 0
        if len(spans) - 1 <= PROBES:
            firsts = list(range(len(spans) - 1))
            rng.shuffle(firsts)
        else:
            firsts = rng.sample(range(len(spans) - 1), PROBES)
        for first in firsts:
            total = 0
            last = first
            while last < len(spans) and (total < size or last - first < 2):
                total += spans[last][1] - spans[last][0]
                last += 1
            if last - first < 2 or total < size:
                continue
            tokens = doc.words[spans[first][0] : spans[last - 1][1]]
            if sum(map(len, tokens)) + len(tokens) - 1 > self.max_chars:
                continue
            if not _natural_ok(tokens):
                continue
            sentences = [" ".join(doc.words[a:b]) for a, b in spans[first:last]]
            if len(set(sentences)) < 2:
                continue
            if best is None or total - size < best_over:
                best, best_over = sentences, total - size
            if best_over <= tight:
                break
        if best is None:
            return None
        return " ".join(_reorder(best, rng)), " ".join(best)

    def _b_word_salad(self, st, rng, size, doc):
        pool = self.vocab.words
        if not pool:
            raise RuntimeError(
                f"text plan stratum {st['name']!r}: empty vocabulary pool"
            )
        text = ""
        for _ in range(20):
            if size <= len(pool):
                text = " ".join(rng.sample(pool, size))
            else:
                text = " ".join(rng.choice(pool) for _ in range(size))
            if len(text) <= self.max_chars:
                break
        return text, None

    def _b_mixed_tokens(self, st, rng, size, doc):
        pool = self.vocab.words
        if not pool:
            raise RuntimeError(
                f"text plan stratum {st['name']!r}: empty vocabulary pool"
            )
        cap = self.max_chars
        tokens = []
        for _ in range(size):
            kind = _wchoice(rng, st["token_mix"])
            if kind == "word":
                tokens.append(rng.choice(pool))
            elif kind == "digits":
                tokens.append(_gen_digits(rng, rng.randint(1, 6), _TOKEN_DIGITS, cap))
            elif kind == "alnum":
                tokens.append(_gen_alnum(rng, rng.randint(2, 8), _TOKEN_ALNUM, cap))
            elif kind == "word_punct":
                tokens.append(rng.choice(pool) + rng.choice(_TRAIL))
            else:
                tokens.append(rng.choice(_CHARSETS["alnum_punct"]))
        return " ".join(tokens), None

    def _b_paragraphs(self, st, rng, size, doc):
        name = st["name"]
        lo, hi = st["words_per_paragraph"]
        counts = [rng.randint(lo, hi) for _ in range(size)]
        while sum(2 * c - 1 for c in counts) + size - 1 > self.max_chars:
            longest = max(range(size), key=lambda q: counts[q])
            if counts[longest] <= lo:
                break
            counts[longest] -= 1
        parts: list[str] = []
        left = self.max_chars - (size - 1)  # characters left for paragraph text
        for j, k in enumerate(counts):
            rest = counts[j + 1 :]
            # a paragraph may use its proportional share of the words still to place
            # (never less than its minimum); this keeps later paragraphs feasible
            share = left * k // (k + sum(rest))
            limit = min(left - sum(2 * c - 1 for c in rest), max(share, 2 * k - 1))
            tokens = None
            for _ in range(MAX_DOCS_PER_ITEM):
                tokens = _find_window(self.next_doc(name).words, k, rng, limit)
                if tokens is not None:
                    break
            if tokens is None:
                raise RuntimeError(
                    f"text plan stratum {name!r}: no usable {k}-word paragraph in "
                    f"{MAX_DOCS_PER_ITEM} consecutive documents "
                    f"(limit {limit} characters)"
                )
            parts.append(" ".join(tokens))
            left -= len(parts[-1])
        return "\n".join(parts), None


def build_items(
    plan: dict,
    n: int,
    seed: int,
    docs,
    normalize,
    supported_chars,
    stats: Optional[dict] = None,
) -> list[dict]:
    """Build exactly ``n`` ``{"text", "info"}`` items from a plan.

    ``docs`` is any iterator of raw document strings (pulled lazily, never
    drained), ``normalize`` the generator's ``_normalize_source_text`` and
    ``supported_chars`` its ``SUPPORTED_ASCII_CHARS``. Output is deterministic
    for a given plan, n, seed and document sequence, and comes back in a seeded
    random order so consecutive chunks mix strata. Raises ``RuntimeError`` naming
    the stratum if documents run out. If ``stats`` is a dict it is filled with
    document, vocabulary, uniqueness and retry counters.
    """
    plan = load_plan(plan)
    n = _check_n(n)
    allocation = allocate(plan, n)
    run = _Run(plan, seed, docs, normalize, supported_chars)
    vocab_users = [st["name"] for st, count in allocation if count and _needs_vocab(st)]
    if vocab_users:
        run.fill_vocab(vocab_users[0])
    items: list[dict] = []
    for st, count in allocation:
        for i, size in enumerate(_assign_sizes(st, count, seed) if count else ()):
            items.append(run.make_item(st, i, size))
    random.Random(f"{seed}/order").shuffle(items)
    if stats is not None:
        stats.update(
            n=n,
            docs_consumed=run.docs_consumed,
            docs_skipped=run.docs_skipped,
            vocab_types=len(run.vocab.words),
            unique_texts=len(run.seen),
            dup_accepted=run.dup_accepted,
            invariant_retries=run.invariant_retries,
            retry_hist=dict(sorted(run.retry_hist.items())),
            per_stratum=run.per_stratum,
        )
    return items


# ===========================================================================
# CLI
# ===========================================================================


def default_normalize(text: str) -> str:
    """Copy of the generator's ``_normalize_source_text`` (used by the CLI only)."""
    replacements = {
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u2019": "'",
        "\u2018": "'",
        "`": "'",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2026": "...",
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
        elif ch in DEFAULT_SUPPORTED_CHARS:
            filtered.append(ch)
        else:
            filtered.append(" ")
    cleaned = "\n".join(
        " ".join(part.split()) for part in "".join(filtered).split("\n")
    )
    return cleaned.strip()


def _iter_corpus(path: str):
    """Lazily yield the ``text`` of every JSON line of a corpus file."""
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            text = obj.get("text", "") if isinstance(obj, dict) else obj
            yield text if isinstance(text, str) else ""


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Show the allocation and example items of a braille OCR text plan"
    )
    parser.add_argument("--plan", required=True, help="plan JSON path")
    parser.add_argument("--n", type=int, required=True, help="number of items")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--corpus",
        required=True,
        help='JSONL file, one {"text": ...} document per line',
    )
    parser.add_argument(
        "--show", type=int, default=10, help="example items to print (default 10)"
    )
    parser.add_argument(
        "--per_stratum",
        action="store_true",
        help="print --show examples of every stratum instead of the first --show items",
    )
    args = parser.parse_args(argv)
    try:
        plan = load_plan(args.plan)
        print(describe_allocation(plan, args.n))
        stats: dict = {}
        items = build_items(
            plan,
            args.n,
            args.seed,
            _iter_corpus(args.corpus),
            default_normalize,
            DEFAULT_SUPPORTED_CHARS,
            stats=stats,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    print(
        f"\ndocs consumed: {stats['docs_consumed']} | "
        f"vocabulary types: {stats['vocab_types']} | "
        f"unique texts: {stats['unique_texts']}/{stats['n']} | "
        f"duplicates accepted: {stats['dup_accepted']}"
    )
    if args.per_stratum:
        rank = {st["name"]: k for k, st in enumerate(plan["strata"])}
        shown: Counter = Counter()
        chosen = []
        for item in items:
            name = item["info"]["stratum"]
            if shown[name] < args.show:
                shown[name] += 1
                chosen.append(item)
        chosen.sort(key=lambda it: rank[it["info"]["stratum"]])
    else:
        chosen = items[: max(0, args.show)]
    for item in chosen:
        info = item["info"]
        print(
            f"[{info['stratum']}] size={info['size']} chars={info['n_chars']} "
            f"words={info['n_words']} shuffle={info['shuffle']}\n    {item['text']!r}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
