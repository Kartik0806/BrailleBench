"""Source texts: FineWeb windows and text files."""

import random
from .braille_text import SUPPORTED_ASCII_CHARS, _normalize_source_text, strip_punct_edge_tokens


DEMO_TEXTS = ["The quick brown fox jumps over the lazy dog."]


def _random_text_window(
    text: str,
    min_chars: int = 1,
    max_chars: int = 420,
    min_words: int = 1,
    max_words: int = 10,
) -> str:
    """Return a random contiguous whole-word window from normalized source text.

    A target word count and target character budget are sampled uniformly from
    the requested ranges for every example. Whole words are never cut in half,
    so the actual character count is the feasible whole-word length nearest the
    sampled character target while never exceeding ``max_chars``. The sampled
    word count is honored whenever a window of that size fits the character cap.
    """
    text = _normalize_source_text(text)
    words = text.replace("\n", " ").split()
    if not words:
        return ""

    min_chars = max(1, int(min_chars))
    max_chars = max(min_chars, int(max_chars))
    min_words = max(1, int(min_words))
    max_words = max(min_words, int(max_words))
    max_words = min(max_words, len(words))
    min_words = min(min_words, max_words)

    target_words = random.randint(min_words, max_words)
    target_chars = random.randint(min_chars, max_chars)

    def candidates_for(k: int) -> list[tuple[int, str]]:
        n_starts = len(words) - k + 1
        if n_starts <= 0:
            return []

        # FineWeb documents can be very long. Probe a bounded random set of
        # contiguous starts instead of scanning every possible window.
        max_probes = min(96, n_starts)
        if n_starts <= max_probes:
            starts = list(range(n_starts))
            random.shuffle(starts)
        else:
            starts = random.sample(range(n_starts), max_probes)

        out = []
        for start in starts:
            phrase = " ".join(words[start : start + k])
            n_chars = len(phrase)
            if n_chars <= max_chars:
                out.append((n_chars, phrase))
        return out

    # First try to honor the uniformly sampled word count exactly.
    candidates = candidates_for(target_words)

    # If that word count cannot fit under max_chars, choose uniformly among
    # the other feasible word counts rather than returning an empty sample.
    if not candidates:
        feasible_by_words = []
        other_counts = [
            k for k in range(min_words, max_words + 1) if k != target_words
        ]
        random.shuffle(other_counts)
        for k in other_counts:
            cands = candidates_for(k)
            if cands:
                feasible_by_words.append((k, cands))
        if not feasible_by_words:
            return ""
        _, candidates = random.choice(feasible_by_words)

    # Prefer windows inside the requested [min_chars, max_chars] range. If the
    # corpus has no such whole-word window, keep the nearest valid <= max_chars
    # candidate so generation can continue.
    in_range = [(n, phrase) for n, phrase in candidates if n >= min_chars]
    pool = in_range if in_range else candidates
    best_distance = min(abs(n - target_chars) for n, _ in pool)
    best = [phrase for n, phrase in pool if abs(n - target_chars) == best_distance]
    return random.choice(best)


def get_texts_from_file(
    path: str,
    num_samples: int,
    min_words: int = 1,
    max_words: int = 10,
    seed: int = 1337,
    min_chars: int = 1,
    max_chars: int = 420,
    strip_punct_tokens: bool = False,
) -> list[str]:
    """Offline alternative to get_data(): one text per line in a UTF-8 file."""
    rng = random.Random(seed)
    pool = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            t = _normalize_source_text(line.strip())
            if strip_punct_tokens:
                t = strip_punct_edge_tokens(t)
            n = len(t.split())
            if (t and min_words <= n <= max_words and min_chars <= len(t) <= max_chars
                    and all(ch in SUPPORTED_ASCII_CHARS for ch in t)):
                pool.append(t)
    if not pool:
        raise RuntimeError(f"{path}: no usable lines ({min_words}-{max_words} words, supported chars)")
    if len(pool) >= num_samples:
        return rng.sample(pool, num_samples)
    return [rng.choice(pool) for _ in range(num_samples)]


def get_data(
    dataset_name: str = "HuggingFaceFW/fineweb",
    dataset_config: str = "sample-10BT",
    num_samples: int = 1000,
    min_text_chars: int = 1,
    max_text_chars: int = 420,
    min_text_words: int = 1,
    max_text_words: int = 10,
    seed: int = 1337,
    strip_punct_tokens: bool = False,
) -> list[str]:
    """Stream FineWeb and collect randomized, braille-safe text windows."""
    try:
        import datasets
    except ImportError as exc:
        raise ImportError(
            "Install Hugging Face datasets: pip install datasets"
        ) from exc

    ds = datasets.load_dataset(
        dataset_name,
        name=dataset_config or None,
        split="train",
        streaming=True,
    )
    try:
        ds = ds.shuffle(seed=seed, buffer_size=max(1000, min(10000, num_samples * 10)))
    except Exception:
        pass

    texts = []
    random.seed(seed)
    for example in ds:
        raw = example.get("text", "")
        text = _random_text_window(
            raw,
            min_chars=min_text_chars,
            max_chars=max_text_chars,
            min_words=min_text_words,
            max_words=max_text_words,
        )
        if not text:
            continue
        if strip_punct_tokens:
            text = strip_punct_edge_tokens(text)
            if not text:
                continue
        # One-word samples are intentionally allowed.
        if len(text.split()) < min_text_words:
            continue
        # Strong invariant: every retained source character is translatable.
        if any(ch not in SUPPORTED_ASCII_CHARS for ch in text):
            continue
        texts.append(text)
        if len(texts) >= num_samples:
            break
    return texts
