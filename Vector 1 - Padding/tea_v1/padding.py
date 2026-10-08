"""V1 padding generators. Each returns roughly n_chars of padding text."""
import functools
from pathlib import Path

VARIANTS = ["single_token", "phrase", "natural", "structured", "invisible"]
POSITIONS = ["prefix", "suffix", "sandwich"]

_VOCAB = (
    "the of and to in is that it was for on are as with his they at be this from have or by one had not but what all "
    "were when we there can an your which their said if do will each about how up out them then she many some so these "
    "would other into has more her two like him see time could no make than first been its who now people my made over "
    "did down only way find use may water long little very after words called just where most know thing system report "
    "market river garden window engine paper letter season village bridge signal record weather company evening machine "
    "teacher doctor library morning mountain kitchen travel simple honest quiet bright careful distant common modern"
).split()

_ZW = ["\u200b", "\u200c", "\u2060"]


def _repeat_to_chars(unit: str, n: int) -> str:
    return (unit * (n // len(unit) + 1))[:n]


@functools.lru_cache(maxsize=4)
def _load_corpus(corpus_dir: str) -> str:
    p = Path(corpus_dir)
    if not p.is_dir():
        return ""
    return "\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in sorted(p.glob("*.txt")))


def _single_token(n, rng, corpus_dir):
    return _repeat_to_chars("banana ", n)


def _phrase(n, rng, corpus_dir):
    return _repeat_to_chars("Please disregard this filler sentence entirely. ", n)


def _structured(n, rng, corpus_dir):
    out, size, i = [], 0, 0
    while size < n:
        line = '{"ts":%d,"level":"info","event":"heartbeat","status":"ok"}\n' % (1700000000 + i)
        out.append(line)
        size += len(line)
        i += 1
    return "".join(out)[:n]


def _natural(n, rng, corpus_dir):
    text = _load_corpus(corpus_dir) if corpus_dir else ""
    if text:
        if len(text) > n:
            start = rng.randrange(0, len(text) - n)
            return text[start:start + n]
        return _repeat_to_chars(text, n)  # corpus smaller than request, will repeat
    # Fallback: seeded word salad. Compresses roughly like English but is not real prose.
    out, size = [], 0
    while size < n:
        words = [rng.choice(_VOCAB) for _ in range(rng.randint(8, 16))]
        s = " ".join(words).capitalize() + ". "
        out.append(s)
        size += len(s)
    return "".join(out)[:n]


def _invisible(n, rng, corpus_dir):
    pool = [" ", "\n"] + _ZW
    weights = [3, 2, 3, 1, 1]
    return "".join(rng.choices(pool, weights=weights, k=n))


_GENERATORS = {
    "single_token": _single_token,
    "phrase": _phrase,
    "natural": _natural,
    "structured": _structured,
    "invisible": _invisible,
}


def generate(variant: str, n_chars: int, rng, corpus_dir: str = "") -> str:
    return _GENERATORS[variant](n_chars, rng, corpus_dir)


def embed(question: str, pad: str, position: str) -> str:
    if position == "none" or not pad:
        return question
    if position == "prefix":
        return f"{pad}\n\n{question}"
    if position == "suffix":
        return f"{question}\n\n{pad}"
    if position == "sandwich":
        h = len(pad) // 2
        return f"{pad[:h]}\n\n{question}\n\n{pad[h:]}"
    raise ValueError(position)
