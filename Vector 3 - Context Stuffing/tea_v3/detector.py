"""Input-layer detector for V3. Looks at the set of chunks a retriever is about to stuff into
context for one query, and returns allow / flag / block with the reasons.

Retrieval chunk checks are the paper's input-layer control for context window stuffing: duplicate
or near-duplicate chunks (an attacker-controlled document repeating the same paragraph so it
crowds out the top_k slots), an abnormally large chunk (a document with no paragraph breaks, so a
naive splitter can't cap it), and irrelevant chunks (keyword-stuffed content that gets retrieved on
a single matching word but contributes little real information for the query).
"""
import re
from dataclasses import dataclass, field

ALLOW, FLAG, BLOCK = 0, 1, 2
ACTIONS = {ALLOW: "allow", FLAG: "flag", BLOCK: "block"}

_WORD_RE = re.compile(r"\w+")


@dataclass
class DetectorConfig:
    # near-duplicate detection: k-word shingles, Jaccard similarity above this counts as a duplicate pair
    shingle_k: int = 8
    dup_jaccard_threshold: float = 0.6
    dup_flag: float = 0.2            # share of retrieved chunks that duplicate an earlier one
    dup_block: float = 0.4
    oversized_flag: float = 1.5      # chunk chars / normal_chunk_chars
    oversized_block: float = 3.0
    relevance_min: float = 0.05      # query/chunk word-overlap below this counts as irrelevant
    irrelevant_flag: float = 0.3     # share of retrieved chunks below relevance_min
    irrelevant_block: float = 0.6
    inflation_flag: float = 1.5      # total retrieved chars / (top_k * normal_chunk_chars)
    inflation_block: float = 3.0


@dataclass
class CheckResult:
    name: str
    value: float
    level: int


@dataclass
class Decision:
    level: int
    chunks: int
    chars: int
    checks: list = field(default_factory=list)

    @property
    def action(self):
        return ACTIONS[self.level]

    def fired(self):
        return [f"{c.name}={c.value:.3f}" for c in self.checks if c.level > ALLOW]


def _level(value, flag, block):
    return BLOCK if value >= block else FLAG if value >= flag else ALLOW


def _shingles(text, k):
    words = _WORD_RE.findall(text.lower())
    if len(words) < k:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + k]) for i in range(len(words) - k + 1)}


def _jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class Detector:
    def __init__(self, cfg=None):
        self.cfg = cfg or DetectorConfig()

    def _duplicate_ratio(self, chunks, top_k):
        c = self.cfg
        seen = []
        dup_count = 0
        for chunk in chunks:
            shingles = _shingles(chunk.text, c.shingle_k)
            is_dup = any(_jaccard(shingles, s) >= c.dup_jaccard_threshold for s in seen)
            if is_dup:
                dup_count += 1
            seen.append(shingles)
        # Against top_k (retrieval capacity), not len(chunks): stuffing means duplicates crowding
        # out most of the available slots. A sparse retrieval of only 1-2 incidentally-matching
        # chunks, one of which happens to duplicate the other, is not the same signal -- dividing by
        # the tiny actual count would swing the ratio from 0 to 0.5+ on a single coincidental pair.
        ratio = dup_count / max(top_k, 1)
        return CheckResult("duplicate_ratio", ratio, _level(ratio, c.dup_flag, c.dup_block))

    def _oversized_ratio(self, chunks, normal_chunk_chars):
        c = self.cfg
        ratio = max((ch.chars for ch in chunks), default=0) / max(normal_chunk_chars, 1)
        return CheckResult("oversized_ratio", ratio, _level(ratio, c.oversized_flag, c.oversized_block))

    def _irrelevant_share(self, chunks, top_k):
        c = self.cfg
        irrelevant = sum(1 for ch in chunks if ch.score < c.relevance_min)
        # Same reasoning as duplicate_ratio: measured against retrieval capacity, not just what came
        # back, so one weakly-relevant chunk out of a near-empty retrieval doesn't look like "100%
        # irrelevant stuffing" the way it would if the corpus had genuinely been flooded.
        share = irrelevant / max(top_k, 1)
        return CheckResult("irrelevant_share", share, _level(share, c.irrelevant_flag, c.irrelevant_block))

    def _context_inflation(self, chunks, top_k, normal_chunk_chars):
        c = self.cfg
        expected = max(top_k * normal_chunk_chars, 1)
        ratio = sum(ch.chars for ch in chunks) / expected
        return CheckResult("context_inflation", ratio, _level(ratio, c.inflation_flag, c.inflation_block))

    def inspect(self, chunks, top_k, normal_chunk_chars):
        checks = [
            self._duplicate_ratio(chunks, top_k),
            self._oversized_ratio(chunks, normal_chunk_chars),
            self._irrelevant_share(chunks, top_k),
            self._context_inflation(chunks, top_k, normal_chunk_chars),
        ]
        level = max(c.level for c in checks)
        return Decision(level=level, chunks=len(chunks), chars=sum(ch.chars for ch in chunks), checks=checks)
