"""Input-layer detector for V1. Returns allow / flag / block with the reasons.

Stateless checks: repetition, compression ratio, invisible characters, whitespace share, token-to-char ratio.
Stateful check: per-session baseline of prompt size.
Every threshold lives in DetectorConfig so the evaluation can sweep them.
"""
import statistics
import unicodedata
import zlib
from collections import deque
from dataclasses import dataclass, field

ALLOW, FLAG, BLOCK = 0, 1, 2
ACTIONS = {ALLOW: "allow", FLAG: "flag", BLOCK: "block"}


@dataclass
class DetectorConfig:
    enable_baseline: bool = True
    min_chars_for_stats: int = 1500      # compression is meaningless on short text
    min_chars_for_unicode: int = 200
    # repetition: share of duplicate word n-grams
    rep_ngram: int = 5
    rep_min_words: int = 100
    rep_flag: float = 0.35
    rep_block: float = 0.70
    # compression: zlib size / raw size (lower means more redundant)
    comp_flag: float = 0.18
    comp_block: float = 0.06
    # invisible / control characters and whitespace as a share of all characters
    invis_flag: float = 0.01
    invis_block: float = 0.10
    ws_flag: float = 0.60
    ws_block: float = 0.85
    # tokens per character (only used with a real tokenizer)
    tpc_flag: float = 0.60
    tpc_block: float = 1.00
    cjk_skip_share: float = 0.20         # CJK text is naturaly token-dense, skip the ratio check if you have too
    # per-session baseline of prompt tokens
    baseline_window: int = 20
    baseline_min_history: int = 3
    baseline_flag_mult: float = 4.0
    baseline_block_mult: float = 10.0
    baseline_flag_min_tokens: int = 1000
    baseline_block_min_tokens: int = 4000
    cold_flag_tokens: int = 3000         # absolute limits while a session has no history, hard to get this right..
    cold_block_tokens: int = 12000


@dataclass
class CheckResult:
    name: str
    value: float
    level: int


@dataclass
class Decision:
    level: int
    tokens: int
    chars: int
    checks: list = field(default_factory=list)

    @property
    def action(self):
        return ACTIONS[self.level]

    def fired(self):
        return [f"{c.name}={c.value:.3f}" for c in self.checks if c.level > ALLOW]


def _level(value, flag, block, higher_is_worse=True):
    if higher_is_worse:
        return BLOCK if value >= block else FLAG if value >= flag else ALLOW
    return BLOCK if value <= block else FLAG if value <= flag else ALLOW


def _is_cjk(c):
    o = ord(c)
    return 0x4E00 <= o <= 0x9FFF or 0x3040 <= o <= 0x30FF or 0xAC00 <= o <= 0xD7AF


class Detector:
    def __init__(self, count_tokens, tokens_reliable=False, cfg=None):
        self.count = count_tokens
        self.tokens_reliable = tokens_reliable
        self.cfg = cfg or DetectorConfig()
        self.history = {}

    # ---- stateless checks -------------------------------------------------
    def _repetition(self, text):
        c = self.cfg
        words = text.split()
        if len(words) < c.rep_min_words:
            return CheckResult("repetition", 0.0, ALLOW)
        n = c.rep_ngram
        grams = [hash(tuple(words[i:i + n])) for i in range(len(words) - n + 1)]
        ratio = 1 - len(set(grams)) / len(grams)
        return CheckResult("repetition", ratio, _level(ratio, c.rep_flag, c.rep_block))

    def _compression(self, text):
        c = self.cfg
        if len(text) < c.min_chars_for_stats:
            return CheckResult("compression", 1.0, ALLOW)
        raw = text.encode("utf-8")
        ratio = len(zlib.compress(raw, 6)) / len(raw)
        return CheckResult("compression", ratio, _level(ratio, c.comp_flag, c.comp_block, higher_is_worse=False))

    def _unicode(self, text):
        c = self.cfg
        n = len(text)
        if n < c.min_chars_for_unicode:
            return [CheckResult("invisible_chars", 0.0, ALLOW), CheckResult("whitespace_share", 0.0, ALLOW)]
        inv = ws = 0
        for ch in text:
            cat = unicodedata.category(ch)
            if cat == "Cf" or (cat == "Cc" and ch not in "\n\r\t"):
                inv += 1
            if ch.isspace():
                ws += 1
        return [
            CheckResult("invisible_chars", inv / n, _level(inv / n, c.invis_flag, c.invis_block)),
            CheckResult("whitespace_share", ws / n, _level(ws / n, c.ws_flag, c.ws_block)),
        ]

    def _token_char_ratio(self, text, tokens):
        c = self.cfg
        if not self.tokens_reliable or not text:
            return CheckResult("tokens_per_char", 0.0, ALLOW)
        if sum(1 for ch in text if _is_cjk(ch)) / len(text) > c.cjk_skip_share:
            return CheckResult("tokens_per_char", 0.0, ALLOW)
        r = tokens / len(text)
        return CheckResult("tokens_per_char", r, _level(r, c.tpc_flag, c.tpc_block))

    # ---- stateful check ---------------------------------------------------
    def _baseline(self, session_id, tokens):
        c = self.cfg
        hist = self.history.setdefault(session_id, deque(maxlen=c.baseline_window))
        if len(hist) >= c.baseline_min_history:
            base = max(statistics.median(hist), 1)
            ratio = tokens / base
            if tokens >= c.baseline_block_min_tokens and ratio >= c.baseline_block_mult:
                lvl = BLOCK
            elif tokens >= c.baseline_flag_min_tokens and ratio >= c.baseline_flag_mult:
                lvl = FLAG
            else:
                lvl = ALLOW
            return CheckResult("baseline_ratio", ratio, lvl)
        lvl = BLOCK if tokens >= c.cold_block_tokens else FLAG if tokens >= c.cold_flag_tokens else ALLOW
        return CheckResult("cold_start_tokens", float(tokens), lvl)

    # ---- public -----------------------------------------------------------
    def inspect(self, text, session_id="default"):
        tokens = self.count(text)
        checks = [self._repetition(text), self._compression(text)]
        checks += self._unicode(text)
        checks.append(self._token_char_ratio(text, tokens))
        if self.cfg.enable_baseline:
            checks.append(self._baseline(session_id, tokens))
        level = max(c.level for c in checks)
        # Only allowed traffic updates the baseline, so flagged input cannot drag it upwards.
        if self.cfg.enable_baseline and level == ALLOW:
            self.history[session_id].append(tokens)
        return Decision(level=level, tokens=tokens, chars=len(text), checks=checks)
