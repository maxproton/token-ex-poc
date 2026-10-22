"""Input-layer detector for V2. Returns allow / flag / block with the reasons.

V1's detector already covers repetition, compression, unicode anomalies and session baseline
(Table 2: input layer addresses V1, V2 and V3 with the same shared checks). This detector adds the
two signals the paper calls out as specific to recursive prompting: instruction nesting depth
(stacked recursive directives) and runaway-generation language (instructions demanding unbounded
or self-referential continuation). Both are stateless and look at a single prompt in isolation.
"""
import re
from dataclasses import dataclass, field

ALLOW, FLAG, BLOCK = 0, 1, 2
ACTIONS = {ALLOW: "allow", FLAG: "flag", BLOCK: "block"}

# Chained/recursive directives: language that tells the model to feed a step's output into the
# next step, or to repeat/redo a step, rather than just answering once.
_NESTING_PATTERNS = [
    r"\bthen take (?:that|this|your) (?:answer|output|response|result)\b",
    r"\bfeed (?:the|your) (?:result|output|answer) back\b",
    r"\brepeat (?:the above|this|that|step)\b",
    r"\bgo back to step\b",
    r"\bonce more\b",
    r"\bonce again\b",
    r"\bcritique (?:your|that|the) (?:own )?(?:answer|revision)\b",
    r"\brevise (?:it|your answer|based on|that)\b",
    r"\bapply (?:this|the above) to (?:your own|its own) output\b",
    r"\bfollow this instruction\b",            # recursive quoting wrapper, one hit per nesting level
    r"\btake the output of\b",                 # chained numbered steps consuming each other
    r"\bstep \d+\s*[:.].*step \d+\s*[:.]",      # at least two explicit numbered steps in one prompt
]

# Instructions demanding the model never reach a natural stopping point.
_RUNAWAY_PATTERNS = [
    r"\bforever\b",
    r"\bindefinitely\b",
    r"\bnever stop\b",
    r"\bnever end(?:ing)?\b",
    r"\bwithout stopping\b",
    r"\bdo not stop\b",
    r"\bdon'?t stop\b",
    r"\bas long as possible\b",
    r"\bkeep (?:going|generating|repeating|expanding)\b",
    r"\bcontinue (?:until i say stop|indefinitely|forever)\b",
    r"\bno limit\b",
    r"\bunlimited\b",
    r"\binfinite(?:ly)?\b",
    r"\bendless(?:ly)?\b",
]

# References to the model's own prior output, used to set up a self-feeding loop within one prompt.
_SELF_REF_PATTERNS = [
    r"\byour previous answer\b",
    r"\byour last (?:answer|response)\b",
    r"\byour own (?:output|answer|response)\b",
    r"\bwhat you just (?:said|wrote|generated|answered)\b",
    r"\bthe (?:answer|response) you just gave\b",
    r"\byour prior (?:output|answer|response)\b",
]

_NESTING_RE = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _NESTING_PATTERNS]
_RUNAWAY_RE = [re.compile(p, re.IGNORECASE) for p in _RUNAWAY_PATTERNS]
_SELF_REF_RE = [re.compile(p, re.IGNORECASE) for p in _SELF_REF_PATTERNS]

_OPENERS = {"(": ")", "[": "]", "{": "}"}
_CLOSERS = {v: k for k, v in _OPENERS.items()}


@dataclass
class DetectorConfig:
    # instruction nesting depth: count of chained-directive matches, plus bracket/quote depth
    nesting_flag: float = 2
    nesting_block: float = 4
    # runaway-generation phrase count
    runaway_flag: float = 1
    runaway_block: float = 3
    # self-reference to the model's own prior output
    self_ref_flag: float = 1
    self_ref_block: float = 2


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


def _level(value, flag, block):
    return BLOCK if value >= block else FLAG if value >= flag else ALLOW


def _count(patterns, text):
    """Sum of all occurrences across all patterns, not just which patterns fired: a chain that
    repeats the same directive five times is a stronger nesting signal than one that uses it once."""
    return sum(len(p.findall(text)) for p in patterns)


def _max_bracket_depth(text):
    """Deepest nesting of (), [] or {}. Deeply nested brackets are a common way to stack
    "do X inside Y inside Z" recursive instructions without using the word "step" at all."""
    depth = best = 0
    stack = []
    for ch in text:
        if ch in _OPENERS:
            stack.append(ch)
            depth = len(stack)
            best = max(best, depth)
        elif ch in _CLOSERS:
            if stack and stack[-1] == _CLOSERS[ch]:
                stack.pop()
                depth = len(stack)
    return best


class Detector:
    def __init__(self, cfg=None):
        self.cfg = cfg or DetectorConfig()

    def _nesting_depth(self, text):
        c = self.cfg
        score = _count(_NESTING_RE, text) + max(0, _max_bracket_depth(text) - 1)
        return CheckResult("nesting_depth", float(score), _level(score, c.nesting_flag, c.nesting_block))

    def _runaway_phrases(self, text):
        c = self.cfg
        score = _count(_RUNAWAY_RE, text)
        return CheckResult("runaway_phrases", float(score), _level(score, c.runaway_flag, c.runaway_block))

    def _self_reference(self, text):
        c = self.cfg
        score = _count(_SELF_REF_RE, text)
        return CheckResult("self_reference", float(score), _level(score, c.self_ref_flag, c.self_ref_block))

    def inspect(self, text, tokens=None):
        checks = [self._nesting_depth(text), self._runaway_phrases(text), self._self_reference(text)]
        level = max(c.level for c in checks)
        return Decision(level=level, tokens=tokens or 0, chars=len(text), checks=checks)
