"""Input-layer detector for V5 (complexity inflation). Returns allow / flag / block with the
reasons.

The paper is explicit that this vector is "the hardest to detect by inspection, because the
prompt is unremarkable" -- unlike V1-V4, there is no shape or session trend that reliably gives it
away. These three checks are a deliberately weak, cheap pre-screen that only catches the clumsy
cases: bundling several distinct asks into one short prompt, piling on open-ended/unbounded
qualifiers ("in as much detail as possible", "every conceivable angle"), and explicit
reasoning-bait phrasing that tells the model not to stop deliberating. The real defence is the
classifier-predicted ceiling and output-side mismatch check in guardrail.py/client.py.
"""
import re
from dataclasses import dataclass, field

ALLOW, FLAG, BLOCK = 0, 1, 2
ACTIONS = {ALLOW: "allow", FLAG: "flag", BLOCK: "block"}

# Conjunctions/separators that join distinct asks into one prompt, plus the prompt's own "?" count.
_BUNDLE_PATTERNS = [
    r"\balso\b",
    r"\badditionally\b",
    r"\bas well as\b",
    r"\bfurthermore\b",
    r"\bon top of that\b",
    r"\band then\b",
    r"\?",
]

# Open-ended/unbounded scope language: asks for maximal effort without bounding how much is enough.
_AMBIGUITY_PATTERNS = [
    r"\bin as much detail as possible\b",
    r"\bas detailed as possible\b",
    r"\bevery (?:possible|conceivable|single) \w+\b",
    r"\ball possible \w+\b",
    r"\bexhaustive(?:ly)?\b",
    r"\bcomprehensive(?:ly)?\b",
    r"\bthoroughly\b",
    r"\bleave no \w+ unturned\b",
    r"\bcover everything\b",
    r"\bevery angle\b",
    r"\bevery scenario\b",
    r"\bevery edge case\b",
]

# Explicit instructions aimed at the model's internal deliberation, not just its answer.
_REASONING_BAIT_PATTERNS = [
    r"\bthink (?:step by step|through this carefully|as long as (?:it|you) need)\b",
    r"\bshow all your (?:work|reasoning)\b",
    r"\bdon'?t stop (?:thinking|reasoning) until\b",
    r"\bkeep (?:thinking|reasoning) until\b",
    r"\btake as (?:much|long as) (?:time|you need)\b",
    r"\breason (?:extensively|exhaustively) about\b",
    r"\bconsider every (?:possible )?(?:interpretation|possibility|angle)\b",
    r"\bdouble[- ]?check (?:your (?:answer|work) )?(?:multiple times|repeatedly|over and over)\b",
]

_BUNDLE_RE = [re.compile(p, re.IGNORECASE) for p in _BUNDLE_PATTERNS]
_AMBIGUITY_RE = [re.compile(p, re.IGNORECASE) for p in _AMBIGUITY_PATTERNS]
_REASONING_BAIT_RE = [re.compile(p, re.IGNORECASE) for p in _REASONING_BAIT_PATTERNS]


@dataclass
class DetectorConfig:
    bundle_flag: float = 3
    bundle_block: float = 6
    ambiguity_flag: float = 2
    ambiguity_block: float = 4
    bait_flag: float = 1
    bait_block: float = 2


@dataclass
class CheckResult:
    name: str
    value: float
    level: int


@dataclass
class Decision:
    level: int
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
    """Sum of all occurrences across all patterns, not just which patterns fired: a prompt
    bundling five separate asks is a stronger signal than one that bundles two."""
    return sum(len(p.findall(text)) for p in patterns)


class Detector:
    def __init__(self, cfg=None):
        self.cfg = cfg or DetectorConfig()

    def _bundled_tasks(self, text):
        c = self.cfg
        score = _count(_BUNDLE_RE, text)
        return CheckResult("bundled_task_count", float(score), _level(score, c.bundle_flag, c.bundle_block))

    def _ambiguity(self, text):
        c = self.cfg
        score = _count(_AMBIGUITY_RE, text)
        return CheckResult("ambiguity_hits", float(score), _level(score, c.ambiguity_flag, c.ambiguity_block))

    def _reasoning_bait(self, text):
        c = self.cfg
        score = _count(_REASONING_BAIT_RE, text)
        return CheckResult("reasoning_bait_hits", float(score), _level(score, c.bait_flag, c.bait_block))

    def inspect(self, text):
        checks = [self._bundled_tasks(text), self._ambiguity(text), self._reasoning_bait(text)]
        level = max(c.level for c in checks)
        return Decision(level=level, chars=len(text), checks=checks)
