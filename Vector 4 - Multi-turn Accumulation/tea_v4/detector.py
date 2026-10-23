"""Session-layer detector for V4 (multi-turn accumulation / "slow bleed"). Unlike V1-V3, no single
prompt is inspected in isolation -- every check here looks at how a session's own prompt and
completion sizes have moved over its recent turns, and how much token budget the session has
burned through in total. This is deliberately the only vector in this project whose state persists
and matters across many separate check() calls for the same session_id.

Two checks rely on per-turn growth within a bounded window (recent half average vs. earlier half
average -- cheap, explainable, robust to a single outlier turn). One tracks an unbroken run of
non-decreasing turns, which is what "slow bleed" actually looks like: nothing dramatic on any one
turn, just steady, repeated growth. The last tracks cumulative spend against a session budget,
which is what eventually turns a tolerated trend into a real cost problem.

A session needs `min_turns_for_trend` turns of history before any growth check fires: a new session
has nothing to compare against, and judging turn 2 against turn 1 is noise, not a trend.
"""
import statistics
from collections import deque
from dataclasses import dataclass, field

ALLOW, FLAG, BLOCK = 0, 1, 2
ACTIONS = {ALLOW: "allow", FLAG: "flag", BLOCK: "block"}


@dataclass
class DetectorConfig:
    window: int = 10
    min_turns_for_trend: int = 4
    session_token_budget: int = 20000
    growth_flag: float = 2.0          # recent-half avg / early-half avg of a token-count window
    growth_block: float = 4.0
    streak_flag: float = 0.6          # share of the window forming an unbroken non-decreasing run
    streak_block: float = 0.9
    budget_flag: float = 1.0          # cumulative session tokens / session_token_budget
    budget_block: float = 2.0


@dataclass
class CheckResult:
    name: str
    value: float
    level: int


@dataclass
class Decision:
    level: int
    turn: int
    tokens: int
    checks: list = field(default_factory=list)

    @property
    def action(self):
        return ACTIONS[self.level]

    def fired(self):
        return [f"{c.name}={c.value:.3f}" for c in self.checks if c.level > ALLOW]


def _level(value, flag, block):
    return BLOCK if value >= block else FLAG if value >= flag else ALLOW


def _growth_ratio(values):
    """Mean of the back half of `values` over the mean of the front half. >1 means the recent
    turns are bigger than the early ones in this window, not just a single spike."""
    mid = len(values) // 2
    front, back = values[:mid], values[mid:]
    front_mean = statistics.mean(front)
    back_mean = statistics.mean(back)
    if front_mean <= 0:
        return 2.0 if back_mean > 0 else 0.0
    return back_mean / front_mean


def _streak_ratio(values):
    """Longest run of consecutive non-decreasing turns ending at the most recent one, as a share
    of the window. Catches steady, modest growth that a magnitude ratio alone might not yet flag."""
    streak = 1
    for i in range(len(values) - 1, 0, -1):
        if values[i] >= values[i - 1]:
            streak += 1
        else:
            break
    return streak / len(values)


class Detector:
    def __init__(self, cfg=None):
        self.cfg = cfg or DetectorConfig()
        self.prompt_hist = {}
        self.completion_hist = {}
        self.cumulative = {}

    def _session_prompt_hist(self, session_id):
        return self.prompt_hist.setdefault(session_id, deque(maxlen=self.cfg.window))

    def _session_completion_hist(self, session_id):
        return self.completion_hist.setdefault(session_id, deque(maxlen=self.cfg.window))

    def _prompt_growth(self, session_id, prompt_tokens):
        c = self.cfg
        window = list(self._session_prompt_hist(session_id)) + [prompt_tokens]
        if len(window) < c.min_turns_for_trend:
            return CheckResult("prompt_growth_ratio", 0.0, ALLOW)
        ratio = _growth_ratio(window)
        return CheckResult("prompt_growth_ratio", ratio, _level(ratio, c.growth_flag, c.growth_block))

    def _prompt_streak(self, session_id, prompt_tokens):
        c = self.cfg
        window = list(self._session_prompt_hist(session_id)) + [prompt_tokens]
        if len(window) < c.min_turns_for_trend:
            return CheckResult("sustained_growth_streak", 0.0, ALLOW)
        ratio = _streak_ratio(window)
        return CheckResult("sustained_growth_streak", ratio, _level(ratio, c.streak_flag, c.streak_block))

    def _completion_growth(self, session_id):
        c = self.cfg
        window = list(self._session_completion_hist(session_id))
        if len(window) < c.min_turns_for_trend:
            return CheckResult("completion_growth_ratio", 0.0, ALLOW)
        ratio = _growth_ratio(window)
        return CheckResult("completion_growth_ratio", ratio, _level(ratio, c.growth_flag, c.growth_block))

    def _cumulative_budget(self, session_id, prompt_tokens):
        c = self.cfg
        spent = self.cumulative.get(session_id, 0) + prompt_tokens
        ratio = spent / max(c.session_token_budget, 1)
        return CheckResult("cumulative_session_ratio", ratio, _level(ratio, c.budget_flag, c.budget_block))

    def inspect(self, session_id, prompt_tokens):
        """Score the current turn's prompt against this session's own history. Does not mutate
        state -- call record_prompt()/record_completion() afterwards for turns that are accepted."""
        checks = [
            self._prompt_growth(session_id, prompt_tokens),
            self._prompt_streak(session_id, prompt_tokens),
            self._completion_growth(session_id),
            self._cumulative_budget(session_id, prompt_tokens),
        ]
        level = max(c.level for c in checks)
        turn = len(self._session_prompt_hist(session_id)) + 1
        return Decision(level=level, turn=turn, tokens=prompt_tokens, checks=checks)

    def record_prompt(self, session_id, prompt_tokens):
        self._session_prompt_hist(session_id).append(prompt_tokens)
        self.cumulative[session_id] = self.cumulative.get(session_id, 0) + prompt_tokens

    def record_completion(self, session_id, completion_tokens):
        self._session_completion_hist(session_id).append(completion_tokens)
        self.cumulative[session_id] = self.cumulative.get(session_id, 0) + completion_tokens
