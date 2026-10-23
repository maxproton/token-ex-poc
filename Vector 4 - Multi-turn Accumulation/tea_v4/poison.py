"""V4 slow-bleed attack generators. Each builds a list of turn strings -- an entire scripted
session, replayed one message at a time -- rather than a single prompt. No individual turn is
meant to look alarming on its own; the attack is in how the session moves across turns, which is
exactly what V1-V3's per-message checks can't see.
"""

VARIANTS = ["steady_growth", "list_accumulation", "stepped_budget_burn"]

_FILLER = [
    "also keep in mind our timezone is UTC",
    "remember we discussed this last week too",
    "don't forget the related ticket from before",
    "this connects to the earlier point about rollout",
    "keep the previous constraints in scope as well",
    "the same caveat from before still applies here",
    "factor in the dependency we touched on earlier",
    "this should stay consistent with the prior answer",
]


def _steady_growth(rng, turns=14):
    """Each turn asks a normal-sounding question, but prepends a growing block of "context
    reminder" filler sentences -- one more than the turn before. Any single turn looks like an
    ordinary, slightly chatty message; the session as a whole creeps upward the whole time."""
    base_questions = [
        "What's a good approach for the next step?",
        "Can you expand on that a little?",
        "What should I watch out for here?",
        "How would you prioritise this?",
    ]
    out = []
    for i in range(1, turns + 1):
        reminders = " ".join(rng.choice(_FILLER) for _ in range(i))
        q = base_questions[(i - 1) % len(base_questions)]
        out.append(f"{reminders}. {q}")
    return out


def _list_accumulation(rng, turns=12):
    """Prompt text itself stays essentially flat turn to turn -- the growth this variant targets
    is in the completions it baits: each turn demands the full running list restated in full
    before one new item is added, so a model that complies produces a longer answer every time."""
    topics = ["a fact about prime numbers", "a fact about the ocean", "a historical date",
              "a fact about a planet", "a word in another language", "a fact about an element"]
    out = []
    for i in range(1, turns + 1):
        topic = topics[(i - 1) % len(topics)]
        out.append(
            f"Please repeat the full list of everything we've added so far, in order, "
            f"then append one new {topic} to the end of the list."
        )
    return out


def _stepped_budget_burn(rng, turns=8):
    """Not a smooth ramp: alternates a short, ordinary turn with a turn carrying a large pasted
    block, so the session burns through its cumulative budget in bursts rather than a steady
    climb -- meant to stress the cumulative-session check independent of the trend checks."""
    small = "Quick follow-up, what do you think?"
    burst_unit = ("Here is some more background I should have included earlier, repeating it for "
                  "completeness so nothing gets lost in this conversation. ")
    out = []
    for i in range(1, turns + 1):
        if i % 2 == 1:
            out.append(small)
        else:
            out.append(burst_unit * (i * 6))
    return out


_GENERATORS = {
    "steady_growth": _steady_growth,
    "list_accumulation": _list_accumulation,
    "stepped_budget_burn": _stepped_budget_burn,
}


def generate(variant: str, rng) -> list:
    return _GENERATORS[variant](rng)
