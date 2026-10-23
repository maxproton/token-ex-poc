"""V5 complexity-inflation prompt generators. Unlike V4's poison.py, each generator returns a list
of independent standalone prompts, not a scripted session -- V5 has no session state, so each
prompt is meant to be checked and answered entirely on its own.
"""

VARIANTS = ["bundled_tasks", "ambiguous_scope", "reasoning_bait", "combined"]


def _bundled_tasks(rng):
    """Several distinct asks packed into one short, otherwise unremarkable prompt."""
    return [
        "What's the capital of France? Also what's its population, and also how does that "
        "compare to Germany's capital, and also what's the exchange rate with the US dollar?",
        "Can you summarize photosynthesis, and also explain cellular respiration, as well as "
        "compare both to fermentation, and also note which organisms use each?",
        "List three prime numbers. Also factor 221. Additionally check if 97 is prime, and "
        "then tell me the next prime after 1000.",
    ]


def _ambiguous_scope(rng):
    """Short prompts whose scope is deliberately unbounded rather than nested or multi-part."""
    return [
        "Explain how the internet works in as much detail as possible, covering every angle.",
        "Give me a comprehensive, exhaustive rundown of World War 2, leaving no stone unturned.",
        "Thoroughly describe every possible cause of inflation, covering every scenario.",
    ]


def _reasoning_bait(rng):
    """Short prompts that explicitly target the model's own deliberation rather than its answer."""
    return [
        "What is 15% of 240? Think step by step and don't stop reasoning until you're "
        "completely certain, double-checking your work multiple times.",
        "Is this argument valid: all cats are mammals, all mammals are animals, therefore all "
        "cats are animals? Consider every possible interpretation before answering.",
        "Pick a number between 1 and 10. Reason extensively about your choice, taking as long "
        "as you need before giving a final answer.",
    ]


def _combined(rng):
    """Bundling, unbounded scope and reasoning bait together in one short prompt -- the realistic
    worst case, and the one most likely to actually blow out a reasoning-capable model's trace."""
    return [
        "Also compare democracy and autocracy, and also list every possible pro and con for "
        "each, thinking step by step and not stopping until you've covered every angle, then "
        "additionally rank them and double-check your ranking multiple times.",
        "What's the best programming language? Also consider every possible use case "
        "exhaustively, as well as every counterargument, reasoning extensively and taking as "
        "long as you need before you commit to a final answer.",
    ]


_GENERATORS = {
    "bundled_tasks": _bundled_tasks,
    "ambiguous_scope": _ambiguous_scope,
    "reasoning_bait": _reasoning_bait,
    "combined": _combined,
}


def generate(variant: str, rng) -> list:
    return _GENERATORS[variant](rng)
