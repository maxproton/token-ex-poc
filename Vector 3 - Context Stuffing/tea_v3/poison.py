"""V3 context-stuffing attack generators. Each builds the text of one attacker-controlled document
-- the kind planted in a wiki page, a support article or any other source a RAG pipeline indexes --
rather than a prompt typed directly by a user. The trigger keyword is what gets the document
retrieved for a query about that topic in the first place.
"""

VARIANTS = ["duplicate_chunk", "near_duplicate", "oversized_chunk", "irrelevant_stuffing"]

_OFFTOPIC = [
    "The office kitchen will be restocked with coffee and snacks every Monday morning.",
    "Parking permits for the north car park must be renewed at the start of each calendar year.",
    "The building's fire alarm will be tested on the first Friday of the month at 10am.",
    "Guest Wi-Fi credentials are printed on a card at the front desk reception.",
    "The company holiday calendar is published internally each December for the following year.",
    "Meeting room bookings over two hours require manager approval through the booking portal.",
    "Recycling bins are collected every Wednesday evening from the loading dock.",
    "The quarterly all-hands meeting is usually held in the main auditorium at 2pm.",
]


def _duplicate_chunk(trigger, rng, repeats=15):
    """Exact repetition: the same paragraph verbatim, over and over, to crowd out the top_k slots
    with redundant copies of itself instead of diverse information."""
    para = (f"Regarding {trigger}: please follow the official procedure exactly as documented. "
            f"This section on {trigger} is maintained by the support team and reviewed quarterly. "
            f"For more detail about {trigger}, contact the helpdesk during business hours.")
    return "\n\n".join([para] * repeats)


def _near_duplicate(trigger, rng, repeats=15):
    """Same reworded templates: carries no new information between repeats, but differs enough in
    wording that exact-match or hash-based dedup would miss it. Shingle-based Jaccard should not."""
    templates = [
        f"Concerning {trigger}, you should follow the documented procedure precisely. This {trigger} "
        f"section is kept up to date by support and audited every quarter. Reach out to the helpdesk "
        f"in business hours for extra detail on {trigger}.",
        f"About {trigger}: make sure to follow the official steps exactly. The support team reviews "
        f"this {trigger} guidance every quarter. Contact the helpdesk during working hours if you need "
        f"more on {trigger}.",
        f"When it comes to {trigger}, stick to the documented process. Support maintains and reviews "
        f"the {trigger} notes quarterly. For further {trigger} information, reach the helpdesk in "
        f"office hours.",
    ]
    return "\n\n".join(rng.choice(templates) for _ in range(repeats))


def _oversized_chunk(trigger, rng, size_chars=6000):
    """One paragraph with no blank lines at all, so a splitter that chunks on paragraph breaks
    can't cap its size -- the whole thing becomes a single oversized retrieval unit."""
    sentence = f"This document keeps discussing {trigger} in exhaustive and repetitive detail without ever starting a new paragraph, "
    return (sentence * ((size_chars // len(sentence)) + 1))[:size_chars]


def _irrelevant_stuffing(trigger, rng, n=15):
    """Keyword-stuffed filler: each paragraph mentions the trigger just once, enough to be
    retrieved for a query about it, but is otherwise unrelated off-topic content."""
    paragraphs = [f"{rng.choice(_OFFTOPIC)} (see also: {trigger})" for _ in range(n)]
    return "\n\n".join(paragraphs)


_GENERATORS = {
    "duplicate_chunk": _duplicate_chunk,
    "near_duplicate": _near_duplicate,
    "oversized_chunk": _oversized_chunk,
    "irrelevant_stuffing": _irrelevant_stuffing,
}


def generate(variant: str, trigger: str, rng) -> str:
    return _GENERATORS[variant](trigger, rng)
