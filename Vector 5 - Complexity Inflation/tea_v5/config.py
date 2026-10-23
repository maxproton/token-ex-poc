"""Central settings for the V5 (complexity inflation) experiment."""
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # LM Studio local server (Developer tab, start server). Default port is 1234.
    base_url: str = os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234/v1")
    # Must match the identifier of the model loaded in LM Studio.
    model: str = os.getenv("LMSTUDIO_MODEL", "huihui-mistral-small-3.2-24b-instruct-2506-abliterated")
    tokenizer: str = os.getenv("TEA_TOKENIZER", "approx")
    # Hard cap declared to the API. This is the outer safety net; the ceiling the output guard
    # actually watches turn to turn is the dynamic, per-prompt one predicted by the classifier.
    max_completion_tokens: int = 4096
    # How often (in generated tokens) the stream is re-examined against the predicted ceiling.
    checkpoint_tokens: int = 64
    # Abort the stream once reasoning tokens / predicted_ceiling_tokens reaches this multiple.
    ceiling_block_mult: float = 4.0
    temperature: float = 0.0
    timeout_s: int = 1800
    seed: int = 1234


@dataclass
class ComplexityConfig:
    """Maps the classifier's predicted-effort label to a reasoning-token ceiling. Deliberately
    rough -- the point of the number is to catch a prompt whose actual reasoning blows past what
    its own predicted complexity would justify, not to size the ceiling precisely."""
    default_ceiling: int = 500  # used when the classifier is unavailable or returns "unknown"
    ceiling_by_label: dict = field(default_factory=lambda: {
        "trivial": 120,
        "low": 350,
        "medium": 900,
        "high": 2200,
    })
