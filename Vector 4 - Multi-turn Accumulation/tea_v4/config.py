"""Central settings for the V4 (multi-turn accumulation / "slow bleed") experiment."""
import os
from dataclasses import dataclass


@dataclass
class Config:
    # LM Studio local server (Developer tab, start server). Default port is 1234.
    base_url: str = os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234/v1")
    # Must match the identifier of the model loaded in LM Studio.
    model: str = os.getenv("LMSTUDIO_MODEL", "huihui-mistral-small-3.2-24b-instruct-2506-abliterated")
    tokenizer: str = os.getenv("TEA_TOKENIZER", "approx")
    max_completion_tokens: int = 512
    temperature: float = 0.0
    timeout_s: int = 1800
    seed: int = 1234


@dataclass
class AccumulationConfig:
    """Session-level bookkeeping. No single prompt is judged against these -- only how a
    session's prompt/completion sizes move over its own recent turns, and how much it has
    consumed in total."""
    window: int = 10               # most recent turns kept for trend detection
    min_turns_for_trend: int = 4   # turns needed (including the current one) before a trend is trusted
    session_token_budget: int = 20000  # cumulative prompt+completion tokens before a session is "spent"
