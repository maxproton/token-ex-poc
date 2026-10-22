"""Central settings for the V2 (recursive prompting) experiment."""
import os
from dataclasses import dataclass


@dataclass
class Config:
    # LM Studio local server (Developer tab, start server). Default port is 1234.
    base_url: str = os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234/v1")
    # Must match the identifier of the model loaded in LM Studio.
    model: str = os.getenv("LMSTUDIO_MODEL", "huihui-mistral-small-3.2-24b-instruct-2506-abliterated")
    # Token counter used by the input detector and the output-side checkpoint budget.
    # "approx" | "tiktoken:<encoding>" | "hf:<repo_or_local_path>"
    tokenizer: str = os.getenv("TEA_TOKENIZER", "approx")
    # Hard cap declared to the API for a single turn. This is the budget, not the prediction.
    max_completion_tokens: int = 512
    # Expected answer length for an ordinary, non-recursive turn. The output guard compares the
    # actual stream against this, not just against max_completion_tokens: a response that is well
    # inside the hard cap but far past this ceiling is still a sign that the model is stuck
    # processing its own output instead of answering.
    predicted_ceiling_tokens: int = 60
    # How often (in generated tokens) the stream is re-examined for runaway/degenerate output.
    checkpoint_tokens: int = 64
    # Abort the stream once completion tokens / predicted_ceiling_tokens reaches this multiple.
    ceiling_block_mult: float = 8.0
    # Output-side repetition (5-gram duplicate share) and compression ratio bounds, checked at
    # each checkpoint. Same thresholds and meaning as tea_v1's input-side equivalents.
    output_rep_block: float = 0.70
    output_comp_block: float = 0.06
    # Tokens/second above this is treated as an anomaly (e.g. near-identical cached emission)
    # rather than genuine generation, for the hardware this harness is expected to run on.
    tps_block: float = 200.0
    temperature: float = 0.0
    timeout_s: int = 1800
    seed: int = 1234
