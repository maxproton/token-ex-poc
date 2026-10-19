"""Central settings for the V1 (volume and padding) experiment."""
import os
from dataclasses import dataclass


@dataclass
class Config:
    # LM Studio local server (Developer tab, start server). Default port is 1234.
    base_url: str = os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234/v1")
    # Must match the identifier of the model loaded in LM Studio.
    model: str = os.getenv("LMSTUDIO_MODEL", "huihui-mistral-small-3.2-24b-instruct-2506-abliterated")
    # Token counter used by the victim cap and the detector.
    # "approx" | "tiktoken:<encoding>" | "hf:<repo_or_local_path>"
    tokenizer: str = os.getenv("TEA_TOKENIZER", "approx")
    # Fixed completion budget so that output cost is constant and only input varies.
    max_completion_tokens: int = 64
    temperature: float = 0.0
    timeout_s: int = 1800
    # Simulated per-request input cap of the victim (tokens, as counted by the tokenizer above).
    victim_cap_tokens: int = 2000
    seed: int = 1234
    results_root: str = "results"
