"""Central settings for the V3 (context window stuffing) experiment."""
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


@dataclass
class RetrieverConfig:
    corpus_dir: str = "tea_v3/corpus"
    top_k: int = 6
    # Expected chunk size for this pipeline (it chunks by paragraph, with no hard cap -- the same
    # gap real naive splitters have). Used as the baseline the oversized-chunk check compares against.
    normal_chunk_chars: int = 500
