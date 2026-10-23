"""Pluggable token counting. The server-reported usage is always the ground truth for cost.
This counter is only used for context-size accounting in the detector and the CLI."""


def make_counter(spec: str):
    if spec == "approx":
        # Crude: about 4 characters per token for English. Wrong for CJK, code and unicode tricks.
        return (lambda t: max(1, round(len(t) / 4))), False
    if spec.startswith("tiktoken:"):
        import tiktoken
        enc = tiktoken.get_encoding(spec.split(":", 1)[1])
        return (lambda t: len(enc.encode(t, disallowed_special=()))), True
    if spec.startswith("hf:"):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec.split(":", 1)[1])
        return (lambda t: len(tok.encode(t, add_special_tokens=False))), True
    raise ValueError(f"Unknown tokenizer spec: {spec}")
