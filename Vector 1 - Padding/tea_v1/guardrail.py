"""Input guardrail in front of the local LLM.

Run from this folder:  python guardrail.py

For each prompt it computes a risk score from 0.0 (looks normal) to 1.0 (clearly padding or abuse).
Score above THRESHOLD: rejected, the model is never called. Otherwise the prompt goes to LM Studio.

Type a prompt on one line, or @path/to/file.txt to load a long prompt from a file
(for example one from results/<run>/prompts.jsonl saved as text). Type quit to exit.
"""
from pathlib import Path

from tea_v1.client import LMStudioClient
from tea_v1.config import Config
from tea_v1.detector import ALLOW, Detector, DetectorConfig
from tea_v1.tasks import SYSTEM_PROMPT
from tea_v1.tokcount import make_counter

# ============================ EDIT THESE ============================
THRESHOLD = 0.70              # reject if score > THRESHOLD. 0.5 is the detector's "flag" level, 1.0 never rejects.
MAX_RESPONSE_TOKENS = 512     # cap on the model's answer for accepted prompts
MIN_TOKENS_FOR_BASELINE = 1000  # the session-size check is ignored for prompts smaller than this
SHOW_BREAKDOWN = True         # print the per-check scores

# Per check: (safe, flag, block). Risk is 0.0 at "safe", 0.5 at "flag", 1.0 at "block", linear in between.
# Values that sit beyond "safe" (further from "flag") score 0.0. Works whether higher or lower is worse.
BANDS = {
    "repetition":        (0.10, 0.35, 0.70),     # share of duplicate 5-word sequences
    "compression":       (0.35, 0.18, 0.06),     # zlib size / raw size, lower is worse
    "invisible_chars":   (0.00, 0.01, 0.10),     # share of zero-width and control characters
    "whitespace_share":  (0.40, 0.60, 0.85),
    "tokens_per_char":   (0.35, 0.60, 1.00),     # needs a real tokenizer, see TEA_TOKENIZER
    "baseline_ratio":    (2.0, 4.0, 10.0),       # prompt size / median of this session's accepted prompts
    "cold_start_tokens": (1000, 3000, 12000),    # absolute size while the session has under 3 accepted prompts
}
# ====================================================================

SESSION = "cli"


def risk(value, safe, flag, block):
    if safe > block:                      # lower is worse: mirror the numbers and reuse the same logic
        return risk(-value, -safe, -flag, -block)
    if value <= safe:
        return 0.0
    if value >= block:
        return 1.0
    if value <= flag:
        return 0.5 * (value - safe) / (flag - safe)
    return 0.5 + 0.5 * (value - flag) / (block - flag)


def score_prompt(detector, text):
    d = detector.inspect(text, SESSION)
    parts = {}
    for c in d.checks:
        band = BANDS.get(c.name)
        if band is None:
            continue
        r = risk(c.value, *band)
        if c.name == "baseline_ratio" and d.tokens < MIN_TOKENS_FOR_BASELINE:
            r = 0.0
        parts[c.name] = (c.value, r)
    score = max((r for _, r in parts.values()), default=0.0)
    return d, score, parts


def read_prompt(raw):
    raw = raw.strip()
    if raw.startswith("@"):
        return Path(raw[1:].strip()).expanduser().read_text(encoding="utf-8")
    return raw


def main():
    cfg = Config()
    cfg.max_completion_tokens = MAX_RESPONSE_TOKENS
    count, reliable = make_counter(cfg.tokenizer)
    detector = Detector(count, reliable, DetectorConfig(enable_baseline=True))
    client = LMStudioClient(cfg)
    print(f"Guardrail ready. Threshold {THRESHOLD}. Model {cfg.model}. Tokenizer {cfg.tokenizer}"
          + ("" if reliable else " (approx: tokens_per_char check is off)"))

    while True:
        try:
            raw = input("\nprompt> ")
        except (EOFError, KeyboardInterrupt):
            break
        if raw.strip().lower() in {"quit", "exit"}:
            break
        if not raw.strip():
            continue
        try:
            text = read_prompt(raw)
        except OSError as e:
            print(f"Could not read file: {e}")
            continue

        d, score, parts = score_prompt(detector, text)
        driver = max(parts, key=lambda k: parts[k][1]) if parts else "none"
        print(f"tokens~{d.tokens}  chars={d.chars}  score={score:.2f}  (driven by {driver})")
        if SHOW_BREAKDOWN:
            for name, (value, r) in parts.items():
                print(f"   {name:<18}{value:>10.3f}   risk {r:.2f}")

        if score > THRESHOLD:
            if d.level == ALLOW:                   # detector already counted it; keep the baseline clean
                detector.history[SESSION].pop()
            print(f"REJECTED: score {score:.2f} is above threshold {THRESHOLD}. The model was not called.")
            continue

        if d.level != ALLOW:                       # detector skipped it, but the guardrail accepted it
            detector.history[SESSION].append(d.tokens)
        res = client.chat(SYSTEM_PROMPT, text)
        if res["status"] != "ok":
            print(f"Model error: {res.get('error')}")
            continue
        print(f"ACCEPTED. Server usage: prompt={res['prompt_tokens']} completion={res['completion_tokens']} "
              f"latency={res['latency_s']:.2f}s\n")
        print(res["text"])


if __name__ == "__main__":
    main()