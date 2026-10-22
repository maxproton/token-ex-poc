"""Input + output guardrail in front of the local LLM for V2 (recursive prompting).

Run from this folder:  python guardrail.py

check() gates the prompt itself (instruction nesting depth, runaway-generation language,
self-reference to the model's own prior output) before the model is ever called. ask() then
streams the reply through chat_guarded(), which can still cut generation short mid-stream if the
model starts looping or runs far past what the question needed -- the input side can only judge the
request, not what the model will actually do with it.

Type a prompt on one line, or @path/to/file.txt to load one from a file. Type quit to exit.
"""
from pathlib import Path

from tea_v2.client import LMStudioClient
from tea_v2.config import Config
from tea_v2.detector import ALLOW, Detector, DetectorConfig
from tea_v2.tasks import SYSTEM_PROMPT
from tea_v2.tokcount import make_counter

# ============================ EDIT THESE ============================
THRESHOLD = 0.70              # reject if score > THRESHOLD. 0.5 is the detector's "flag" level, 1.0 never rejects.
SHOW_BREAKDOWN = True          # print the per-check scores

# Per check: (safe, flag, block). Risk is 0.0 at "safe", 0.5 at "flag", 1.0 at "block", linear in between.
BANDS = {
    "nesting_depth":     (0, 2, 4),     # stacked recursive directives (chained steps, repeated wrappers)
    "runaway_phrases":   (0, 1, 3),     # instructions demanding unbounded/self-referential continuation
    "self_reference":    (0, 1, 2),     # references to the model's own prior output in the same prompt
}
# ====================================================================

SESSION = "cli"


def risk(value, safe, flag, block):
    if value <= safe:
        return 0.0
    if value >= block:
        return 1.0
    if value <= flag:
        return 0.5 * (value - safe) / (flag - safe)
    return 0.5 + 0.5 * (value - flag) / (block - flag)


def score_prompt(detector, text):
    d = detector.inspect(text)
    parts = {}
    for c in d.checks:
        band = BANDS.get(c.name)
        if band is None:
            continue
        parts[c.name] = (c.value, risk(c.value, *band))
    score = max((r for _, r in parts.values()), default=0.0)
    return d, score, parts


def read_prompt(raw):
    raw = raw.strip()
    if raw.startswith("@"):
        return Path(raw[1:].strip()).expanduser().read_text(encoding="utf-8")
    return raw


class Guardrail:
    """Scores prompts for recursive-prompting structure and gates access to the model. Shared by
    the CLI here and by other front-ends (e.g. the Flask UI in web_interface/), same pattern as
    tea_v1's Guardrail."""

    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        self.count, self.tokens_reliable = make_counter(self.cfg.tokenizer)
        self.detector = Detector(DetectorConfig())
        self.client = LMStudioClient(self.cfg)

    def check(self, text, session_id=SESSION):
        """Score a prompt for recursive-prompting structure. Returns (allowed, score, parts, decision)."""
        d, score, parts = score_prompt(self.detector, text)
        allowed = score <= THRESHOLD
        return allowed, score, parts, d

    def ask(self, text):
        """Send an already-accepted prompt to the model with the output-side streaming guard applied."""
        return self.client.chat_guarded(SYSTEM_PROMPT, text, self.count)


def main():
    guardrail = Guardrail()
    print(f"Guardrail ready. Threshold {THRESHOLD}. Model {guardrail.cfg.model}. "
          f"Predicted ceiling {guardrail.cfg.predicted_ceiling_tokens} tokens.")

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

        allowed, score, parts, d = guardrail.check(text)
        driver = max(parts, key=lambda k: parts[k][1]) if parts else "none"
        print(f"chars={d.chars}  score={score:.2f}  (driven by {driver})")
        if SHOW_BREAKDOWN:
            for name, (value, r) in parts.items():
                print(f"   {name:<18}{value:>10.3f}   risk {r:.2f}")

        if not allowed:
            print(f"REJECTED: score {score:.2f} is above threshold {THRESHOLD}. The model was not called.")
            continue

        res = guardrail.ask(text)
        if res["status"] == "error":
            print(f"Model error: {res.get('error')}")
            continue
        print(f"completion_tokens={res['completion_tokens']} ceiling_ratio={res['ceiling_ratio']:.2f} "
              f"reasoning_ratio={res['reasoning_ratio']:.2f} latency={res['latency_s']:.2f}s")
        if res["status"] == "truncated":
            print(f"OUTPUT TRUNCATED mid-stream: {res['abort_reason']}")
        print()
        print(res["text"])


if __name__ == "__main__":
    main()
