"""Input guardrail in front of the local LLM.

Run from this folder:  python guardrail.py

For each prompt it computes a risk score from 0.0 (looks normal) to 1.0 (clearly padding or abuse).
Score above THRESHOLD: rejected, the model is never called. Otherwise the prompt goes to LM Studio.

Type a prompt on one line, or @path/to/file.txt to load a long prompt from a file
(for example one from results/<run>/prompts.jsonl saved as text). Type quit to exit.
"""
from pathlib import Path

from tea_v1.classifier import CLASSIFIER_MODEL, ClassifierUnavailable, classify_coherence, loaded_classifier
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

# Semantic check, catches "natural" word-salad padding that passes every stats check above.
# Costs an extra model round trip, so it only runs on prompts already large enough to matter,
# and only once the stats-based score has already passed.
CLASSIFIER_ENABLED = True
CLASSIFIER_MIN_CHARS = 500    # matches detector.py's min_chars_for_stats: below this, stats are noisy anyway

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


def score_prompt(detector, text, session_id=SESSION):
    d = detector.inspect(text, session_id)
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


class Guardrail:
    """Scores prompts and gates access to the model. Shared by the CLI here and by other
    front-ends (e.g. the Flask UI in web_interface/) so they apply identical scoring and
    baseline-tracking rules instead of reimplementing them."""

    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        self.cfg.max_completion_tokens = MAX_RESPONSE_TOKENS
        count, self.tokens_reliable = make_counter(self.cfg.tokenizer)
        self.detector = Detector(count, self.tokens_reliable, DetectorConfig(enable_baseline=True))
        self.client = LMStudioClient(self.cfg)

    def check(self, text, session_id=SESSION):
        """Score a prompt, update the per-session baseline, and (for long prompts that already
        passed the stats check) run the semantic coherence check. Returns
        (allowed, score, parts, decision, classifier_info). classifier_info is None if the
        semantic check did not run."""
        d, score, parts = score_prompt(self.detector, text, session_id)
        allowed = score <= THRESHOLD

        classifier_info = None
        if allowed and CLASSIFIER_ENABLED and len(text) >= CLASSIFIER_MIN_CHARS:
            classifier_info = self._classify(text)
            if classifier_info["label"] == "nonsense":
                allowed = False

        if not allowed:
            if d.level == ALLOW:                   # detector already counted it; keep the baseline clean
                self.detector.history[session_id].pop()
        elif d.level != ALLOW:                      # detector skipped it, but the guardrail accepted it
            self.detector.history[session_id].append(d.tokens)
        return allowed, score, parts, d, classifier_info

    def _classify(self, text):
        """Load the classifier model, ask it, unload it. Fails open: if the model can't be
        loaded or the call errors, the semantic check is skipped (stats-based decision stands)
        rather than blocking every long prompt because of an infrastructure problem."""
        try:
            with loaded_classifier(CLASSIFIER_MODEL) as client:
                return classify_coherence(client, text)
        except ClassifierUnavailable as e:
            return {"label": "unknown", "raw": "", "error": str(e)}

    def ask(self, text):
        """Send an already-accepted prompt to the model."""
        return self.client.chat(SYSTEM_PROMPT, text)

    def reset_session(self, session_id):
        """Drop this session's accepted-prompt baseline, e.g. when a front-end starts a clean
        slate instead of actually rotating session_id."""
        self.detector.history.pop(session_id, None)


def main():
    guardrail = Guardrail()
    print(f"Guardrail ready. Threshold {THRESHOLD}. Model {guardrail.cfg.model}. Tokenizer {guardrail.cfg.tokenizer}"
          + ("" if guardrail.tokens_reliable else " (approx: tokens_per_char check is off)"))

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

        allowed, score, parts, d, classifier_info = guardrail.check(text)
        driver = max(parts, key=lambda k: parts[k][1]) if parts else "none"
        print(f"tokens~{d.tokens}  chars={d.chars}  score={score:.2f}  (driven by {driver})")
        if SHOW_BREAKDOWN:
            for name, (value, r) in parts.items():
                print(f"   {name:<18}{value:>10.3f}   risk {r:.2f}")
            if classifier_info:
                if classifier_info["error"]:
                    print(f"   semantic check     skipped (classifier unavailable: {classifier_info['error']})")
                else:
                    print(f"   semantic check     {classifier_info['label']}")

        if not allowed:
            if classifier_info and classifier_info["label"] == "nonsense":
                print("REJECTED: semantic check found this prompt to be nonsense/padding. The model was not called.")
            else:
                print(f"REJECTED: score {score:.2f} is above threshold {THRESHOLD}. The model was not called.")
            continue

        res = guardrail.ask(text)
        if res["status"] != "ok":
            print(f"Model error: {res.get('error')}")
            continue
        print(f"ACCEPTED. Server usage: prompt={res['prompt_tokens']} completion={res['completion_tokens']} "
              f"latency={res['latency_s']:.2f}s\n")
        print(res["text"])


if __name__ == "__main__":
    main()