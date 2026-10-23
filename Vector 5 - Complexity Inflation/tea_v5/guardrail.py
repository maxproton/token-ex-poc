"""Input + output guardrail in front of the local LLM for V5 (complexity inflation).

check() runs the cheap structural pre-screen (detector.py) and, if that passes, the complexity
classifier (classifier.py), returning a predicted reasoning-token ceiling for the prompt. ask()
then streams the reply through chat_guarded(), which watches the model's actual reasoning-token
count against that predicted ceiling and can cut generation short mid-stream the moment the
mismatch becomes extreme. This is the real defence: the paper is explicit that V5 prompts look
unremarkable going in, so only the gap between predicted and actual effort gives the attack away.

Run from this folder: python guardrail.py
Type a prompt on one line, or @path/to/file.txt to replay every (non-blank) line of a file as a
separate, independent prompt in turn -- unlike V4, there is no session state here, so each line is
scored and answered completely on its own. Type quit to exit.
"""
from pathlib import Path

from tea_v5.classifier import CLASSIFIER_MODEL, ClassifierUnavailable, classify_complexity, loaded_classifier
from tea_v5.client import LMStudioClient
from tea_v5.config import ComplexityConfig, Config
from tea_v5.detector import Detector, DetectorConfig
from tea_v5.tokcount import make_counter

# ============================ EDIT THESE ============================
THRESHOLD = 0.70              # reject if score > THRESHOLD. 0.5 is the detector's "flag" level, 1.0 never rejects.
SHOW_BREAKDOWN = True

# Costs an extra (tiny, capped) model round trip, so it only runs once the structural pre-screen
# has already passed, same principle as V1's semantic classifier.
CLASSIFIER_ENABLED = True

SYSTEM_PROMPT = "You are a helpful assistant."

# Per check: (safe, flag, block). Risk is 0.0 at "safe", 0.5 at "flag", 1.0 at "block", linear in between.
BANDS = {
    "bundled_task_count":  (1, 3, 6),    # distinct asks/conjunctions bundled into one prompt
    "ambiguity_hits":      (0, 2, 4),    # open-ended/unbounded scope qualifiers
    "reasoning_bait_hits": (0, 1, 2),    # explicit instructions targeting the model's own deliberation
}
# The real signal: how far the model's actual reasoning ran past what its own predicted
# complexity would justify. Only known once ask() returns -- see score_response().
RESPONSE_BANDS = {
    "reasoning_ceiling_ratio": (1.0, 2.0, 4.0),   # reasoning_tokens / predicted_ceiling_tokens
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


def read_prompts(raw):
    raw = raw.strip()
    if raw.startswith("@"):
        text = Path(raw[1:].strip()).expanduser().read_text(encoding="utf-8")
        return [line for line in text.splitlines() if line.strip()]
    return [raw]


class Guardrail:
    """Scores prompts for complexity-inflation structure, predicts a reasoning-token ceiling via
    the classifier, and gates access to the model. Shared by the CLI here and by other front-ends
    (e.g. the Flask UI in web_interface/), same pattern as tea_v1/tea_v2/tea_v4's Guardrail."""

    def __init__(self, cfg=None, complexity_cfg=None):
        self.cfg = cfg or Config()
        self.complexity_cfg = complexity_cfg or ComplexityConfig()
        self.count, self.tokens_reliable = make_counter(self.cfg.tokenizer)
        self.detector = Detector(DetectorConfig())
        self.client = LMStudioClient(self.cfg)

    def check(self, text):
        """Score a prompt for complexity-inflation structure and predict its reasoning-token
        ceiling. Returns (allowed, score, parts, decision, predicted_ceiling_tokens)."""
        d, score, parts = score_prompt(self.detector, text)
        allowed = score <= THRESHOLD

        predicted_ceiling_tokens = self.complexity_cfg.default_ceiling
        if allowed and CLASSIFIER_ENABLED:
            info = self._classify(text)
            predicted_ceiling_tokens = self.complexity_cfg.ceiling_by_label.get(
                info["label"], self.complexity_cfg.default_ceiling)
            parts["predicted_complexity"] = (info["label"] or "unknown", 0.0)

        return allowed, score, parts, d, predicted_ceiling_tokens

    def _classify(self, text):
        """Load the classifier model, ask it, unload it. Fails open: if it's unavailable, the
        default ceiling is used instead of blocking every prompt for an infrastructure problem."""
        try:
            with loaded_classifier(CLASSIFIER_MODEL) as client:
                return classify_complexity(client, text)
        except ClassifierUnavailable as e:
            return {"label": "unknown", "raw": "", "error": str(e)}

    def ask(self, text, predicted_ceiling_tokens, system=None):
        """Send an already-accepted prompt to the model with the reasoning-ceiling output guard
        applied. `system` lets another vector's guardrail reuse this with its own system prompt."""
        return self.client.chat_guarded(system or SYSTEM_PROMPT, text, self.count, predicted_ceiling_tokens)

    def score_response(self, result):
        """Score the real signal: how far actual reasoning ran past the predicted ceiling. Call
        after ask() returns. Returns (score, parts)."""
        ratio = result.get("reasoning_ceiling_ratio", 0.0)
        r = risk(ratio, *RESPONSE_BANDS["reasoning_ceiling_ratio"])
        return r, {"reasoning_ceiling_ratio": (ratio, r)}


def main():
    guardrail = Guardrail()
    print(f"Guardrail ready. Threshold {THRESHOLD}. Model {guardrail.cfg.model}.")

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
            prompts = read_prompts(raw)
        except OSError as e:
            print(f"Could not read file: {e}")
            continue

        for text in prompts:
            allowed, score, parts, d, predicted_ceiling_tokens = guardrail.check(text)
            driver = max(parts, key=lambda k: 0.0 if isinstance(parts[k][0], str) else parts[k][1]) if parts else "none"
            print(f"\n> {text[:100]!r}")
            print(f"chars={d.chars}  score={score:.2f}  (driven by {driver})  "
                  f"predicted_ceiling={predicted_ceiling_tokens}")
            if SHOW_BREAKDOWN:
                for name, (value, r) in parts.items():
                    if isinstance(value, str):
                        print(f"   {name:<20}{value:>10}")
                    else:
                        print(f"   {name:<20}{value:>10.3f}   risk {r:.2f}")

            if not allowed:
                print(f"REJECTED: score {score:.2f} is above threshold {THRESHOLD}. The model was not called.")
                continue

            res = guardrail.ask(text, predicted_ceiling_tokens)
            if res["status"] == "error":
                print(f"Model error: {res.get('error')}")
                continue
            resp_score, resp_parts = guardrail.score_response(res)
            print(f"reasoning_tokens={res['reasoning_tokens']} answer_tokens={res['answer_tokens']} "
                  f"reasoning_ceiling_ratio={res['reasoning_ceiling_ratio']:.2f} mismatch_score={resp_score:.2f} "
                  f"latency={res['latency_s']:.2f}s")
            if res["status"] == "truncated":
                print(f"OUTPUT TRUNCATED mid-stream: {res['abort_reason']}")
            print()
            print(res["text"])


if __name__ == "__main__":
    main()
