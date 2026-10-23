"""Session-layer guardrail in front of the local LLM for V4 (multi-turn accumulation / "slow
bleed"). Unlike V1-V3, check() judges a prompt against its own session's history, not the prompt
in isolation -- a session_id with no history yet is always allowed, since there's nothing to trend
against.

Run from this folder: python guardrail.py
Type a prompt on one line to extend the current CLI session, or @path/to/transcript.txt to replay
every line of a file as a separate turn in sequence. Type quit to exit, or reset to start a fresh
session_id without restarting the process.
"""
from pathlib import Path

from tea_v4.client import LMStudioClient
from tea_v4.config import AccumulationConfig, Config
from tea_v4.detector import ALLOW, Detector, DetectorConfig
from tea_v4.tokcount import make_counter

# ============================ EDIT THESE ============================
THRESHOLD = 0.70              # reject if score > THRESHOLD. 0.5 is the detector's "flag" level, 1.0 never rejects.
# Growth ratios are noise when every value involved is tiny: "7 tokens then 14" is a textbook 2x
# ratio from one slightly longer sentence, not a trend. Below this floor, prompt-side growth
# checks are not scored at all -- mirrors V1's MIN_TOKENS_FOR_BASELINE for the same reason.
MIN_TOKENS_FOR_TREND = 40
SHOW_BREAKDOWN = True

SYSTEM_PROMPT = "You are a helpful assistant."

# Per check: (safe, flag, block). Risk is 0.0 at "safe", 0.5 at "flag", 1.0 at "block", linear in between.
BANDS = {
    "prompt_growth_ratio":      (1.2, 2.0, 4.0),   # recent-half avg / early-half avg, this session's prompt tokens
    "sustained_growth_streak":  (0.4, 0.6, 0.9),   # share of the tracked window that's an unbroken non-decreasing run
    "completion_growth_ratio":  (1.2, 2.0, 4.0),   # same trend, this session's completion tokens
    "cumulative_session_ratio": (0.5, 1.0, 2.0),   # cumulative session tokens / session_token_budget
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


def score_turn(detector, session_id, prompt_tokens):
    d = detector.inspect(session_id, prompt_tokens)
    parts = {}
    for c in d.checks:
        band = BANDS.get(c.name)
        if band is None:
            continue
        r = risk(c.value, *band)
        if c.name in ("prompt_growth_ratio", "sustained_growth_streak") and prompt_tokens < MIN_TOKENS_FOR_TREND:
            r = 0.0
        parts[c.name] = (c.value, r)
    score = max((r for _, r in parts.values()), default=0.0)
    return d, score, parts


def read_prompt(raw):
    raw = raw.strip()
    if raw.startswith("@"):
        return Path(raw[1:].strip()).read_text(encoding="utf-8").splitlines()
    return [raw]


class Guardrail:
    """Scores a turn against its own session's accumulated history and gates access to the
    model. Shared by the CLI here and by other front-ends (e.g. the Flask UI in web_interface/),
    same pattern as tea_v1/tea_v2/tea_v3's Guardrail. Unlike those, this one also needs a
    record_completion() call after each accepted turn's response comes back, since the
    completion-side checks can't be computed until that's known."""

    def __init__(self, cfg=None, acc_cfg=None):
        self.cfg = cfg or Config()
        self.acc_cfg = acc_cfg or AccumulationConfig()
        self.count, self.tokens_reliable = make_counter(self.cfg.tokenizer)
        self.detector = Detector(DetectorConfig(
            window=self.acc_cfg.window,
            min_turns_for_trend=self.acc_cfg.min_turns_for_trend,
            session_token_budget=self.acc_cfg.session_token_budget,
        ))
        self.client = LMStudioClient(self.cfg)

    def check(self, text, session_id=SESSION):
        """Score this turn's prompt against the session's history. Returns
        (allowed, score, parts, decision). Accepted turns are folded into the session's history;
        rejected ones are not, so a blocked turn can't itself poison the baseline."""
        prompt_tokens = self.count(text)
        d, score, parts = score_turn(self.detector, session_id, prompt_tokens)
        allowed = score <= THRESHOLD
        if allowed:
            self.detector.record_prompt(session_id, prompt_tokens)
        return allowed, score, parts, d

    def record_completion(self, session_id, completion_tokens):
        """Call after an accepted turn's response comes back, so the next turn's completion-growth
        and cumulative-budget checks see it."""
        if completion_tokens is not None:
            self.detector.record_completion(session_id, completion_tokens)

    def ask(self, text):
        """Standalone, non-streaming call for the CLI."""
        return self.client.chat(SYSTEM_PROMPT, text)

    def reset_session(self, session_id):
        """Drop this session's accumulated prompt/completion/cumulative history, e.g. when a
        front-end starts a clean slate instead of actually rotating session_id."""
        self.detector.prompt_hist.pop(session_id, None)
        self.detector.completion_hist.pop(session_id, None)
        self.detector.cumulative.pop(session_id, None)


def main():
    guardrail = Guardrail()
    session_id = SESSION
    print(f"Guardrail ready. Threshold {THRESHOLD}. Session budget {guardrail.acc_cfg.session_token_budget} tokens. "
          f"Window {guardrail.acc_cfg.window} turns.")

    while True:
        try:
            raw = input("\nprompt> ")
        except (EOFError, KeyboardInterrupt):
            break
        if raw.strip().lower() in {"quit", "exit"}:
            break
        if raw.strip().lower() == "reset":
            session_id = f"{SESSION}-{id(raw)}"
            print("Started a fresh session.")
            continue
        if not raw.strip():
            continue
        try:
            turns = [t for t in read_prompt(raw) if t.strip()]
        except OSError as e:
            print(f"Could not read file: {e}")
            continue

        for text in turns:
            allowed, score, parts, d = guardrail.check(text, session_id)
            driver = max(parts, key=lambda k: parts[k][1]) if parts else "none"
            print(f"\nturn={d.turn}  tokens~{d.tokens}  score={score:.2f}  (driven by {driver})")
            print(f"  > {text[:100]!r}")
            if SHOW_BREAKDOWN:
                for name, (value, r) in parts.items():
                    print(f"   {name:<26}{value:>10.3f}   risk {r:.2f}")

            if not allowed:
                print(f"REJECTED: score {score:.2f} is above threshold {THRESHOLD}. The model was not called.")
                continue

            res = guardrail.ask(text)
            if res["status"] != "ok":
                print(f"Model error: {res.get('error')}")
                continue
            guardrail.record_completion(session_id, res.get("completion_tokens"))
            print(f"ACCEPTED. completion_tokens={res.get('completion_tokens')} latency={res['latency_s']:.2f}s")
            print(res["text"][:300])


if __name__ == "__main__":
    main()
