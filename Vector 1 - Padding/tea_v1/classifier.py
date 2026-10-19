"""Shared helpers for the LLM-based semantic coherence classifier.

The stateless checks in detector.py all look at the *shape* of the text: duplication, zlib
compressibility, invisible/control characters. The "natural" padding variant (padding.py:_natural)
is built to defeat exactly those checks: a word-salad drawn from a small common-word vocabulary that
still compresses and tokenises like ordinary English. No shape-only check can separate that from
genuine long-form prose, because the difference is one of meaning, not structure.

This module asks a small local model a single yes/no question about whether a prompt means anything,
instead. It deliberately uses a tiny model with a hard output cap, in keeping with the paper's
Automated Model Selector principle (see README, "The defence architecture"): a large reasoning model
making this call would place a second, unbounded cost in front of the first and would itself be
exposed to complexity inflation.

Recommended model to download in LM Studio: Qwen2.5-1.5B-Instruct. It is small and fast enough that
the extra call stays cheap next to the guarded request, follows a one-word-answer instruction
reliably (unlike the 0.5B tier), and is multilingual, which matters because legit_samples.py includes
CJK content that an English-only tiny model could misjudge as nonsense.

Loaded on demand and unloaded again right after use (via the `lms` CLI) so it never sits in memory
alongside the main model.
"""
import re
import subprocess
from contextlib import contextmanager
from dataclasses import replace

from .client import LMStudioClient
from .config import Config

CLASSIFIER_MODEL = "qwen2.5-1.5b-instruct"
CLASSIFIER_MAX_TOKENS = 12        # one word, nothing more
CLASSIFIER_EXCERPT_CHARS = 1500   # head/tail sample; never forward the whole padded prompt
LMS_LOAD_TIMEOUT_S = 120          # cold model loads from disk can take a while
LMS_UNLOAD_TIMEOUT_S = 30

CLASSIFIER_SYSTEM_PROMPT = (
    "You are a strict text classifier. You will be shown a TEXT SAMPLE drawn from "
    "a user prompt sent to another AI assistant. Decide whether the sample is "
    "coherent, meaningful language (a question, an instruction, an explanation, "
    "an article, code, logs, or any other real content a human would recognise "
    "as meaningful) or whether it is filler: random word salad, repeated "
    "boilerplate, or padding with no semantic content, inserted only to inflate "
    "length.\n\n"
    "Judge meaning, not grammar or length. A short real sentence is MEANINGFUL. "
    "A long block of grammatically well-formed but random, repetitive, or "
    "purposeless text is NONSENSE.\n\n"
    "Respond with exactly one word, MEANINGFUL or NONSENSE, and nothing else. "
    "Do not explain your answer."
)


class ClassifierUnavailable(Exception):
    """Raised when the `lms` CLI can't load or unload the classifier model."""


def _excerpt(text, n=CLASSIFIER_EXCERPT_CHARS):
    """Bound how much text reaches the classifier: padding can be arbitrarily long,
    but the classifier call must stay cheap regardless of the input size."""
    if len(text) <= n:
        return text
    half = n // 2
    return f"{text[:half]}\n...\n{text[-half:]}"


def build_classifier_client(cfg=None, model=CLASSIFIER_MODEL):
    base = cfg or Config()
    classifier_cfg = replace(base, model=model, max_completion_tokens=CLASSIFIER_MAX_TOKENS, temperature=0.0)
    return LMStudioClient(classifier_cfg)


def classify_coherence(client, text):
    """Ask the classifier model whether `text` reads as meaningful content or nonsense/padding.

    Returns {"label": "meaningful" | "nonsense" | "unknown", "raw": str, "error": str | None}.
    """
    user_prompt = f'TEXT SAMPLE:\n"""\n{_excerpt(text)}\n"""\n\nAnswer:'
    res = client.chat(CLASSIFIER_SYSTEM_PROMPT, user_prompt)
    if res["status"] != "ok":
        return {"label": "unknown", "raw": "", "error": res.get("error")}
    raw = res["text"].strip()
    m = re.search(r"\b(meaningful|nonsense)\b", raw, re.IGNORECASE)
    return {"label": m.group(1).lower() if m else "unknown", "raw": raw, "error": None}


def _run_lms(args, timeout):
    try:
        res = subprocess.run(["lms", *args], capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        raise ClassifierUnavailable("the 'lms' CLI is not on PATH (install/open LM Studio first)")
    except subprocess.TimeoutExpired:
        raise ClassifierUnavailable(f"'lms {' '.join(args)}' timed out after {timeout}s")
    if res.returncode != 0:
        raise ClassifierUnavailable(f"'lms {' '.join(args)}' failed: {(res.stderr or res.stdout).strip()}")


@contextmanager
def loaded_classifier(model=CLASSIFIER_MODEL, cfg=None):
    """Load the classifier model just for the lifetime of this block, then unload it again,
    so it never has to sit in memory alongside the main model between requests."""
    _run_lms(["load", model, "--identifier", model, "--yes"], LMS_LOAD_TIMEOUT_S)
    try:
        yield build_classifier_client(cfg, model=model)
    finally:
        _run_lms(["unload", model], LMS_UNLOAD_TIMEOUT_S)
