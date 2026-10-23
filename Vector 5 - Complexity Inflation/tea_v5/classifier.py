"""Shared helpers for the LLM-based complexity classifier.

Complexity inflation (V5) is the hardest TEA vector to catch by inspection: the paper notes the
prompt itself is "unremarkable". The structural heuristics in detector.py (bundled tasks, vague
open-ended qualifiers, explicit reasoning-bait phrasing) only catch the clumsy cases. The real
defence is a mismatch check: ask a small, cheap model how much reasoning effort this prompt looks
like it should need, map that judgment to a predicted reasoning-token ceiling, then watch the
actual model's reasoning tokens against that ceiling while it streams (client.py). A short prompt
classified "trivial" that goes on to burn thousands of reasoning tokens is complexity inflation,
regardless of how innocuous its text looked going in.

Deliberately a tiny model with a hard output cap, per the paper's Automated Model Selector
principle (Section 5.3): a large reasoning model making this call would place a second, unbounded
cost in front of the first and would itself be exposed to the exact attack it's meant to catch.
"""
import re
import subprocess
from contextlib import contextmanager
from dataclasses import replace

from tea_v5.client import LMStudioClient
from tea_v5.config import Config

CLASSIFIER_MODEL = "qwen2.5-1.5b-instruct"
CLASSIFIER_MAX_TOKENS = 12        # one word, nothing more
CLASSIFIER_EXCERPT_CHARS = 1500   # head/tail sample; never forward an arbitrarily long prompt
LMS_LOAD_TIMEOUT_S = 120          # cold model loads from disk can take a while
LMS_UNLOAD_TIMEOUT_S = 30

CLASSIFIER_SYSTEM_PROMPT = (
    "You are a strict complexity classifier. You will be shown a PROMPT that a user sent to "
    "another AI assistant. Estimate how much genuine reasoning effort a competent assistant needs "
    "to answer it well -- how many distinct sub-tasks it bundles together, how open-ended or "
    "unbounded its scope is, and how much real step-by-step working it requires, as opposed to a "
    "short, direct answer.\n\n"
    "Respond with exactly one word, TRIVIAL, LOW, MEDIUM or HIGH, and nothing else.\n"
    "TRIVIAL: a fact lookup, a single simple calculation, or a short direct question.\n"
    "LOW: a short explanation or one well-scoped task.\n"
    "MEDIUM: multi-step reasoning, a moderately open-ended request, or a couple of bundled tasks.\n"
    "HIGH: deep multi-step reasoning, many bundled tasks, or an explicitly open-ended/exhaustive request.\n"
    "Do not explain your answer. Do not say HIGHEST, it must be only TRIVIAL, LOW, MEDIUM OR HIGH."
)


class ClassifierUnavailable(Exception):
    """Raised when the `lms` CLI can't load or unload the classifier model."""


def _excerpt(text, n=CLASSIFIER_EXCERPT_CHARS):
    """Bound how much text reaches the classifier: a bundled/ambiguous prompt could in principle
    be long, but the classifier call must stay cheap regardless of the input size."""
    if len(text) <= n:
        return text
    half = n // 2
    return f"{text[:half]}\n...\n{text[-half:]}"


def build_classifier_client(cfg=None, model=CLASSIFIER_MODEL):
    base = cfg or Config()
    classifier_cfg = replace(base, model=model, max_completion_tokens=CLASSIFIER_MAX_TOKENS, temperature=0.0)
    return LMStudioClient(classifier_cfg)


def classify_complexity(client, text):
    """Ask the classifier model how much reasoning effort `text` looks like it should need.

    Returns {"label": "trivial" | "low" | "medium" | "high" | "unknown", "raw": str, "error": str | None}.
    """
    user_prompt = f'PROMPT:\n"""\n{_excerpt(text)}\n"""\n\nAnswer:'
    res = client.chat(CLASSIFIER_SYSTEM_PROMPT, user_prompt)
    if res["status"] != "ok":
        return {"label": "unknown", "raw": "", "error": res.get("error")}
    raw = res["text"].strip()
    m = re.search(r"\b(trivial|low|medium|high)\b", raw, re.IGNORECASE)
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
