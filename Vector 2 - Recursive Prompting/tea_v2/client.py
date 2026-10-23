"""LM Studio client (OpenAI-compatible endpoint) for V2.

`chat()` is the plain non-streaming call, used for one-shot requests. `chat_guarded()` is the
output-side defence: it streams the completion and re-examines it at fixed token checkpoints so a
response that is degenerating into a loop, or has simply run far past what the question needed, can
be cut off mid-stream instead of only being noticed (and billed) after the fact. It also separates
`<think>...</think>` reasoning tokens from the visible answer, since a blown-out reasoning trace and
a blown-out answer are distinct failure patterns (see paper section 5.2).
"""
import json
import re
import time
import zlib

import requests

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _repetition_ratio(text, n=5, min_words=20):
    """Share of duplicate n-word sequences. Reimplemented independently of tea_v1's detector so
    this folder has no cross-vector import, per the project's one-folder-per-vector convention."""
    words = text.split()
    if len(words) < min_words:
        return 0.0
    grams = [hash(tuple(words[i:i + n])) for i in range(len(words) - n + 1)]
    return 1 - len(set(grams)) / len(grams)


def _compression_ratio(text, min_chars=200):
    """zlib size / raw size. Lower means more redundant. Meaningless on very short text."""
    if len(text) < min_chars:
        return 1.0
    raw = text.encode("utf-8")
    return len(zlib.compress(raw, 6)) / len(raw)


def _split_reasoning(text):
    """Pull out <think>...</think> content (reasoning-model traces) from the visible answer.
    An unclosed trailing <think> (stream aborted mid-thought) is still counted as reasoning."""
    reasoning_parts = []
    answer = text
    while _THINK_OPEN in answer:
        start = answer.index(_THINK_OPEN)
        end = answer.find(_THINK_CLOSE, start)
        if end == -1:
            reasoning_parts.append(answer[start + len(_THINK_OPEN):])
            answer = answer[:start]
            break
        reasoning_parts.append(answer[start + len(_THINK_OPEN):end])
        answer = answer[:start] + answer[end + len(_THINK_CLOSE):]
    return "".join(reasoning_parts), answer


class LMStudioClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.url = cfg.base_url.rstrip("/") + "/chat/completions"

    def chat(self, system: str, user: str, history: list = None) -> dict:
        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user})
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_completion_tokens,
            "stream": False,
        }
        t0 = time.perf_counter()
        try:
            r = requests.post(self.url, json=payload, timeout=self.cfg.timeout_s, proxies={"http": None, "https": None})
        except requests.RequestException as e:
            return {"status": "error", "error": str(e)[:300], "latency_s": time.perf_counter() - t0}
        latency = time.perf_counter() - t0
        if r.status_code != 200:
            return {"status": "error", "error": r.text[:300], "latency_s": latency}
        data = r.json()
        usage = data.get("usage") or {}
        return {
            "status": "ok",
            "text": data["choices"][0]["message"]["content"],
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "latency_s": latency,
        }

    def chat_guarded(self, system: str, user: str, count_tokens, history: list = None,
                     reasoning_ceiling_tokens: int = None) -> dict:
        """Stream the completion, checking it at fixed token checkpoints. Aborts the connection
        (status "truncated") on degenerate repetition/compression or on exceeding the predicted
        ceiling by the block multiplier, rather than waiting for the model to stop on its own.
        `history` is an optional list of prior {"role", "content"} turns, resent ahead of `user`
        so the model actually sees a growing conversation rather than one isolated message.
        `reasoning_ceiling_tokens`, if given, is a per-prompt predicted reasoning-token budget
        (e.g. from V5's complexity classifier); the stream is also aborted once actual reasoning
        tokens exceed it by `reasoning_ceiling_block_mult`, independent of the (total-token)
        ceiling check below."""
        cfg = self.cfg
        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user})
        payload = {
            "model": cfg.model,
            "messages": messages,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_completion_tokens,
            "stream": True,
        }
        t0 = time.perf_counter()
        try:
            r = requests.post(self.url, json=payload, timeout=cfg.timeout_s, stream=True,
                               proxies={"http": None, "https": None})
        except requests.RequestException as e:
            return {"status": "error", "error": str(e)[:300], "latency_s": time.perf_counter() - t0}
        if r.status_code != 200:
            latency = time.perf_counter() - t0
            return {"status": "error", "error": r.text[:300], "latency_s": latency}

        text = ""
        last_checkpoint_tokens = 0
        abort_reason = None
        try:
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta", {}) or {}
                piece = delta.get("content") or ""
                if piece:
                    text += piece
                tokens = count_tokens(text)
                if tokens - last_checkpoint_tokens >= cfg.checkpoint_tokens:
                    last_checkpoint_tokens = tokens
                    abort_reason = self._checkpoint_violation(text, tokens, t0, count_tokens, reasoning_ceiling_tokens)
                    if abort_reason:
                        break
        finally:
            r.close()

        latency = time.perf_counter() - t0
        tokens = count_tokens(text)
        reasoning_text, answer_text = _split_reasoning(text)
        reasoning_tokens = count_tokens(reasoning_text) if reasoning_text.strip() else 0
        answer_tokens = count_tokens(answer_text) if answer_text.strip() else 0
        ceiling_ratio = tokens / max(cfg.predicted_ceiling_tokens, 1)
        result = {
            "status": "truncated" if abort_reason else "ok",
            "abort_reason": abort_reason,
            "text": answer_text.strip(),
            "reasoning_text": reasoning_text.strip(),
            "completion_tokens": tokens,
            "reasoning_tokens": reasoning_tokens,
            "answer_tokens": answer_tokens,
            "reasoning_ratio": reasoning_tokens / tokens if tokens else 0.0,
            "ceiling_ratio": ceiling_ratio,
            "tokens_per_s": tokens / latency if latency > 0 else 0.0,
            "repetition": _repetition_ratio(text),
            "compression": _compression_ratio(text),
            "latency_s": latency,
        }
        if reasoning_ceiling_tokens is not None:
            result["reasoning_ceiling_ratio"] = reasoning_tokens / max(reasoning_ceiling_tokens, 1)
        return result

    def _checkpoint_violation(self, text, tokens, t0, count_tokens, reasoning_ceiling_tokens=None):
        cfg = self.cfg
        ceiling_ratio = tokens / max(cfg.predicted_ceiling_tokens, 1)
        if ceiling_ratio >= cfg.ceiling_block_mult:
            return "ceiling_exceeded"
        if reasoning_ceiling_tokens is not None:
            reasoning_text, _ = _split_reasoning(text)
            reasoning_tokens = count_tokens(reasoning_text) if reasoning_text.strip() else 0
            if reasoning_tokens / max(reasoning_ceiling_tokens, 1) >= cfg.reasoning_ceiling_block_mult:
                return "reasoning_ceiling_exceeded"
        rep = _repetition_ratio(text)
        if rep >= cfg.output_rep_block:
            return "degenerate_repetition"
        comp = _compression_ratio(text)
        if comp <= cfg.output_comp_block:
            return "degenerate_repetition"
        elapsed = time.perf_counter() - t0
        if elapsed > 0 and tokens / elapsed >= cfg.tps_block:
            return "tokens_per_second_anomaly"
        return None
