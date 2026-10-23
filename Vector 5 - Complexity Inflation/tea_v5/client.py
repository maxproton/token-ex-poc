"""LM Studio client (OpenAI-compatible endpoint) for V5.

`chat()` is the plain non-streaming call, used by the classifier (classifier.py) for its one-word
complexity judgment. `chat_guarded()` is the actual defence: it streams the completion, separates
`<think>...</think>` reasoning tokens from the visible answer (reasoning models spend tokens here
that never reach the user), and at fixed checkpoints compares the running reasoning-token count
against a ceiling that was predicted *before* generation started. A prompt the classifier judged
cheap but whose reasoning keeps climbing past what that judgment would justify is complexity
inflation happening in real time, and gets cut off mid-stream instead of only being noticed (and
billed) after the fact.
"""
import json
import time

import requests

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _split_reasoning(text):
    """Pull out <think>...</think> content from the visible answer. An unclosed trailing <think>
    (stream aborted mid-thought, or still in progress) is still counted as reasoning."""
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

    def chat(self, system: str, user: str) -> dict:
        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
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

    def chat_guarded(self, system: str, user: str, count_tokens, predicted_ceiling_tokens: int) -> dict:
        """Stream the completion, checking the running reasoning-token count at fixed checkpoints
        against `predicted_ceiling_tokens`. Aborts (status "truncated") once reasoning tokens
        exceed `predicted_ceiling_tokens * ceiling_block_mult`."""
        cfg = self.cfg
        payload = {
            "model": cfg.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
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
                    reasoning_text, _ = _split_reasoning(text)
                    reasoning_tokens = count_tokens(reasoning_text) if reasoning_text.strip() else 0
                    ratio = reasoning_tokens / max(predicted_ceiling_tokens, 1)
                    if ratio >= cfg.ceiling_block_mult:
                        abort_reason = "reasoning_ceiling_exceeded"
                        break
        finally:
            r.close()

        latency = time.perf_counter() - t0
        tokens = count_tokens(text)
        reasoning_text, answer_text = _split_reasoning(text)
        reasoning_tokens = count_tokens(reasoning_text) if reasoning_text.strip() else 0
        answer_tokens = count_tokens(answer_text) if answer_text.strip() else 0
        return {
            "status": "truncated" if abort_reason else "ok",
            "abort_reason": abort_reason,
            "text": answer_text.strip(),
            "reasoning_text": reasoning_text.strip(),
            "completion_tokens": tokens,
            "reasoning_tokens": reasoning_tokens,
            "answer_tokens": answer_tokens,
            "reasoning_ratio": reasoning_tokens / tokens if tokens else 0.0,
            "reasoning_ceiling_ratio": reasoning_tokens / max(predicted_ceiling_tokens, 1),
            "tokens_per_s": tokens / latency if latency > 0 else 0.0,
            "latency_s": latency,
        }
