"""LM Studio client (OpenAI-compatible endpoint) and the simulated victim with a per-request cap."""
import time
import requests


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
            r = requests.post(self.url, json=payload, timeout=self.cfg.timeout_s)
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


class Victim:
    """A typical deployment: fixed system prompt plus a per-request input cap. cap_tokens=None disables the cap."""

    def __init__(self, client, count_tokens, system, cap_tokens=None):
        self.client, self.count, self.system, self.cap = client, count_tokens, system, cap_tokens

    def handle(self, user_text: str) -> dict:
        est = self.count(user_text)
        if self.cap is not None and est > self.cap:
            return {"status": "rejected_by_cap", "est_tokens": est, "latency_s": 0.0}
        res = self.client.chat(self.system, user_text)
        res["est_tokens"] = est
        return res
