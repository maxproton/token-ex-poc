"""LM Studio client (OpenAI-compatible endpoint) for V4's standalone CLI. The web app routes
generation through V2's streaming output guard instead, same as V3."""
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
