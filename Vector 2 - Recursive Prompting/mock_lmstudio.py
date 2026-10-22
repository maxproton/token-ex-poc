"""Tiny stand-in for the LM Studio server so the harness can be dry-run without loading a model.
Unlike tea_v1's mock, this one streams: it serves a normal short reply on most requests, but any
prompt containing the word "loop" in the user turn gets an endless near-duplicate stream instead,
so chat_guarded()'s checkpoint-abort logic can be exercised without a real model."""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


def _sse(obj):
    return f"data: {json.dumps(obj)}\n\n".encode()


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        user_text = next((m["content"] for m in body["messages"] if m["role"] == "user"), "")
        stream = body.get("stream", False)

        if not stream:
            out = json.dumps({"choices": [{"message": {"content": "391"}}],
                               "usage": {"prompt_tokens": 10, "completion_tokens": 1}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        if "loop" in user_text.lower():
            # Degenerate: the same five words forever, simulating runaway/repetitive generation.
            for _ in range(2000):
                self.wfile.write(_sse({"choices": [{"delta": {"content": "forever and ever again "}}]}))
                time.sleep(0.001)
        else:
            for piece in ["391", " is", " the", " answer", "."]:
                self.wfile.write(_sse({"choices": [{"delta": {"content": piece}}]}))
                time.sleep(0.001)
        self.wfile.write(_sse("[DONE]").replace(b'"[DONE]"', b"[DONE]"))

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1234)
    HTTPServer(("127.0.0.1", ap.parse_args().port), H).serve_forever()
