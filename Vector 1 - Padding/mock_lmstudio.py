"""Tiny stand-in for the LM Studio server so the harness can be dry-run without loading a model.
Token counts are fake (4 chars per token, invisible characters count double)."""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


def fake_tokens(text):
    inv = sum(1 for c in text if c in "\u200b\u200c\u2060")
    return max(1, (len(text) - inv) // 4 + inv * 2)


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = " ".join(m["content"] for m in body["messages"])
        pt = fake_tokens(text)
        time.sleep(min(pt / 200000, 0.5))
        out = json.dumps({"choices": [{"message": {"content": "391 Canberra Na"}}],
                          "usage": {"prompt_tokens": pt, "completion_tokens": 5, "total_tokens": pt + 5}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=1234)
    HTTPServer(("127.0.0.1", ap.parse_args().port), H).serve_forever()
