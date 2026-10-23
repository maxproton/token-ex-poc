"""Retrieval-layer guardrail in front of the local LLM for V3 (context window stuffing).

Run from this folder:  python guardrail.py

check() retrieves the top_k chunks for a query from the local corpus and scores the *set of
chunks*, not the query text: duplicate/near-duplicate chunks, an abnormally large chunk, irrelevant
chunks, and total context inflation versus what a clean retrieval for this query should look like.
assemble_context() then builds the final RAG prompt from the (accepted) chunks plus the query, the
thing that actually gets sent onward to the model.

Type a query on one line. Type quit to exit.
"""
from tea_v3.config import Config, RetrieverConfig
from tea_v3.client import LMStudioClient
from tea_v3.detector import ALLOW, Detector, DetectorConfig
from tea_v3 import retriever

# ============================ EDIT THESE ============================
THRESHOLD = 0.70              # reject if score > THRESHOLD. 0.5 is the detector's "flag" level, 1.0 never rejects.
SHOW_BREAKDOWN = True

SYSTEM_PROMPT = ("You are a helpful assistant. Answer the user's question using only the provided "
                  "context. If the context does not contain the answer, say so.")

# Per check: (safe, flag, block). Risk is 0.0 at "safe", 0.5 at "flag", 1.0 at "block", linear in between.
BANDS = {
    "duplicate_ratio":    (0.0, 0.2, 0.4),
    "oversized_ratio":    (1.0, 1.5, 3.0),
    "irrelevant_share":   (0.0, 0.3, 0.6),
    "context_inflation":  (1.0, 1.5, 3.0),
}
# ====================================================================


def risk(value, safe, flag, block):
    if value <= safe:
        return 0.0
    if value >= block:
        return 1.0
    if value <= flag:
        return 0.5 * (value - safe) / (flag - safe)
    return 0.5 + 0.5 * (value - flag) / (block - flag)


def score_chunks(detector, chunks, top_k, normal_chunk_chars):
    d = detector.inspect(chunks, top_k, normal_chunk_chars)
    parts = {}
    for c in d.checks:
        band = BANDS.get(c.name)
        if band is None:
            continue
        parts[c.name] = (c.value, risk(c.value, *band))
    score = max((r for _, r in parts.values()), default=0.0)
    return d, score, parts


class Guardrail:
    """Retrieves and scores context for a query, and gates access to the model. Shared by the CLI
    here and by other front-ends (e.g. the Flask UI in web_interface/), same pattern as tea_v1 and
    tea_v2's Guardrail."""

    def __init__(self, cfg=None, retriever_cfg=None):
        self.cfg = cfg or Config()
        self.rcfg = retriever_cfg or RetrieverConfig()
        self.detector = Detector(DetectorConfig())
        self.client = LMStudioClient(self.cfg)

    def check(self, query):
        """Retrieve chunks for `query` and score the retrieved set. Returns
        (allowed, score, parts, decision, chunks)."""
        chunks = retriever.retrieve(query, self.rcfg.corpus_dir, self.rcfg.top_k)
        d, score, parts = score_chunks(self.detector, chunks, self.rcfg.top_k, self.rcfg.normal_chunk_chars)
        allowed = score <= THRESHOLD
        return allowed, score, parts, d, chunks

    def assemble_context(self, query, chunks):
        """Build the final RAG prompt: the accepted context chunks plus the query. This, not the
        bare query, is what gets sent to the model."""
        if not chunks:
            return query
        context = "\n\n".join(f"[{c.doc}] {c.text}" for c in chunks)
        return f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"

    def ask(self, query, chunks):
        """Standalone, non-streaming call for the CLI. Used when wired into the web UI, V2's
        streaming-guarded ask() is used instead so output-side protection still applies."""
        return self.client.chat(SYSTEM_PROMPT, self.assemble_context(query, chunks))


def main():
    guardrail = Guardrail()
    print(f"Guardrail ready. Threshold {THRESHOLD}. Corpus {guardrail.rcfg.corpus_dir}. "
          f"top_k={guardrail.rcfg.top_k}.")

    while True:
        try:
            query = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in {"quit", "exit"}:
            break
        if not query:
            continue

        allowed, score, parts, d, chunks = guardrail.check(query)
        driver = max(parts, key=lambda k: parts[k][1]) if parts else "none"
        print(f"chunks={d.chunks}  chars={d.chars}  score={score:.2f}  (driven by {driver})")
        if SHOW_BREAKDOWN:
            for name, (value, r) in parts.items():
                print(f"   {name:<18}{value:>10.3f}   risk {r:.2f}")
            for c in chunks:
                print(f"   [{c.doc}] score={c.score:.2f} chars={c.chars}  {c.text[:80]!r}")

        if not allowed:
            print(f"REJECTED: score {score:.2f} is above threshold {THRESHOLD}. The model was not called.")
            continue

        res = guardrail.ask(query, chunks)
        if res["status"] != "ok":
            print(f"Model error: {res.get('error')}")
            continue
        print(f"\n{res['text']}")


if __name__ == "__main__":
    main()
