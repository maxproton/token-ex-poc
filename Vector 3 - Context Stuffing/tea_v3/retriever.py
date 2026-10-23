"""Naive retrieval over a local corpus, standing in for a RAG pipeline's retriever. Documents are
chunked by paragraph (blank-line separated) with no hard size cap -- a common and realistic gap in
simple splitters, and exactly the gap the oversized-chunk check in detector.py exists to catch.
Relevance is plain word-overlap (Jaccard), not embeddings: this harness tests the chunk-level
defences, not retrieval quality itself.
"""
import re
from dataclasses import dataclass
from pathlib import Path

_WORD_RE = re.compile(r"\w+")

# Common English function words excluded from relevance scoring: without this, a stray shared
# stopword ("new", "how", "through") is enough to pull a chunk into the results purely because it
# shares a functional word with the query, not because it's actually relevant -- and for a highly
# repetitive chunk, keeping its total vocabulary small also inflates its Jaccard score. A standard,
# broad stopword list is used (rather than patching in words one at a time as they're found to cause
# false matches) since any single common word missing from a narrower list reproduces the same bug.
_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few",
    "for", "from", "further", "get", "gets", "getting", "had", "has", "have", "having", "he", "her",
    "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "itself", "just", "let's", "like", "many", "me", "might", "more", "most", "must", "my",
    "myself", "need", "new", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "please", "same", "shall",
    "she", "should", "so", "some", "still", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "would", "yet", "you", "your", "yours",
    "yourself", "yourselves",
}


def _tokens(text):
    return set(w.lower() for w in _WORD_RE.findall(text)) - _STOPWORDS


def _split_paragraphs(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


@dataclass
class Chunk:
    doc: str
    text: str
    score: float

    @property
    def chars(self):
        return len(self.text)


def load_corpus(corpus_dir):
    """doc_id -> raw text, for every .txt file in corpus_dir."""
    root = Path(corpus_dir)
    if not root.is_dir():
        return {}
    return {p.stem: p.read_text(encoding="utf-8", errors="ignore") for p in sorted(root.glob("*.txt"))}


def chunk_documents(docs):
    """doc_id -> text mapping to a flat list of (doc_id, paragraph) chunks, unscored."""
    chunks = []
    for doc_id, text in docs.items():
        for para in _split_paragraphs(text):
            chunks.append(Chunk(doc=doc_id, text=para, score=0.0))
    return chunks


def retrieve(query, corpus_dir, top_k):
    """Score every chunk in the corpus against the query by word-overlap and return the top_k.

    Chunks with zero overlap are dropped before the top_k cut: a real retriever has nothing to
    offer for a query that shares no vocabulary with the corpus at all, and forcing top_k slots to
    fill anyway would make every off-topic or too-short query look like a 100% irrelevant,
    content-stuffed retrieval to the detector -- indistinguishable from an actual attack. A query
    that shares at least one real word with a document is a genuine (if possibly weak) match, which
    the irrelevant_share check downstream still penalizes if it's too weak.
    """
    docs = load_corpus(corpus_dir)
    chunks = chunk_documents(docs)
    q = _tokens(query)
    scored = []
    for c in chunks:
        t = _tokens(c.text)
        overlap = len(q & t) / len(q | t) if (q or t) else 0.0
        if overlap > 0:
            scored.append(Chunk(doc=c.doc, text=c.text, score=overlap))
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:top_k]
