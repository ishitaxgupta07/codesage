import json
import os
import re
from rank_bm25 import BM25Okapi

_bm25 = None
_chunks = None

def tokenize(text):
    return re.findall(r"\w+", text.lower())

def load_bm25_index():
    global _bm25, _chunks
    if _bm25 is not None:
        return _bm25, _chunks

    chunks_path = os.path.join(os.path.dirname(__file__), "..", "chunks.json")
    with open(chunks_path, "r", encoding="utf-8") as f:
        _chunks = json.load(f)

    corpus = []
    for c in _chunks:
        if c.get("type") == "doc_section":
            text = c.get("content", "")
        else:
            text = f"{c.get('name','')} {c.get('docstring','')} {c.get('code','')}"
        corpus.append(tokenize(text))

    _bm25 = BM25Okapi(corpus)
    return _bm25, _chunks

def bm25_search(query, top_k=5):
    bm25, chunks = load_bm25_index()
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    results = []
    for score, chunk in ranked[:top_k]:
        results.append({"score": float(score), "chunk": chunk})
    return results