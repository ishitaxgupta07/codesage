import sys, os
sys.path.append(os.path.dirname(__file__))
from hybrid_search import hybrid_search

queries = [
    "how does httpx handle connection pooling",   # conceptual query
    "iter_bytes",                                  # exact function name query
]

for q in queries:
    print(f"\n{'='*60}\nQUERY: {q}\n{'='*60}")
    results = hybrid_search(q, top_k=5)
    for i, r in enumerate(results):
        chunk = r["chunk"]
        print(f"\n--- Result {i+1} (RRF score: {r['score']:.4f}) ---")
        print(f"Name: {chunk.get('name', chunk.get('type'))}")
        preview = str(chunk.get("code") or chunk.get("content"))[:150]
        print(f"Preview: {preview}")