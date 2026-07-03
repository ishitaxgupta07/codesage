import sys, os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ingestion"))

from embedder import embed_texts
from vector_store import get_client, COLLECTION_NAME
from bm25_search import bm25_search

def vector_search(query, top_k=10):
    query_vector = embed_texts([query])[0]
    client = get_client()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=top_k
    ).points
    return [{"score": r.score, "chunk": r.payload} for r in results]

def chunk_key(chunk):
    # unique-ish identifier for a chunk to dedupe/match across both searches
    return (chunk.get("file"), chunk.get("name"), chunk.get("start_line"), chunk.get("content", "")[:50])

def hybrid_search(query, top_k=5, k_rrf=60):
    vec_results = vector_search(query, top_k=20)
    bm25_results = bm25_search(query, top_k=20)

    rrf_scores = {}
    chunk_lookup = {}

    for rank, r in enumerate(vec_results):
        key = chunk_key(r["chunk"])
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k_rrf + rank + 1)
        chunk_lookup[key] = r["chunk"]

    for rank, r in enumerate(bm25_results):
        key = chunk_key(r["chunk"])
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k_rrf + rank + 1)
        chunk_lookup[key] = r["chunk"]

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    final = []
    for key, score in ranked[:top_k]:
        final.append({"score": score, "chunk": chunk_lookup[key]})
    return final