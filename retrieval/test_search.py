import sys, os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ingestion"))

from embedder import embed_texts
from vector_store import get_client, COLLECTION_NAME

query = "how does httpx handle connection pooling"
query_vector = embed_texts([query])[0]

client = get_client()
results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector.tolist(),
    limit=5
).points

for i, r in enumerate(results):
    print(f"\n--- Result {i+1} (score: {r.score:.4f}) ---")
    print(f"File: {r.payload.get('file')}")
    print(f"Name: {r.payload.get('name', r.payload.get('type'))}")
    print(f"Preview: {str(r.payload.get('code') or r.payload.get('content'))[:200]}")