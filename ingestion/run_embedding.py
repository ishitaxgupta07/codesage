import json
import sys
import os

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "retrieval"))

from embedder import embed_texts
from chunk_formatter import chunk_to_text
from vector_store import create_collection, upsert_chunks

chunks_path = os.path.join(os.path.dirname(__file__), "..", "chunks.json")
with open(chunks_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")

texts = [chunk_to_text(c) for c in chunks]

print("Generating embeddings...")
embeddings = embed_texts(texts)

create_collection()
upsert_chunks(chunks, embeddings)

print("Done! Chunks embedded and stored in Qdrant.")
