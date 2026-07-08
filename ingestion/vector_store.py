from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

COLLECTION_NAME = "codesage_httpx"
VECTOR_SIZE = 384  # bge-small-en-v1.5 output dimension

import os
from dotenv import load_dotenv
load_dotenv()

def get_client():
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    if qdrant_url:
        return QdrantClient(url=qdrant_url, api_key=qdrant_key)
    return QdrantClient(host="localhost", port=6333)

def create_collection():
    client = get_client()
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        print(f"Collection already exists: {COLLECTION_NAME}")

def upsert_chunks(chunks, embeddings):
    client = get_client()
    points = []
    for chunk, vector in zip(chunks, embeddings):
        payload = dict(chunk)  # store all chunk metadata alongside the vector
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload=payload,
            )
        )
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Upserted {len(points)} points into {COLLECTION_NAME}")