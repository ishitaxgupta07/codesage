from sentence_transformers import SentenceTransformer

_model = None

def get_model():
    global _model
    if _model is None:
        print("Loading embedding model (first time only, may take a minute)...")
        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _model

def embed_texts(texts):
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return embeddings