# embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np
import os

_model = None
EMB_MODEL = os.getenv("EMB_MODEL", "all-mpnet-base-v2")  # better quality than MiniLM

def get_embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMB_MODEL)
    return _model

def embed_chunks(chunks):
    model = get_embedder()
    embs = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return np.array(embs, dtype="float32")

# persistence helpers
def save_embeddings(path: str, embeddings: np.ndarray, metas: list, texts: list):
    """Save embeddings + meta + texts as npz"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, embeddings=embeddings, metas=metas, texts=texts)

def load_embeddings(path: str):
    data = np.load(path, allow_pickle=True)
    return data["embeddings"], data["metas"].tolist(), data["texts"].tolist()
