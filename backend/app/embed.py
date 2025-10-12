from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from .settings import settings

_model = None


def get_model():
    global _model
    if _model is None:
        # Small, fast, good quality; 384-dim
        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _model

def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_model()
    # generate embeddings for all texts
    vecs = model.encode(texts, normalize_embeddings=True)

    # sanity check: make sure dimensionality matches config
    for i, v in enumerate(vecs):
        if len(v) != settings.EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dimension mismatch for item {i}: "
                f"expected {settings.EMBEDDING_DIM}, got {len(v)}"
            )
    return vecs
