import os
import numpy as np
from sentence_transformers import SentenceTransformer

model_name = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def get_model() -> SentenceTransformer:
    model: SentenceTransformer | None = None
    if model is None:
        model = SentenceTransformer(model_name)
    return model


def embed_query(text: str) -> np.ndarray:
    model = get_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.astype(np.float32)
