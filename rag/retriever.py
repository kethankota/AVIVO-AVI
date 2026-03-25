import sqlite3
from pathlib import Path

import numpy as np

from rag.embedder import embed_query

DB_PATH = Path("/app/data/embeddings.db")


def load_all_chunks() -> list[dict]:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Embeddings DB not found at {DB_PATH}. Run: python -m rag.ingest"
        )
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, source, text, vector FROM chunks")
    rows = cur.fetchall()
    conn.close()

    chunks = []
    for row_id, source, text, vec_bytes in rows:
        vec = np.frombuffer(vec_bytes, dtype=np.float32)
        chunks.append({"id": row_id, "source": source,
                      "text": text, "vector": vec})
    return chunks


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    query_vec = embed_query(query)
    chunks = load_all_chunks()

    if not chunks:
        return []

    matrix = np.stack([c["vector"] for c in chunks])

    scores = matrix @ query_vec

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({
            "text": chunks[idx]["text"],
            "source": chunks[idx]["source"],
            "score": float(scores[idx]),
        })
    return results
