import json
import os
import sqlite3

from pathlib import Path
from sentence_transformers import SentenceTransformer

DOCS_DIR = Path("/app/docs")
DB_PATH = Path("/app/data/embeddings.db")
chunk_size = int(os.environ.get("CHUNK_SIZE", 250))
chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", 50))
embed_model = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def split_into_chunks(text: str, source: str) -> list[dict]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({"text": chunk_text, "source": source})
        if end == len(words):
            break
        start += chunk_size - chunk_overlap
    return chunks


def load_docs() -> list[dict]:
    all_chunks = []
    for path in DOCS_DIR.glob("**/*"):
        if path.suffix not in {".md", ".txt"}:
            continue
        content = path.read_text(encoding="utf-8")
        source = path.name
        chunks = split_into_chunks(content, source)
        all_chunks.extend(chunks)
        print(f"  Loaded {path.name} → {len(chunks)} chunks")
    return all_chunks


def run() -> None:
    print("=== Ingestion started ===")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(embed_model)
    chunks = load_docs()

    if not chunks:
        print("No documents found in docs/. Add .md or .txt files and re-run.")
        return

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts, show_progress_bar=True, normalize_embeddings=True)
    dim = embeddings.shape[1]

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS chunks")
    cur.execute("""
        CREATE TABLE chunks (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            source  TEXT NOT NULL,
            text    TEXT NOT NULL,
            vector  BLOB NOT NULL
        )
    """)

    for i, chunk in enumerate(chunks):
        vec_bytes = embeddings[i].tobytes()
        cur.execute(
            "INSERT INTO chunks (source, text, vector) VALUES (?, ?, ?)",
            (chunk["source"], chunk["text"], vec_bytes),
        )

    conn.commit()
    conn.close()
    print(
        f"\n=== Done. {len(chunks)} chunks stored in {DB_PATH} (dim={dim}) ===")


if __name__ == "__main__":
    run()
