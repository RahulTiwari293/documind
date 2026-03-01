import uuid
import json
import fitz
from src.embedder import embed
from src import endee_client as db


def load_text(path):
    if path.endswith(".pdf"):
        doc = fitz.open(path)
        return "\n".join(page.get_text() for page in doc)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text, size=400, overlap=80):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += size - overlap
    return chunks


def ingest(file_path, index_name, metric="cosine", precision="FLOAT32"):
    raw = load_text(file_path)
    chunks = chunk_text(raw)
    source = file_path.split("/")[-1]

    try:
        db.create_index(index_name, dims=384, metric=metric, precision=precision)
        print(f"[Endee] Created index '{index_name}' — metric={metric}, precision={precision}")
    except Exception:
        print(f"[Endee] Index '{index_name}' already exists, appending vectors")

    vectors = embed(chunks)

    payload = []
    for i, (vec, chunk) in enumerate(zip(vectors, chunks)):
        metadata = {
            "text": chunk,
            "source": source,
            "chunk_id": i,
            "word_count": len(chunk.split())
        }
        payload.append({
            "id": str(uuid.uuid4()),
            "vector": vec,
            "metadata": metadata,
            "filter": {"source": source, "chunk_id": i}
        })

    batch_size = 64
    for i in range(0, len(payload), batch_size):
        batch = payload[i: i + batch_size]
        db.insert(index_name, batch)
        print(f"[Endee] Inserted batch {i // batch_size + 1} — {len(batch)} vectors")

    print(f"[Endee] Done — {len(payload)} chunks from '{source}' in index '{index_name}'")
    return len(payload)
