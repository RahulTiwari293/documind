import uuid
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
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += size - overlap
    return chunks

def ingest(file_path, index_name, metric="cosine"):
    raw = load_text(file_path)
    chunks = chunk_text(raw)

    try:
        db.create_index(index_name, dims=384, metric=metric)
    except Exception:
        pass

    vectors = embed(chunks)
    source = file_path.split("/")[-1]

    payload = []
    for i, (vec, chunk) in enumerate(zip(vectors, chunks)):
        payload.append({
            "id": str(uuid.uuid4()),
            "vector": vec,
            "metadata": {
                "text": chunk,
                "source": source,
                "chunk_id": i
            }
        })

    batch_size = 64
    for i in range(0, len(payload), batch_size):
        db.upsert(index_name, payload[i: i + batch_size])

    return len(payload)
