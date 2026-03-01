from src.embedder import embed_one
from src import endee_client as db

def retrieve(question, index_name, top_k=5):
    query_vector = embed_one(question)
    raw = db.search(index_name, query_vector, top_k=top_k)

    hits = []
    for match in raw.get("results", []):
        meta = match.get("metadata", {})
        hits.append({
            "text": meta.get("text", ""),
            "source": meta.get("source", "unknown"),
            "chunk_id": meta.get("chunk_id", -1),
            "score": round(match.get("score", 0.0), 4)
        })
    return hits
