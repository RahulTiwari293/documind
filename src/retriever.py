import json
from src.embedder import embed_one
from src import endee_client as db

MIN_SCORE = 0.10


def _parse_hit(match):
    score = float(match[0])
    meta_raw = match[2] if len(match) > 2 else b"{}"
    if isinstance(meta_raw, (bytes, bytearray)):
        meta_raw = meta_raw.decode("utf-8")
    meta = json.loads(meta_raw) if meta_raw else {}
    return {
        "text": meta.get("text", ""),
        "source": meta.get("source", "unknown"),
        "chunk_id": meta.get("chunk_id", -1),
        "word_count": meta.get("word_count", 0),
        "score": round(score, 4)
    }


def retrieve(question, index_name, top_k=5, source_filter=None):
    query_vector = embed_one(question)

    filters = None
    if source_filter:
        filters = [{"source": {"$eq": source_filter}}]

    raw = db.search(index_name, query_vector, top_k=top_k, filters=filters)

    results = raw if isinstance(raw, list) else []
    hits = [_parse_hit(m) for m in results if float(m[0]) >= MIN_SCORE]
    return sorted(hits, key=lambda x: x["score"], reverse=True)


def retrieve_from_multiple_sources(question, index_name, sources, top_k=5):
    query_vector = embed_one(question)
    filters = [{"source": {"$in": sources}}]
    raw = db.search(index_name, query_vector, top_k=top_k, filters=filters)

    results = raw if isinstance(raw, list) else []
    hits = [_parse_hit(m) for m in results if float(m[0]) >= MIN_SCORE]
    return sorted(hits, key=lambda x: x["score"], reverse=True)
