from src.embedder import embed_one
from src import endee_client as db

MIN_SCORE = 0.35


def retrieve(question, index_name, top_k=5, source_filter=None, chunk_range=None):
    """
    Semantic search against an Endee index.

    source_filter  – only return chunks from a specific file
                     uses Endee's $eq metadata filter
    chunk_range    – only return chunks within a position range
                     uses Endee's $range metadata filter
    MIN_SCORE      – discard results with cosine similarity below this threshold
    """
    query_vector = embed_one(question)

    filters = {}
    if source_filter:
        filters["source"] = {"$eq": source_filter}
    if chunk_range:
        filters["chunk_id"] = {"$range": list(chunk_range)}

    raw = db.search(
        index_name,
        query_vector,
        top_k=top_k,
        filters=filters if filters else None
    )

    hits = []
    for match in raw.get("results", []):
        score = round(match.get("score", 0.0), 4)
        if score < MIN_SCORE:
            continue
        meta = match.get("metadata", {})
        hits.append({
            "text": meta.get("text", ""),
            "source": meta.get("source", "unknown"),
            "chunk_id": meta.get("chunk_id", -1),
            "word_count": meta.get("word_count", 0),
            "score": score
        })

    return sorted(hits, key=lambda x: x["score"], reverse=True)


def retrieve_from_multiple_sources(question, index_name, sources, top_k=5):
    """
    Use Endee's $in filter to search across a specific subset of sources.
    Useful when an index holds multiple documents and you want to restrict
    the search to a named list of files.
    """
    query_vector = embed_one(question)
    filters = {"source": {"$in": sources}}
    raw = db.search(index_name, query_vector, top_k=top_k, filters=filters)

    hits = []
    for match in raw.get("results", []):
        score = round(match.get("score", 0.0), 4)
        if score < MIN_SCORE:
            continue
        meta = match.get("metadata", {})
        hits.append({
            "text": meta.get("text", ""),
            "source": meta.get("source", "unknown"),
            "chunk_id": meta.get("chunk_id", -1),
            "score": score
        })

    return sorted(hits, key=lambda x: x["score"], reverse=True)
