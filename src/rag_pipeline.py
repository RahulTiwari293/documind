from src.retriever import retrieve, retrieve_from_multiple_sources
from src.generator import generate
from src import endee_client as db


def ask(question, index_name, top_k=5, source_filter=None):
    hits = retrieve(question, index_name, top_k=top_k, source_filter=source_filter)

    if not hits:
        return {
            "answer": "No relevant information found. The document may not cover this topic, or try rephrasing.",
            "sources": [],
            "top_score": 0.0
        }

    answer = generate(question, hits)
    top_score = hits[0]["score"] if hits else 0.0

    return {
        "answer": answer,
        "sources": hits,
        "top_score": top_score
    }


def ask_across_sources(question, index_name, sources, top_k=5):
    hits = retrieve_from_multiple_sources(question, index_name, sources, top_k=top_k)

    if not hits:
        return {
            "answer": "No relevant information found in the selected sources.",
            "sources": [],
            "top_score": 0.0
        }

    answer = generate(question, hits)
    return {"answer": answer, "sources": hits, "top_score": hits[0]["score"]}


def index_summary(index_name):
    try:
        info = db.index_info(index_name)
        indexes = db.list_indexes()
        all_names = [idx.get("name", "") for idx in indexes.get("indexes", [])]
        return {"info": info, "all_indexes": all_names}
    except Exception as e:
        return {"error": str(e)}
