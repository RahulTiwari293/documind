from src.retriever import retrieve
from src.generator import generate

def ask(question, index_name, top_k=5):
    hits = retrieve(question, index_name, top_k=top_k)

    if not hits:
        return {
            "answer": "No relevant documents found in the index.",
            "sources": []
        }

    answer = generate(question, hits)

    sources = [
        {"source": h["source"], "chunk_id": h["chunk_id"], "score": h["score"], "text": h["text"]}
        for h in hits
    ]

    return {"answer": answer, "sources": sources}
