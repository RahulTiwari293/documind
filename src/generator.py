import os
from groq import Groq
from src.config import get_secret

client = Groq(api_key=get_secret("GROQ_API_KEY", "GROQ_API_KEY"))
model = get_secret("GROQ_MODEL", "GROQ_MODEL") or "llama-3.1-8b-instant"

def generate(question, context_chunks):
    context = "\n\n".join(
        f"[Source: {hit['source']}, chunk {hit['chunk_id']}]\n{hit['text']}"
        for hit in context_chunks
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the user's question based only on the "
                "provided context. If the context doesn't contain the answer, say so honestly. "
                "Be concise and factual."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]

    response = client.chat.completions.create(model=model, messages=messages, temperature=0.2)
    return response.choices[0].message.content
