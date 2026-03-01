import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = os.getenv("GROQ_MODEL", "llama3-8b-8192")

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
