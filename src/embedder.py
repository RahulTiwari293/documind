from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts):
    return model.encode(texts, convert_to_numpy=True).tolist()

def embed_one(text):
    return embed([text])[0]
