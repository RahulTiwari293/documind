# 📖 LEARN.md — Understanding DocuMind & Endee

This file is your complete guide to what this project does, how each piece works,
and how Endee powers the entire thing. Written for someone applying to Endee —
so you can speak confidently about every design decision.

---

## 1. The Problem We're Solving

Imagine you have a 200-page internal company document. You want to ask:

> "What is the return policy for international shipments?"

You *could* Ctrl+F for keywords. But keyword search breaks when:
- The document says "overseas delivery" instead of "international shipments"
- The answer is spread across 3 paragraphs
- You don't know the exact word to search for

**Large Language Models (LLMs)** like GPT and LLaMA can read and answer — but:
- They have a training cutoff (they don't know your private document)
- They hallucinate (they confidently make up answers)
- You can't feed a 200-page PDF into their context window

**Solution: Retrieval-Augmented Generation (RAG)**

Instead of asking the LLM to "remember" your document, we:
1. Break the document into chunks
2. Convert chunks to numbers (embeddings) that capture meaning
3. Store those numbers in a vector database (Endee)
4. At query time, find the chunks *closest in meaning* to the question
5. Give those chunks to the LLM as context → factual, grounded answer

---

## 2. What is a Vector and Why Does It Matter?

A vector is just a list of numbers. For example:

```
"neural network" → [0.23, -0.61, 0.88, ..., 0.14]   (384 numbers)
"deep learning"  → [0.21, -0.59, 0.91, ..., 0.11]   (384 numbers)
"banana bread"   → [-0.72, 0.33, -0.14, ..., 0.90]  (384 numbers)
```

"neural network" and "deep learning" produce very similar vectors because their
meanings are similar. "banana bread" is far away.

This is called an **embedding** — an embedding model reads text and maps it to
a point in 384-dimensional space, where similar meanings cluster together.

We use `sentence-transformers/all-MiniLM-L6-v2`, which is:
- Free, runs locally (no API key)
- Fast (~14,000 sentences/second on GPU)
- Produces 384-dimensional embeddings
- Strong at semantic similarity tasks

```python
# src/embedder.py
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_one(text):
    return model.encode([text], convert_to_numpy=True).tolist()[0]

# "What is a neural network?" → [0.23, -0.61, 0.88, ...]  (384 floats)
```

---

## 3. What is Endee and Why Use It?

**Endee** is an open-source, high-performance vector database. Think of it as
PostgreSQL, but instead of storing rows and doing SQL queries, it stores vectors
and does similarity searches.

### Why Endee specifically?

| Feature | What it means for us |
|---|---|
| HNSW indexing | Sub-5ms search on millions of vectors |
| Open source | Self-hosted, no cloud costs, full data privacy |
| Simple REST API | No heavy SDK needed — just HTTP calls |
| Multiple distance metrics | Cosine, L2, inner product |
| Metadata support | Store text + source alongside each vector |
| 5 quantization levels | FLOAT32 for accuracy, BINARY for max speed |

### The Endee workflow (exactly what our code does):

```
Step 1: Create an index (like creating a table in SQL)
   POST /api/v1/index/create
   { "name": "documind", "dimension": 384, "metric": "cosine" }

Step 2: Upsert vectors (store chunks with their embeddings)
   POST /api/v1/index/documind/upsert
   { "vectors": [{ "id": "abc-123", "vector": [0.23, ...], "metadata": {"text": "...", "source": "doc.pdf"} }] }

Step 3: Search (at query time)
   POST /api/v1/index/documind/search
   { "vector": [0.21, ...], "top_k": 5 }
   → Returns the 5 most similar chunks, ranked by cosine similarity
```

### Distance metrics explained

- **Cosine similarity**: Measures the *angle* between two vectors. Two chunks about
  the same topic point in the same direction even if one is longer. Best for text.
- **L2 (Euclidean)**: Measures straight-line distance. Sensitive to vector magnitude.
- **Inner product**: Dot product — used when vectors are normalized.

We use **cosine** because text length shouldn't affect similarity (a one-sentence
summary and a full paragraph on the same topic should match well).

---

## 4. How Endee's HNSW Index Works

HNSW = Hierarchical Navigable Small World. It's a graph-based indexing algorithm.

**The problem**: With 1 million vectors, comparing your query to each one would take
millions of dot products — too slow for real-time use.

**HNSW's solution**: Build a multi-layer graph where:
- Bottom layer: all vectors connected to their nearest neighbors
- Upper layers: progressively sparser "express lanes" for fast navigation

At search time, you start at the top (sparse) layer, find the approximate region
your query belongs to, then descend to the dense layer for precise results.

Result: instead of 1,000,000 comparisons, you do ~50–100. Sub-5ms latency.

---

## 5. Walking Through the Code — File by File

### `src/embedder.py` — The Embedding Engine

```python
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts):
    return model.encode(texts, convert_to_numpy=True).tolist()

def embed_one(text):
    return embed([text])[0]
```

**What it does**: Takes any string(s) and converts them into 384-float vectors.
Called during both ingestion (encoding chunks) and retrieval (encoding the query).
The same model is used for both — this is critical. If you encode chunks with
model A and queries with model B, the vectors live in different spaces and search
is meaningless.

---

### `src/endee_client.py` — The Database Driver

```python
def create_index(name, dims=384, metric="cosine"):
    payload = {"name": name, "dimension": dims, "metric": metric, "precision": "FLOAT32"}
    resp = requests.post(f"{base_url}/api/v1/index/create", json=payload, headers=headers())
    resp.raise_for_status()
    return resp.json()
```

**What it does**: Wraps Endee's REST API in clean Python functions. Pure `requests`
— no SDK, no magic.  Each function is one HTTP call:
- `create_index()` → registers a new index
- `upsert()` → adds/updates vectors
- `search()` → finds top-k nearest vectors to a query
- `list_indexes()` → shows all indexes
- `delete_index()` → removes an index

**Design choice**: We don't use a Python SDK because transparency matters. You can
read exactly what's being sent to Endee and why.

---

### `src/ingestion.py` — Document Processor

```python
def chunk_text(text, size=400, overlap=80):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += size - overlap
    return chunks
```

**What it does**: Takes a document, splits it into *overlapping word windows*, embeds
each, and upserts to Endee.

**Why overlap?** If a key sentence falls at the boundary of two chunks, overlap
ensures it's fully captured in at least one of them. 80-word overlap on 400-word
chunks = 20% overlap.

**Why 400 words?** Embedding models have token limits. 400 words ≈ 500 tokens —
well within the 512-token limit of MiniLM while keeping enough context for the
LLM to reason from.

**Metadata stored per chunk**:
```python
{ "text": "...", "source": "report.pdf", "chunk_id": 7 }
```
This is what gets displayed back to the user as "source citations".

---

### `src/retriever.py` — Vector Search

```python
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
```

**What it does**: Converts the user's question to a vector, asks Endee for the 5
most similar stored chunks, and returns them with their scores.

**Score interpretation**: Cosine similarity ranges from -1 to 1. A score of 0.85+
means very high semantic similarity. A score below 0.5 usually means the document
doesn't contain relevant information.

---

### `src/generator.py` — The LLM Brain

```python
messages = [
    {"role": "system", "content": "Answer only from context. If unsure, say so."},
    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
]
response = client.chat.completions.create(model=model, messages=messages, temperature=0.2)
```

**What it does**: Takes the top-k retrieved chunks and the user's question, builds
a prompt, and sends it to Groq's LLaMA3.

**Why temperature=0.2?** Temperature controls randomness. Lower = more deterministic
and factual. For RAG, we want the LLM to stick to what's in the context, not
freestyle. 0.2 is a good balance between fluid language and factual accuracy.

**Why Groq?** Free tier, extremely fast inference (300+ tokens/second), LLaMA3
is strong at instruction following.

---

### `src/rag_pipeline.py` — The Orchestrator

```python
def ask(question, index_name, top_k=5):
    hits = retrieve(question, index_name, top_k=top_k)
    if not hits:
        return {"answer": "No relevant documents found.", "sources": []}
    answer = generate(question, hits)
    return {"answer": answer, "sources": hits}
```

**What it does**: The single public function of the whole pipeline. One call in,
one structured response out. The Streamlit app only ever calls `ask()`.

---

### `app.py` — The User Interface

A Streamlit app with:
- **Sidebar**: file upload → triggers `ingest()` → chunks stored in Endee
- **Chat area**: user types question → `ask()` → answer + source expander rendered
- **"Load sample.txt" button**: ingests the built-in AI/ML doc for instant demo

The UI stores chat history in `st.session_state` (Streamlit's in-memory state
between re-renders) and re-renders on every question.

---

## 6. The Complete Data Flow

```
──── INGESTION (one time per document) ────────────────────────────

PDF/TXT
  │
  ▼ load_text()
Raw text string
  │
  ▼ chunk_text(size=400, overlap=80)
["chunk 0 ...", "chunk 1 ...", "chunk 2 ...", ...]
  │
  ▼ embed(chunks)  [SentenceTransformer]
[[0.23, -0.61, ...], [0.18, -0.55, ...], ...]   ← 384 floats each
  │
  ▼ db.upsert(index_name, payload)  [Endee REST API]
Stored in Endee HNSW index with metadata


──── QUERY (every user question) ───────────────────────────────────

User question: "What is a neural network?"
  │
  ▼ embed_one(question)  [same SentenceTransformer]
[0.21, -0.59, 0.91, ...]   ← 384 floats
  │
  ▼ db.search(index_name, query_vector, top_k=5)  [Endee ANN search]
Top 5 similar chunks with scores & metadata
  │
  ▼ generate(question, hits)  [Groq LLaMA3]
Grounded answer string
  │
  ▼ Streamlit renders answer + source citations
```

---

## 7. Running Endee Locally (No Docker)

### Prerequisites (one-time setup)

```bash
# Confirm you have the tools (should already work on Mac)
xcode-select -p          # should show /Library/Developer/CommandLineTools
cmake --version          # should show cmake 3.x+
brew install cmake       # only if cmake missing
```

### Build Endee

```bash
git clone https://github.com/endee-io/endee.git ~/endee
cd ~/endee
chmod +x install.sh

# Apple Silicon (M1/M2/M3):
./install.sh --release --neon

# Intel Mac:
./install.sh --release --avx2
```

This compiles the C++ source into a native binary. Takes 2-5 minutes.

### Start Endee

```bash
cd ~/endee
chmod +x run.sh
./run.sh
# Endee is now running at http://localhost:8080
```

Verify it's working:
```bash
curl http://localhost:8080/api/v1/index/list
# Should return: {"indexes": []}
```

### Start DocuMind (in a separate terminal)

```bash
cd /Users/rahultiwari/.gemini/antigravity/scratch/documind
source .venv/bin/activate
cp .env.example .env        # then add your GROQ_API_KEY to .env
streamlit run app.py
# Open http://localhost:8501
```

---

## 8. Testing Without the UI (CLI)

```bash
source .venv/bin/activate

# Index the sample document
python3 -c "
from src.ingestion import ingest
count = ingest('samples/sample.txt', 'demo')
print('Indexed', count, 'chunks')
"

# Search Endee directly
python3 -c "
from src.retriever import retrieve
hits = retrieve('What is RAG?', 'demo', top_k=3)
for h in hits:
    print(h['score'], h['text'][:120])
"

# Full RAG answer
python3 -c "
from src.rag_pipeline import ask
result = ask('What is a neural network?', 'demo')
print(result['answer'])
"
```

---

## 9. Why This Architecture?

| Decision | Reason |
|---|---|
| Endee as vector DB | Open-source, self-hosted, HNSW speed, simple REST |
| SentenceTransformers (local) | No API cost, fast, strong semantic quality |
| Overlap chunking | Avoids missing context at chunk boundaries |
| Cosine similarity | Text length invariant, standard for NLP |
| Groq + LLaMA3 | Free, fastest inference available, instruction-tuned |
| Streamlit UI | Fastest path to a working chat interface in Python |
| Pure `requests` for Endee | Transparent, no magic, easy to understand and debug |

---

## 10. Key Concepts Cheat Sheet

| Term | One-line definition |
|---|---|
| Embedding | A list of numbers that captures the "meaning" of text |
| Vector database | A database optimised to find similar vectors fast |
| ANN search | Approximate Nearest Neighbor — fast similarity search |
| HNSW | Graph algorithm for ANN — O(log n) search complexity |
| RAG | Retrieve relevant context → augment LLM prompt → generate |
| Cosine similarity | Similarity based on vector angle (ignores magnitude) |
| Chunk | A piece of a document suitable for embedding |
| top_k | Number of most similar results to return from search |
| Temperature | LLM randomness dial — lower means more factual |
