# 🧠 DocuMind — RAG Q&A with Endee Vector Database

> **Ask natural-language questions about your documents. Get grounded answers with source citations — powered by Endee, SentenceTransformers, and Groq.**

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![Endee](https://img.shields.io/badge/Vector%20DB-Endee-purple?style=flat-square)
![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA3-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square)
![Docker](https://img.shields.io/badge/Deploy-Docker%20Compose-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Project Overview & Problem Statement

Large Language Models (LLMs) are powerful but have two hard limitations:

| Problem | Impact |
|---|---|
| **Knowledge cutoff** | Models cannot answer about recent or domain-specific information |
| **Hallucinations** | Models confidently generate plausible but factually wrong answers |

**DocuMind** solves both by implementing **Retrieval-Augmented Generation (RAG)**:
1. User uploads a PDF or TXT document
2. The document is chunked and embedded into **Endee**
3. On each question, the most relevant chunks are retrieved via vector similarity search
4. Those chunks are injected as grounded context into the LLM prompt
5. The LLM generates a factual answer — never guessing beyond what the document says

---

## 🏗️ System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                         User (Browser)                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Upload Doc / Ask Question
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit App  (app.py)                       │
│                                                                 │
│  ┌─────────────────┐      ┌──────────────────────────────────┐  │
│  │  Ingestion Flow │      │          RAG Query Flow          │  │
│  │                 │      │                                  │  │
│  │ 1. Load file    │      │ 1. Embed question                │  │
│  │ 2. Chunk text   │      │    (SentenceTransformers)        │  │
│  │ 3. Embed chunks │      │ 2. ANN Search → top-k chunks     │  │
│  │    (SBERT)      │      │    (Endee vector DB)             │  │
│  │ 4. Upsert to    │      │ 3. Build prompt with context     │  │
│  │    Endee        │      │ 4. Generate answer (Groq LLaMA3) │  │
│  └─────────────────┘      └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┴──────────────┐
              ▼                            ▼
┌─────────────────────┐        ┌─────────────────────┐
│     Endee (DB)      │        │    Groq LLM API      │
│  HNSW + cosine sim  │        │  llama3-8b-8192      │
│  384-dim vectors    │        │  Temperature: 0.2    │
│  metadata filters   │        └─────────────────────┘
└─────────────────────┘
```

### Data Flow Summary

```
Document ──► Chunks (400 words, 80 overlap)
                 │
                 ▼
         SentenceTransformer
         (all-MiniLM-L6-v2)
                 │
                 ▼
         384-dim embeddings
                 │
                 ▼
         Endee.upsert()   ←── stored with metadata {text, source, chunk_id}
                 │
            [at query time]
                 │
    Question ──► embed_one(question)
                 │
                 ▼
         Endee.search()  → top-5 chunks
                 │
                 ▼
         Groq Chat API (with context)
                 │
                 ▼
         Answer + Sources
```

---

## 🔍 How Endee Is Used

Endee is the core vector store powering all similarity search in DocuMind.

### 1. Index Creation
A named index is created with 384 dimensions (matching `all-MiniLM-L6-v2`) and cosine similarity:

```python
# src/endee_client.py
db.create_index("documind", dims=384, metric="cosine")
```

### 2. Upserting Vectors
Each document chunk is embedded and stored with rich metadata:

```python
payload = {
    "id": str(uuid.uuid4()),
    "vector": embed(chunk),
    "metadata": {"text": chunk, "source": "report.pdf", "chunk_id": 7}
}
db.upsert("documind", [payload])
```

### 3. Semantic Search
At query time, the question is embedded and the nearest chunks are retrieved:

```python
hits = db.search("documind", embed_one("What is RAG?"), top_k=5)
```

Endee returns results ranked by cosine similarity, with metadata attached — so we get back the actual text, source file, and relevance score together.

### Why Endee?
- **Sub-5ms latency** on millions of vectors via HNSW indexing
- **Open-source and self-hosted** — no cloud lock-in, full data privacy
- **Simple REST API** — no heavy SDK required, just `requests`
- **Docker-friendly** — single command to stand up the database

---

## 📁 Project Structure

```
documind/
├── src/
│   ├── __init__.py
│   ├── embedder.py        # SentenceTransformer wrapper
│   ├── endee_client.py    # Endee REST API client
│   ├── ingestion.py       # Document chunking + indexing
│   ├── retriever.py       # ANN search wrapper
│   ├── generator.py       # Groq LLM answer generation
│   └── rag_pipeline.py    # End-to-end RAG orchestration
├── samples/
│   └── sample.txt         # Built-in demo document
├── app.py                 # Streamlit chat UI
├── docker-compose.yml     # Endee + App services
├── Dockerfile             # App container
├── requirements.txt
└── .env.example
```

---

## ⚙️ Setup & Execution

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Free [Groq API key](https://console.groq.com)

### Step 1 — Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/documind.git
cd documind
```

### Step 2 — Configure environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Step 3 — Option A: Full Docker Compose (recommended)

```bash
docker compose up --build
```

Open **http://localhost:8501** in your browser.

### Step 3 — Option B: Local Python + Docker Endee

```bash
# Start Endee only
docker compose up endee -d

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run Streamlit
streamlit run app.py
```

### Step 4 — Try it out

1. Click **"📄 Load sample.txt"** in the sidebar to index the built-in AI/ML document
2. Type a question like: *"What is Retrieval-Augmented Generation?"*
3. Get a grounded answer with source citations

---

## 🧪 Testing the Pipeline via CLI

```bash
# Ingest a document
python3 -c "
from src.ingestion import ingest
count = ingest('samples/sample.txt', 'demo')
print(f'Indexed {count} chunks')
"

# Retrieval test
python3 -c "
from src.retriever import retrieve
hits = retrieve('What is a neural network?', 'demo', top_k=3)
for h in hits:
    print(h['score'], h['text'][:100])
"

# End-to-end RAG
python3 -c "
from src.rag_pipeline import ask
result = ask('What is a neural network?', 'demo')
print(result['answer'])
"
```

---

## 🤝 Contributing

Pull requests are welcome! Please open an issue first to discuss significant changes.

---

## 📄 License

MIT © 2024 DocuMind Contributors
