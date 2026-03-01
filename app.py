import sys
import os
import tempfile
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingestion import ingest
from src.rag_pipeline import ask, index_summary
from src import endee_client as db

st.set_page_config(
    page_title="DocuMind — RAG with Endee",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e2e8f0;
        min-height: 100vh;
    }
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    .hero { text-align: center; padding: 2rem 0 1rem; }
    .hero h1 {
        font-size: 2.8rem; font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero p { color: #94a3b8; font-size: 1.05rem; }

    .chat-user {
        background: linear-gradient(135deg, #6d28d9, #4f46e5);
        border-radius: 18px 18px 4px 18px; padding: 14px 18px;
        margin: 8px 0; max-width: 80%; margin-left: auto;
        color: #fff; box-shadow: 0 4px 15px rgba(109,40,217,0.35);
    }
    .chat-ai {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px 18px 18px 4px; padding: 14px 18px;
        margin: 8px 0; max-width: 85%; color: #e2e8f0;
        backdrop-filter: blur(8px);
    }
    .source-card {
        background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.3);
        border-radius: 10px; padding: 10px 14px; margin: 5px 0;
        font-size: 0.84rem; color: #a5b4fc;
    }
    .score-bar-wrap { background: rgba(255,255,255,0.07); border-radius: 8px; height: 6px; margin: 5px 0; }
    .score-bar { height: 6px; border-radius: 8px;
        background: linear-gradient(90deg, #6d28d9, #34d399); }
    .badge {
        display: inline-block; background: rgba(52,211,153,0.13);
        color: #34d399; border: 1px solid rgba(52,211,153,0.3);
        border-radius: 20px; padding: 2px 10px; font-size: 0.74rem; font-weight: 600;
    }
    .warn-badge {
        background: rgba(251,191,36,0.13); color: #fbbf24;
        border: 1px solid rgba(251,191,36,0.3);
    }
    .stat-box {
        background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.09);
        border-radius: 12px; padding: 12px; text-align: center;
    }
    .stat-num { font-size: 1.5rem; font-weight: 700; color: #a78bfa; }
    .stat-label { font-size: 0.76rem; color: #64748b; }
    .stButton > button {
        background: linear-gradient(135deg, #6d28d9, #4f46e5); color: white;
        border: none; border-radius: 10px; font-weight: 600; width: 100%;
        transition: all 0.2s ease;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(109,40,217,0.4); }
    div[data-testid="stChatInput"] textarea {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important; color: #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_index" not in st.session_state:
    st.session_state.active_index = None
if "ingested_count" not in st.session_state:
    st.session_state.ingested_count = 0
if "known_sources" not in st.session_state:
    st.session_state.known_sources = []

with st.sidebar:
    st.markdown("## 🗄️ Knowledge Base")
    st.markdown("---")

    index_name = st.text_input("Index name", value="documind", placeholder="my-index")

    st.markdown("**Endee Precision (quantization)**")
    precision = st.selectbox(
        "Vector precision",
        options=["FLOAT32", "FLOAT16", "INT16", "INT8", "BINARY"],
        index=0,
        help="FLOAT32 = full accuracy. BINARY = 32x smaller, fastest search. INT8 = great for large corpora."
    )

    metric = st.selectbox(
        "Distance metric",
        options=["cosine", "L2", "inner_product"],
        index=0,
        help="cosine = best for text. L2 = Euclidean distance. inner_product = for normalized vectors."
    )

    uploaded = st.file_uploader("Upload document", type=["pdf", "txt"])

    if st.button("⚡ Ingest Document", disabled=uploaded is None):
        with st.spinner("Chunking and indexing into Endee…"):
            suffix = ".pdf" if uploaded.name.endswith(".pdf") else ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            count = ingest(tmp_path, index_name, metric=metric, precision=precision)
            st.session_state.active_index = index_name
            st.session_state.ingested_count += count
            if uploaded.name not in st.session_state.known_sources:
                st.session_state.known_sources.append(uploaded.name)
            st.success(f"✅ {count} chunks → Endee (`{precision}`, `{metric}`)")

    st.markdown("---")
    st.markdown("**Quick start sample:**")
    sample_precision = st.selectbox("Sample precision", ["FLOAT32", "FLOAT16", "INT8"], key="sp")
    if st.button("📄 Load sample.txt"):
        with st.spinner("Indexing into Endee…"):
            sample_path = os.path.join(os.path.dirname(__file__), "samples", "sample.txt")
            count = ingest(sample_path, index_name, metric=metric, precision=sample_precision)
            st.session_state.active_index = index_name
            st.session_state.ingested_count += count
            if "sample.txt" not in st.session_state.known_sources:
                st.session_state.known_sources.append("sample.txt")
            st.success(f"✅ {count} chunks indexed")

    st.markdown("---")
    st.markdown("### 🔌 Endee Index Status")

    if st.session_state.active_index:
        st.markdown(f"<span class='badge'>✓ {st.session_state.active_index}</span>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div class='stat-box'><div class='stat-num'>{st.session_state.ingested_count}</div><div class='stat-label'>chunks</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='stat-box'><div class='stat-num'>{len(st.session_state.known_sources)}</div><div class='stat-label'>docs</div></div>", unsafe_allow_html=True)

        try:
            all_indexes = db.list_indexes()
            names = [i.get("name", "") for i in all_indexes.get("indexes", [])]
            if names:
                st.markdown(f"<small style='color:#64748b'>All indexes: {', '.join(names)}</small>", unsafe_allow_html=True)
        except Exception:
            pass

        if st.button("🗑️ Delete Index"):
            try:
                db.delete_index(st.session_state.active_index)
                st.session_state.active_index = None
                st.session_state.ingested_count = 0
                st.session_state.known_sources = []
                st.session_state.messages = []
                st.rerun()
            except Exception as e:
                st.error(str(e))
    else:
        st.info("No index loaded yet.")

    st.markdown("---")
    st.markdown("**Search options**")
    top_k = st.slider("Top-K results from Endee", 1, 10, 5)
    source_filter = None
    if st.session_state.known_sources:
        filter_choice = st.selectbox(
            "Filter by source (Endee $eq)",
            options=["All sources"] + st.session_state.known_sources
        )
        if filter_choice != "All sources":
            source_filter = filter_choice

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("<small style='color:#475569'>Powered by Endee · Groq · SentenceTransformers</small>", unsafe_allow_html=True)

st.markdown("<div class='hero'><h1>🧠 DocuMind</h1><p>Grounded answers from your documents — powered by Endee vector search</p></div>", unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div style='text-align:center; color:#475569; padding: 3rem 0;'>
        <div style='font-size:3rem'>📂</div>
        <p>Upload a document in the sidebar and start asking questions.</p>
        <p style='font-size:0.85rem'>Endee uses HNSW indexing for sub-5ms similarity search.</p>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-user'>🙋 {msg['content']}</div>", unsafe_allow_html=True)
    else:
        confidence = msg.get("top_score", 0)
        badge_class = "badge" if confidence >= 0.6 else "badge warn-badge"
        label = "high confidence" if confidence >= 0.6 else "low confidence"
        st.markdown(
            f"<div class='chat-ai'>🤖 {msg['content']}<br><br>"
            f"<span class='{badge_class}'>Endee score: {confidence:.2f} · {label}</span></div>",
            unsafe_allow_html=True
        )
        if msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} Endee results", expanded=False):
                for src in msg["sources"]:
                    bar_width = int(src['score'] * 100)
                    st.markdown(
                        f"<div class='source-card'>"
                        f"<strong>{src['source']}</strong> · chunk #{src['chunk_id']} · {src.get('word_count', '?')} words"
                        f"<div class='score-bar-wrap'><div class='score-bar' style='width:{bar_width}%'></div></div>"
                        f"<small>cosine similarity: <code>{src['score']}</code></small>"
                        f"<br><small style='color:#94a3b8'>{src['text'][:220]}…</small>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

question = st.chat_input("Ask a question about your document…")

if question:
    if not st.session_state.active_index:
        st.warning("⚠️ Please ingest a document first using the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner(f"Searching Endee (top-{top_k}, {precision})…"):
            result = ask(
                question,
                st.session_state.active_index,
                top_k=top_k,
                source_filter=source_filter
            )
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "top_score": result["top_score"]
        })
        st.rerun()
