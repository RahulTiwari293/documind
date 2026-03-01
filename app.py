import sys
import os
import tempfile
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingestion import ingest
from src.rag_pipeline import ask
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

    .hero {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-top: 0;
    }

    .chat-bubble-user {
        background: linear-gradient(135deg, #6d28d9, #4f46e5);
        border-radius: 18px 18px 4px 18px;
        padding: 14px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        color: #fff;
        font-size: 0.97rem;
        box-shadow: 0 4px 15px rgba(109,40,217,0.3);
    }
    .chat-bubble-ai {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px 18px 18px 4px;
        padding: 14px 18px;
        margin: 8px 0;
        max-width: 85%;
        color: #e2e8f0;
        font-size: 0.97rem;
        backdrop-filter: blur(8px);
    }

    .source-card {
        background: rgba(99,102,241,0.12);
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 10px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.85rem;
        color: #a5b4fc;
    }

    .badge {
        display: inline-block;
        background: rgba(52,211,153,0.15);
        color: #34d399;
        border: 1px solid rgba(52,211,153,0.3);
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }

    div[data-testid="stChatInput"] textarea {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #6d28d9, #4f46e5);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 8px 20px;
        font-weight: 600;
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(109,40,217,0.4);
    }

    .stTextInput > div > input, .stSelectbox > div {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }

    .stat-box {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 12px;
        text-align: center;
    }
    .stat-num { font-size: 1.6rem; font-weight: 700; color: #a78bfa; }
    .stat-label { font-size: 0.8rem; color: #64748b; }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_index" not in st.session_state:
    st.session_state.active_index = None
if "ingested_count" not in st.session_state:
    st.session_state.ingested_count = 0

with st.sidebar:
    st.markdown("## 🗄️ Knowledge Base")
    st.markdown("---")

    index_name = st.text_input("Index name", value="documind", placeholder="my-index")

    uploaded = st.file_uploader("Upload document", type=["pdf", "txt"])

    if st.button("⚡ Ingest Document", disabled=uploaded is None):
        with st.spinner("Chunking and indexing…"):
            suffix = ".pdf" if uploaded.name.endswith(".pdf") else ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            count = ingest(tmp_path, index_name)
            st.session_state.active_index = index_name
            st.session_state.ingested_count += count
            st.success(f"✅ Indexed {count} chunks from **{uploaded.name}**")

    st.markdown("---")
    st.markdown("### 🔌 Active Index")
    if st.session_state.active_index:
        st.markdown(f"<span class='badge'>✓ {st.session_state.active_index}</span>", unsafe_allow_html=True)
        st.markdown(f"<div class='stat-box'><div class='stat-num'>{st.session_state.ingested_count}</div><div class='stat-label'>chunks indexed</div></div>", unsafe_allow_html=True)
    else:
        st.info("No index loaded yet.")

    st.markdown("---")
    st.markdown("**Try the sample doc:**")
    if st.button("📄 Load sample.txt"):
        with st.spinner("Indexing sample document…"):
            sample_path = os.path.join(os.path.dirname(__file__), "samples", "sample.txt")
            count = ingest(sample_path, index_name)
            st.session_state.active_index = index_name
            st.session_state.ingested_count += count
            st.success(f"✅ Indexed {count} chunks from sample.txt")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("<small style='color:#475569'>Powered by Endee + Groq + SentenceTransformers</small>", unsafe_allow_html=True)

st.markdown("<div class='hero'><h1>🧠 DocuMind</h1><p>Ask anything about your documents — grounded answers, instant retrieval</p></div>", unsafe_allow_html=True)

chat_container = st.container()
with chat_container:
    if not st.session_state.messages:
        st.markdown("""
        <div style='text-align:center; color:#475569; padding: 3rem 0;'>
            <div style='font-size:3rem'>📂</div>
            <p>Upload a document in the sidebar and start asking questions.</p>
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-bubble-user'>🙋 {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble-ai'>🤖 {msg['content']}</div>", unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander(f"📎 {len(msg['sources'])} source chunks", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(
                            f"<div class='source-card'>"
                            f"<strong>{src['source']}</strong> · chunk #{src['chunk_id']} · score <code>{src['score']}</code>"
                            f"<br><small>{src['text'][:200]}…</small>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

question = st.chat_input("Ask a question about your document…")

if question:
    if not st.session_state.active_index:
        st.warning("⚠️ Please ingest a document first using the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("Searching vectors and generating answer…"):
            result = ask(question, st.session_state.active_index)

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        })
        st.rerun()
