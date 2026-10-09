import os
import streamlit as st
import requests

# Auto-configured: Docker uses 'backend' hostname, local uses 'localhost'
BASE_URL = os.environ.get("API_URL", "http://localhost:8000/query").replace("/query", "")
QUERY_URL = f"{BASE_URL}/query"
UPLOAD_URL = f"{BASE_URL}/upload"

st.set_page_config(
    page_title="Production RAG System",
    page_icon="🤖",
    layout="wide"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2.4rem; font-weight: 800; }
    .upload-box {
        border: 2px dashed #4A90D9;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        background: rgba(74, 144, 217, 0.05);
        margin-bottom: 1rem;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-green { background: #1a472a; color: #69db7c; }
    .badge-yellow { background: #3b2f00; color: #ffd43b; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.code(QUERY_URL, language=None)
    st.caption("Backend endpoint (auto-configured)")
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. 📄 Upload a PDF above")
    st.markdown("2. ⏳ Wait for indexing to finish")
    st.markdown("3. 💬 Ask questions below")
    st.markdown("---")
    st.markdown("**Features:**")
    st.markdown("✅ Semantic Chunking")
    st.markdown("✅ Hybrid Search (Vector + BM25)")
    st.markdown("✅ Guardrails (PII protection)")
    st.markdown("✅ LangSmith Tracing")

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 📚 Domain-Specific QA System")
st.markdown("Upload any PDF and instantly chat with its contents using AI.")
st.markdown("---")

# ─── Section 1: Upload ─────────────────────────────────────────────────────────
st.subheader("📤 Step 1: Upload a PDF")

# Track currently indexed file in session state
if "indexed_file" not in st.session_state:
    st.session_state.indexed_file = None

with st.container():
    uploaded_file = st.file_uploader(
        "Drag & drop or click to browse",
        type=["pdf"],
        help="Only PDF files are supported."
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"📄 **{uploaded_file.name}** ({round(uploaded_file.size / 1024, 1)} KB)")
        with col2:
            index_btn = st.button("🔄 Index this PDF", use_container_width=True)

        if index_btn:
            with st.spinner(f"Indexing **{uploaded_file.name}**... This may take 30–60 seconds."):
                try:
                    response = requests.post(
                        UPLOAD_URL,
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                        timeout=300  # 5 min timeout for large PDFs
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.indexed_file = uploaded_file.name
                        st.success(f"✅ {data['message']}  |  **{data['num_chunks']} chunks** created")
                    else:
                        st.error(f"❌ Upload failed: {response.json().get('detail', response.text)}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend. Make sure Docker is running.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# Show currently active document
if st.session_state.indexed_file:
    st.markdown(
        f'<span class="status-badge badge-green">✅ Active document: {st.session_state.indexed_file}</span>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<span class="status-badge badge-yellow">⚠️ No document indexed yet — upload one above</span>',
        unsafe_allow_html=True
    )

st.markdown("---")

# ─── Section 2: Query ──────────────────────────────────────────────────────────
st.subheader("💬 Step 2: Ask a Question")

question = st.text_input(
    "Type your question about the uploaded document:",
    placeholder="e.g. What are the key skills mentioned in this resume?"
)

if st.button("🔍 Submit", use_container_width=False):
    if not question:
        st.warning("Please enter a question first.")
    elif not st.session_state.indexed_file:
        st.warning("Please upload and index a PDF first (Step 1).")
    else:
        with st.spinner("Searching and generating answer..."):
            try:
                response = requests.post(
                    QUERY_URL,
                    json={"query": question},
                    timeout=60
                )
                if response.status_code == 200:
                    data = response.json()

                    st.subheader("🤖 Answer")
                    st.success(data["answer"])

                    if data["context"]:
                        with st.expander("📖 View Retrieved Context Chunks"):
                            for i, ctx in enumerate(data["context"]):
                                st.markdown(f"**Chunk {i + 1}:**")
                                st.info(ctx)
                else:
                    err = response.json().get("detail", response.text)
                    st.error(f"❌ Error from backend: {err}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend at `" + QUERY_URL + "`. Is Docker running?")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
