import streamlit as st
import requests

# ── Config ─────────────────────────────────────────────────────────────────
BACKEND_URL = "http://localhost:8001"  # change this after deploying backend

# ── Page Setup ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediBot",
    page_icon="🏥",
    layout="wide"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1a73e8, #0d47a1);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .citation-box {
        background-color: #f0f4ff;
        border-left: 4px solid #1a73e8;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    .status-ok {
        color: #2e7d32;
        font-weight: bold;
    }
    .status-err {
        color: #c62828;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏥 MediBot</h1>
    <p>Ask questions about your medical documents</p>
</div>
""", unsafe_allow_html=True)

# ── Session State ───────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "backend_ready" not in st.session_state:
    st.session_state.backend_ready = False

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Document Manager")

    # Health check
    if st.button("🔌 Check Backend Status"):
        try:
            res = requests.get(f"{BACKEND_URL}/health", timeout=5)
            data = res.json()
            if data["vector_store_loaded"]:
                st.markdown('<p class="status-ok">✅ Backend ready!</p>',
                           unsafe_allow_html=True)
                st.session_state.backend_ready = True
            else:
                st.markdown('<p class="status-err">⚠️ Backend up but no docs loaded</p>',
                           unsafe_allow_html=True)
        except Exception:
            st.markdown('<p class="status-err">❌ Backend not reachable</p>',
                       unsafe_allow_html=True)

    st.divider()

    # File uploader
    st.subheader("📤 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("🚀 Upload & Process"):
        with st.spinner("Uploading and ingesting PDFs..."):
            try:
                files = [
                    ("files", (f.name, f.getvalue(), "application/pdf"))
                    for f in uploaded_files
                ]
                res = requests.post(
                    f"{BACKEND_URL}/upload",
                    files=files,
                    timeout=120
                )
                if res.status_code == 200:
                    data = res.json()
                    st.success(data["message"])
                    st.session_state.backend_ready = True
                    # Show uploaded files
                    for fname in data.get("files", []):
                        st.write(f"📄 {fname}")
                else:
                    st.error(f"Upload failed: {res.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Built with LangChain + FAISS + Euri API")

# ── Chat Area ───────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Chat")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            # Show citations if present
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📚 Sources"):
                    for src in msg["sources"]:
                        st.markdown(
                            f'<div class="citation-box">📄 <b>{src["file"]}</b> — Page {src["page"]}</div>',
                            unsafe_allow_html=True
                        )

    # Chat input
    if question := st.chat_input("Ask a question about your medical documents..."):
        # Check backend
        if not st.session_state.backend_ready:
            st.warning("⚠️ Please check backend status first using the sidebar button.")
        else:
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": question
            })
            with st.chat_message("user"):
                st.write(question)

            # Get answer from backend
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        res = requests.post(
                            f"{BACKEND_URL}/chat",
                            json={"question": question},
                            timeout=60
                        )
                        if res.status_code == 200:
                            data = res.json()
                            answer = data["answer"]
                            sources = data.get("sources", [])

                            st.write(answer)

                            # Show citations
                            if sources:
                                with st.expander("📚 Sources"):
                                    for src in sources:
                                        st.markdown(
                                            f'<div class="citation-box">📄 <b>{src["file"]}</b> — Page {src["page"]}</div>',
                                            unsafe_allow_html=True
                                        )

                            # Save to history
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "sources": sources
                            })
                        else:
                            err = res.json().get("detail", res.text)
                            st.error(f"Error: {err}")

                    except requests.exceptions.Timeout:
                        st.error("Request timed out. The backend may be processing a large document.")
                    except Exception as e:
                        st.error(f"Could not reach backend: {str(e)}")

with col2:
    st.subheader("ℹ️ How to Use")
    st.info("""
    **Steps:**
    1. Click **Check Backend Status**
    2. Upload your PDF files
    3. Click **Upload & Process**
    4. Ask questions in the chat!

    **Tips:**
    - Be specific in your questions
    - Ask about specific patients or conditions
    - Citations show which document was used
    """)

    st.subheader("📊 Session Info")
    st.metric("Messages", len(st.session_state.messages))
    st.metric("Backend Ready", "✅ Yes" if st.session_state.backend_ready else "❌ No")