import streamlit as st
from utils.qa_chain import get_qa_chain
from utils.vector_store import create_vector_store, load_vector_store
import tempfile
import os

st.set_page_config(page_title="SIRUS - Personal Bot", page_icon="🤖", layout="wide")

# Minimal professional styling
st.markdown(
    """
    <style>
      .sirus-header { font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem; }
      .sirus-sub { color: #6b7280; margin-bottom: 1rem; }
      .msg { padding: 0.9rem 1rem; border-radius: 12px; margin-bottom: 0.5rem; color: #111827; line-height: 1.5; white-space: pre-wrap; }
      .msg strong { color: #111827; }
      .msg a { color: #2563eb; text-decoration: none; }
      .msg a:hover { text-decoration: underline; }
      .msg-user { background: #e0e7ff; border: 1px solid #c7d2fe; }
      .msg-bot { background: #e5e7eb; border: 1px solid #cbd5e1; }

      /* Dark theme overrides */
      @media (prefers-color-scheme: dark) {
        .sirus-sub { color: #9ca3af; }
        .msg { color: #e5e7eb; }
        .msg strong { color: #e5e7eb; }
        .msg a { color: #93c5fd; }
        .msg-user { background: #1e293b; border: 1px solid #334155; }
        .msg-bot { background: #0f172a; border: 1px solid #1f2937; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="sirus-header">SIRUS (LangChain + Ollama)</div>', unsafe_allow_html=True)
st.markdown('<div class="sirus-sub">Private RAG assistant for your PDFs and docs</div>', unsafe_allow_html=True)

# Sidebar for uploadinfg documents
st.sidebar.header("📄 Upload Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload Documents (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

# Directory for persistent Chroma Storage
PERSIST_DIR = "chroma_store"

# Button to process uploaded files
if st.sidebar.button("Process Documents"):
    if uploaded_files:
        temp_dir = tempfile.mkdtemp()
        file_paths = []

        # Progress bar for processing pipeline
        total_steps = max(1, len(uploaded_files)) + 2  # write files + load/split + index
        step = 0
        progress = st.sidebar.progress(0, text="Preparing...")

        # Write uploaded files to temp
        for file in uploaded_files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
            file_paths.append(file_path)
            step += 1
            progress.progress(min(100, int(step / total_steps * 100)), text=f"Saved {file.name}")
        
        from utils.loader import load_documents, split_documents
        progress.progress(min(100, int((step + 0.5) / total_steps * 100)), text="Loading documents...")
        docs = split_documents(load_documents(file_paths))
        step += 1
        progress.progress(min(100, int(step / total_steps * 100)), text="Splitting into chunks...")
        
        create_vector_store(docs, persist_directory=PERSIST_DIR)
        # Refresh QA chain so new docs are available immediately
        st.session_state.qa_chain = get_qa_chain(persist_directory=PERSIST_DIR, model_name='mistral')
        progress.progress(100, text="Indexing complete ✅")
        st.sidebar.success("Documents Processed and stored! ✅")
    else:
        st.sidebar.warning("⚠️ Please upload at least one file.")

# Load QA Chain
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = get_qa_chain(persist_directory=PERSIST_DIR, model_name='mistral')

# Chat interface
st.subheader("Ask Your Question")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input('Enter Your Question:')

if st.button("Ask"):
    if query.strip():
        with st.spinner("Thinking..."):
            # First, check retrieval to avoid hallucinations
            vectordb = load_vector_store(persist_directory=PERSIST_DIR)
            top_k = 3
            search_results = vectordb.similarity_search_with_score(query, k=top_k)
            if not search_results:
                st.warning("No relevant context found. Please upload/process documents containing the answer.")
            else:
                # Simple guard: ensure at least one result is confidently similar
                # Note: lower scores are more similar for cosine distance
                best_score = search_results[0][1]
                if best_score > 0.5:
                    st.warning("Low-confidence retrieval. The documents may not contain the answer.")
                result = st.session_state.qa_chain({"query": query})
                answer = result["result"]
                st.session_state.history.append((query, answer))

                # Show retrieved sources and scores
                st.markdown("**Top sources:**")
                for i, (doc, score) in enumerate(search_results, start=1):
                    source = doc.metadata.get("source") or doc.metadata.get("file_path") or doc.metadata.get("filename") or "unknown"
                    page = doc.metadata.get("page")
                    page_str = f", page {page}" if page is not None else ""
                    st.markdown(f"{i}. {source}{page_str} — score: {score:.3f}")

    else:
        st.warning("Please enter a question.")

if st.session_state.history:
    st.subheader('Chat History')
    # Show latest first
    for q, a in reversed(st.session_state.history):
        st.markdown(f"<div class='msg msg-user'><strong>You</strong><br>{q}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='msg msg-bot'><strong>Sirus</strong><br>{a}</div>", unsafe_allow_html=True)