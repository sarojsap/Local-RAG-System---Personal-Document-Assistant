import streamlit as st
from utils.qa_chain import get_qa_chain
from utils.vector_store import create_vector_store, load_vector_store
import tempfile
import os

st.set_page_config(page_title="SIRUS - Personal Bot", layout="centered")
st.title("SIRUS (LangChain + Ollama)")

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
        for file in uploaded_files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
            file_paths.append(file_path)
        
        from utils.loader import load_documents, split_documents
        docs = split_documents(load_documents(file_paths))
        
        create_vector_store(docs, persist_directory=PERSIST_DIR)
        # Refresh QA chain so new docs are available immediately
        st.session_state.qa_chain = get_qa_chain(persist_directory=PERSIST_DIR, model_name='mistral')
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
    for q, a in st.session_state.history:
        st.markdown(f"You: {q}")
        st.markdown(f"Sirus: {a}")
        st.markdown("---")