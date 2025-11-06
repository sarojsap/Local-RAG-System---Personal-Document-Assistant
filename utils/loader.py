from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

# Load and parse documents (PDF, DOCX, etc.) using UnstructuredLoader.
# Returns a list of LangChain Document objects.
def load_documents(file_paths: List[str]) -> List[Document]:

    all_docs = []

    for path in file_paths:
        print(f"Loading: {path}")
        loader = UnstructuredLoader(path)
        docs = loader.load()
        all_docs.extend(docs)

    print(f"Loaded {len(all_docs)} documents.")
    return all_docs

# Split documents into smaller chunks for embedding and retrieval
def split_documents(docs: List[Document], chunk_size:int=1000, chunk_overlap:int=200) -> List[Document]:
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_docs = text_splitter.split_documents(docs)
    print(f"Split into {len(split_docs)} chunks.")
    return split_docs

