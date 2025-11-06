from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List

# Create & save a Chroma Vector Store from a list of LangChain documents.
def create_vector_store(docs: List[Document], persist_directory: str='chroma_store'):

    # filter out non-supported metadata types
    docs = filter_complex_metadata(docs)

    # HuggingFaceEmbeddings (a local model)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create a vector store
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    # Ensure on-disk persistence
    # vectordb.persist()
    print(f"Vector store created and saved in, {persist_directory} ")

    return vectordb

# Load an existing Chroma vector store from disk
def load_vector_store(persist_directory: str="chroma_store"):
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    print('Vector Store loaded from disk')
    return vectordb