from langchain_classic.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM


# Create a RetrievalQA chain that uses the Chroma vector store for retrieval
def get_qa_chain(persist_directory: str='chroma_store', model_name:str='mistral'):
    # Load local embeddings
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    # Load vector store
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever(search_kwargs={"k":3})

    # Load local LLM
    llm = OllamaLLM(model=model_name)

    # Grounded prompt to avoid using outside knowledge
    grounded_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful assistant. Use ONLY the following context to answer. "
            "If the answer is not contained in the context, say you don't know.\n\n"
            "But you can use your knowledge to answer the question if it's not contained in the context."
            "You can greet the user to make them feel welcome and friendly."
            "Context:\n{context}\n\nQuestion: {question}"
        ),
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type='stuff',
        chain_type_kwargs={"prompt": grounded_prompt},
        return_source_documents= True
    )

    print("✅ Fully local QA chain created!")
    return qa_chain


