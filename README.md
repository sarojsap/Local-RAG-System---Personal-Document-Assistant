# 🧠 Local RAG System (Retrieval-Augmented Generation)

A **fully local** Retrieval-Augmented Generation (RAG) pipeline built using **LangChain**, **Chroma**, **Ollama**, and **HuggingFace embeddings**  all running **100% offline**.  
Upload your documents, process them into embeddings, and chat with your data privately without using any external API or cloud service.

---

## 🚀 Features

- ⚙️ **End-to-End Local RAG** — No API keys, no internet required.  
- 📄 **Supports multiple file types** — PDFs, DOCX, and text files.  
- 💬 **Chat with your documents** — Query and retrieve contextually relevant answers.  
- 💾 **Persistent local vector storage** with Chroma DB.  
- 🧩 **Lightweight & Fast** — Uses MiniLM embeddings and Mistral via Ollama.  
- 🔒 **Privacy-first** — All data and computation happen locally.  

---

## 🧰 Tech Stack

| Component | Technology Used |
|------------|----------------|
| Framework | [LangChain](https://python.langchain.com/) |
| Vector Store | [Chroma](https://www.trychroma.com/) |
| Embeddings | [HuggingFace MiniLM (all-MiniLM-L6-v2)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| LLM | [Ollama (Mistral model)](https://ollama.ai/library/mistral) |
| UI | [Streamlit](https://streamlit.io/) |
| Language | Python 3.10+ |

---

## ⚡ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/LangChain-RAG.git
cd LangChain-RAG
```
### 2️⃣ Create a Virtual Environment
```bash
python -m venv env
venv\Scripts\activate      # On Windows
# source venv/bin/activate # On Mac/Linux
```
### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Install Ollama (if not already)
Download from https://ollama.ai/download
Then pull the Mistral model:
```bash
ollama pull mistral
```
### 5️⃣ Run the App
```bash
streamlit run app.py
```
---

## 🧠 How It Works

**1. Upload Documents**→ Load and split documents into smaller chunks.

**2. Embed Text** → Generate vector embeddings using HuggingFace MiniLM.

**3. Store in Chroma DB** → Store embeddings locally for retrieval.

**4. Ask Questions** → Query is embedded, matched with stored chunks, and passed to the LLM.

**5. LLM Generates Response** → Mistral via Ollama produces the final contextual answer.

---

## 📸 Demo Preview
<img width="900" height="1183" alt="Screenshot 2025-11-06 205434" src="https://github.com/user-attachments/assets/40547f05-e016-435b-86e8-3105dbad2ba2" />
<img width="1661" height="1022" alt="Screenshot 2025-11-06 205447" src="https://github.com/user-attachments/assets/e96bd5a9-d0c0-4e7a-ad5f-56328bdb99a3" />
<img width="1633" height="1104" alt="Screenshot 2025-11-06 205533" src="https://github.com/user-attachments/assets/b6567e1f-57b8-4146-8fd1-7280196c6a46" />

