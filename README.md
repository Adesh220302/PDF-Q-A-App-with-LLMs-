# 📄 PDF Question Answering System using LLaMA3, FAISS & LangChain

This project is a **Streamlit-based web app** that allows users to upload a PDF file and ask **natural language questions** about its content. It uses **LangChain**, **FAISS**, **HuggingFace Embeddings**, and **Ollama’s LLaMA3 model** to provide contextual answers with references.

---

## 🚀 Features

- ✅ Upload any PDF and extract meaningful information
- 🤖 Ask questions and get LLM-generated responses based on your PDF
- 📚 Answers are grounded in real context (no hallucinations)
- 🧠 Uses vector similarity search with FAISS
- 🧾 Displays source documents for transparency
- 💬 Maintains history of your Q&A session

---

## 🧰 Tech Stack

- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Ollama (LLaMA3)](https://ollama.com/)
- [HuggingFace Sentence Transformers](https://www.sbert.net/)

---

## 📁 Project Structure


.
├── app.py               # Streamlit front-end app
├── backend.py           # Core PDF processing & LLM logic
├── vectorstore/         # Stores FAISS index locally
└── app.log              # Logs for debugging and status
⚙️ Installation
1. Clone the repository
git clone https://github.com/your-username/pdf-qa-app.git
cd pdf-qa-app

2. Create & activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Start Ollama with LLaMA3
Make sure you have Ollama installed and the LLaMA3 model pulled:
ollama run llama3

5. Run the app
streamlit run app.py

🧪 How It Works
PDF Upload: You upload a PDF document.

Text Extraction & Chunking: Text is split into chunks using RecursiveCharacterTextSplitter.

Embedding Creation: Chunks are embedded using sentence-transformers/all-MiniLM-L6-v2.

FAISS Indexing: Embeddings are saved locally using FAISS for retrieval.

Question Answering: You enter a question. The most relevant chunks are retrieved and passed to the LLaMA3 model.

Answer + Sources: The model generates a grounded answer and shows source text from the document.

🔐 Environment Setup (Optional)
If you need to set model paths or API keys (e.g., for other LLMs), consider adding a .env file.

📷 Preview

📝 License
This project is open-source under the MIT License.

🙌 Acknowledgements
LangChain

Ollama

HuggingFace Transformers

Streamlit

# output look like 

<img width="959" height="884" alt="image" src="https://github.com/user-attachments/assets/fb0db6c3-0080-45c0-bb54-070bf86fb120" />
