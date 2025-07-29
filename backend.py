# backend.py

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Disable TensorFlow warning
os.environ["TRANSFORMERS_NO_TF"] = "1"

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say you don't know. Don't try to make up an answer.
Only use the provided context.

Context: {context}
Question: {question}

Start the answer directly. No small talk.
"""

def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

def create_vector_store(chunks, db_path="vectorstore/db_faiss"):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embedding)
    db.save_local(db_path)
    return db

def load_vector_store(db_path="vectorstore/db_faiss"):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(db_path, embedding, allow_dangerous_deserialization=True)

def get_llm_model(key="llama3"):
    return OllamaLLM(model=key)

def set_custom_prompt(template=CUSTOM_PROMPT_TEMPLATE):
    return PromptTemplate(template=template, input_variables=["context", "question"])

def get_qa_chain(llm, db):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": set_custom_prompt()}
    )
