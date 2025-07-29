# app.py

import streamlit as st
import tempfile
import logging
from backend import (
    load_and_split_pdf,
    create_vector_store,
    load_vector_store,
    get_llm_model,
    get_qa_chain
)

# -------- LOGGING SETUP --------
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

st.set_page_config(page_title="PDF Q&A System", layout="centered")
st.title("📄 Ask Questions from a PDF")

# -------- SESSION STATE --------
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "history" not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    with st.spinner("🔄 Processing PDF..."):
        try:
            chunks = load_and_split_pdf(pdf_path)
            create_vector_store(chunks)
            db = load_vector_store()
            llm = get_llm_model("llama3")
            st.session_state.qa_chain = get_qa_chain(llm, db)
            st.success("✅ PDF successfully processed!")
            logging.info("PDF processed and vector store created.")
        except Exception as e:
            st.error("❌ Failed to process PDF.")
            logging.error(f"Error during PDF processing: {e}")

if st.session_state.qa_chain:
    user_query = st.text_input("💬 Enter your question here:")
    if user_query:
        with st.spinner("🧠 Generating answer..."):
            try:
                response = st.session_state.qa_chain.invoke({"query": user_query})
                answer = response["result"]
                sources = response["source_documents"]

                # Store in session history
                st.session_state.history.append({
                    "question": user_query,
                    "answer": answer,
                    "sources": sources
                })

                logging.info(f"Answered question: {user_query}")
                st.markdown("### ✅ Answer:")
                st.write(answer)

                with st.expander("📚 Source Documents"):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Source {i+1}:**\n{doc.page_content[:300]}...\n")

            except Exception as e:
                st.error("❌ Error generating answer.")
                logging.error(f"Answering failed: {e}")

# -------- HISTORY SECTION --------
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🕘 Question & Answer History")
    for idx, entry in enumerate(reversed(st.session_state.history)):
        st.markdown(f"**Q{len(st.session_state.history) - idx}:** {entry['question']}")
        st.markdown(f"**A:** {entry['answer']}")
        st.markdown("---")
