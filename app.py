import streamlit as st
import time
from app.PDF_loader import pdf_bytes_to_documents
from app.splitter import split_documents
from app.vectorstore_DB import create_inmemory_vectorstore
from app.RAG_system_pipeline import build_rag_chain

# --- UI Setup ---
st.set_page_config(page_title="PDF RAG Question Answering", layout="wide")
st.title("📗 PDF-based Question Answering AI")
st.markdown("Upload a PDF — processing starts automatically, then ask questions.😊")

# --- Session State ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_file_bytes" not in st.session_state:
    st.session_state.processed_file_bytes = None

if "question_input" not in st.session_state:
    st.session_state.question_input = ""


# PDF Upload
uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

# Reset state if no file
if uploaded_pdf is None:
    st.session_state.vectorstore = None
    st.session_state.processed_file_bytes = None
    st.session_state.question_input = ""
else:
    uploaded_pdf.seek(0)
    current_bytes = uploaded_pdf.read()

    # Detect NEW file upload → trigger automatic processing
    if st.session_state.processed_file_bytes != current_bytes:

        st.session_state.question_input = ""  # clear question input

        progress_text = "Starting PDF processing..."
        progress_bar = st.progress(0, text=progress_text)

        try:
            # Step 1: Extract Text  
            progress_bar.progress(10, text="Extracting text from PDF...")
            raw_docs = pdf_bytes_to_documents(current_bytes, source_name="uploaded_pdf")

            if not raw_docs:
                progress_bar.empty()
                st.error("❌ Could not extract text from this PDF.")
            else:
                # Step 2: Split into chunks
                progress_bar.progress(40, text="Splitting document into clean chunks...")
                chunks = split_documents(raw_docs)

                # Sanitize chunks to avoid tokenizer errors
                chunks = [
                    doc for doc in chunks
                    if doc.page_content
                    and isinstance(doc.page_content, str)
                    and doc.page_content.strip()
                ]

                if not chunks:
                    progress_bar.empty()
                    st.error("❌ No valid text found after splitting.")
                else:

                    # Step 3: Create Embeddings
                    progress_bar.progress(75, text="Building embeddings & storing in memory...")
                    vectorstore = create_inmemory_vectorstore(chunks)

                    # Step 4: Update session
                    st.session_state.vectorstore = vectorstore
                    st.session_state.processed_file_bytes = current_bytes

                    # Complete
                    progress_bar.progress(100, text="🎉 Processing Complete!")
                    time.sleep(0.4)
                    progress_bar.empty()

                    st.success("PDF processed successfully. You may now ask questions!")

        except Exception as e:
            progress_bar.empty()
            st.error(f"Processing failed: {e}")

    else:
        st.success("PDF already processed. Ready to answer your questions.")


# Q&A
st.subheader("Ask Questions from the PDF")

query = st.text_input("Enter your question", key="question_input")

if st.button("Get Answer", type="primary"):
    if not query.strip():
        st.error("Please enter a question.")
    elif uploaded_pdf is None:
        st.error("⚠️ Upload a PDF first.")
    elif st.session_state.vectorstore is None:
        st.error("⚠️ PDF is still processing. Please wait.")
    else:
        try:
            with st.spinner("Generating answer..."):
                qa_chain = build_rag_chain(st.session_state.vectorstore)
                output = qa_chain.invoke({"query": query})

            st.write("### 🧑‍🏫 Answer")
            st.info(output["result"])

            st.write("### 📚 Sources:")
            for i, src in enumerate(output["source_documents"], 1):
                snippet = src.page_content.replace('\n', ' ').strip()[:150]
                st.markdown(
                    f"**Source {i}:** Page {src.metadata.get('page', 'Unknown')} - *{snippet}...*"
                )

        except Exception as e:
            st.error(f"Error: {e}")
