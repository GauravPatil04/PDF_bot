import streamlit as st
from app.PDF_loader import pdf_bytes_to_documents
from app.splitter import split_documents
from app.vectorstore_DB import create_inmemory_vectorstore
from app.RAG_system_pipeline import build_rag_chain

# --- UI Setup ---
st.set_page_config(page_title="PDF RAG Question Answering", layout="wide")
st.title("📘 PDF-based Question Answering AI")
st.markdown("Upload a PDF, process it, and then ask questions.")

# --- Session State Initialization ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

# --- Main Layout ---
# 1. File Upload
uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_pdf is not None:
    st.success(f"PDF uploaded successfully")

# 2. Processing
disable_process_button = (uploaded_pdf is None)

if st.button("Process PDF (Embed & Store)", disabled=disable_process_button):
    # Prevent re-processing if user clicks multiple times on the same file
    if st.session_state.processed_file == uploaded_pdf.name:
        st.warning("This PDF is already processed.")
    else:
        with st.spinner("Processing PDF..."):
            # A. Read bytes directly from the uploaded file object
            # We reset the pointer to the start of the file just in case
            uploaded_pdf.seek(0)
            pdf_bytes = uploaded_pdf.read()
                
            # B. Load (Extract text from bytes)
            raw_docs = pdf_bytes_to_documents(pdf_bytes, source_name=uploaded_pdf.name)
                
            if not raw_docs:
                st.error("Could not extract text from this PDF.")
            else:
                # C. Splitting
                chunks = split_documents(raw_docs)
                st.write(f"Document split into {len(chunks)} chunks.")
                
                # D. Embed & Store (In-Memory)
                vectorstore = create_inmemory_vectorstore(chunks)
                    
                # Save to Session State
                st.session_state.vectorstore = vectorstore
                st.session_state.processed_file = uploaded_pdf.name
                    
                st.success("PDF processed! Ready for questions.")    

# 3. Q&A
st.subheader("Ask Questions from the PDF")
query = st.text_input("Enter your question")

if st.button("Get Answer", type="primary"):
    if not query:
        st.error("Please enter a question.")
    elif st.session_state.vectorstore is None:
         st.error("⚠️ Please process a PDF file first before asking questions.")
    else:
        try:
            with st.spinner("Retrieving answer..."):
                # Pass the in-memory vectorstore to the pipeline
                qa_chain = build_rag_chain(st.session_state.vectorstore)
                output = qa_chain({"query": query})

            st.write("### Answer:")
            st.info(output["result"])

            st.write("### 📚 Sources:")
            for i, src in enumerate(output["source_documents"], 1):
                snippet = src.page_content.replace('\n', ' ').strip()[:150]
                st.markdown(f"**Source {i}:** Page {src.metadata.get('page', 'Unknown')} - *{snippet}...*")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")