from typing import List
from langchain_core.documents import Document
from pypdf import PdfReader
from io import BytesIO

def pdf_bytes_to_documents(pdf_bytes: bytes, source_name: str = "uploaded_pdf") -> List[Document]:
    """
    Convert PDF bytes into a list of LangChain Document objects.
    Each page becomes one Document.
    """
    reader = PdfReader(BytesIO(pdf_bytes))
    docs = []
    
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        # Only add pages that have actual text
        if text.strip():
            doc = Document(
                page_content=text, 
                metadata={"source": source_name, "page": i}
            )
            docs.append(doc)
            
    return docs