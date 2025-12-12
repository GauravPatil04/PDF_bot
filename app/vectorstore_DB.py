from langchain_community.vectorstores import Chroma
from typing import List
from langchain_core.documents import Document
from .text_embeddings import get_embedding_model

def create_inmemory_vectorstore(documents: List[Document]):
    """
    Create a Chroma vectorstore in-memory.
    No persist_directory is specified, so data lives in RAM.
    """
    if not documents:
        return None
        
    embeddings = get_embedding_model()
    
    # Create vectorstore from documents without persisting to disk
    vectorstore = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings
    )
    return vectorstore