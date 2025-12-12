from langchain_community.vectorstores import Chroma
from typing import List, Optional
from langchain_core.documents import Document
from .text_embeddings import get_embedding_model
import uuid
import logging

logger = logging.getLogger(__name__)


def _sanitize_text(text: str) -> str:
    """Ensure the text sent to the embedding model is clean and safe."""
    if not isinstance(text, str):
        text = str(text)

    text = text.replace("\x00", " ").replace("\u0000", " ").strip()

    if len(text.strip()) == 0:
        return ""

    return text


def _generate_collection_name() -> str:
    return f"col_{uuid.uuid4().hex}"


def create_inmemory_vectorstore(documents: List[Document]) -> Optional[Chroma]:
    if not documents:
        return None

    cleaned_docs = []
    for doc in documents:
        cleaned_text = _sanitize_text(doc.page_content)
        if cleaned_text.strip():  
            cleaned_docs.append(
                Document(page_content=cleaned_text, metadata=doc.metadata)
            )

    if not cleaned_docs:
        raise ValueError("No valid text found after sanitizing PDF content.")

    embeddings = get_embedding_model()
    collection_name = _generate_collection_name()

    try:
        vectorstore = Chroma.from_documents(
            documents=cleaned_docs,
            embedding=embeddings,
            collection_name=collection_name
        )
        logger.info(f"Vectorstore created with collection: {collection_name}")
        return vectorstore

    except Exception as e:
        logger.exception("Chroma failed while creating vectorstore.")
        raise e
