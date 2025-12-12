from langchain_community.embeddings import HuggingFaceEmbeddings
from .config import EMBEDDING_MODEL

def get_embedding_model():
    """Return the configured Hugging Face embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)