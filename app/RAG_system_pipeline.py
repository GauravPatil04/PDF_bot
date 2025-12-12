from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq
from .config import GROQ_API_KEY, GROQ_MODEL, RETRIEVAL_K, GENERATION_PARAMS

def get_llm():
    """Returns the Groq LLM instance."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing. Please set it in your .env file.")
        
    return ChatGroq(
        api_key=GROQ_API_KEY, 
        model=GROQ_MODEL, 
        temperature=GENERATION_PARAMS["temperature"],
        max_tokens=GENERATION_PARAMS["max_tokens"]
    )

def build_rag_chain(vectorstore):
    """Builds the RAG chain using the provided in-memory vectorstore."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    llm = get_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=retriever, 
        return_source_documents=True
    )
    return qa_chain