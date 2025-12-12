import os
from dotenv import load_dotenv

load_dotenv()

# Groq API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR_KEY_HERE")

# Embedding model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Groq model name 
GROQ_MODEL = "llama-3.3-70b-versatile" 

# Retrieval parameters
RETRIEVAL_K = 3

# Model generation parameters
GENERATION_PARAMS = {
    "temperature": 0.1,
    "max_tokens": 1024
}