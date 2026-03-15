import os

# Base directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "db")

# Chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Embedding settings (Gemini)
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
EMBEDDING_DIMENSION = 768

# Endee Vector DB settings
ENDEE_HOST = os.getenv("ENDEE_HOST", "localhost")
ENDEE_PORT = os.getenv("ENDEE_PORT", "8080")

# LLM configurations (Gemini)
LLM_PROVIDER = "gemini"
LLM_MODEL = "gemini-1.5-flash"
