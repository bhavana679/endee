import os

# Base directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "db")

# Chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Embedding settings
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

# Endee Vector DB settings (Sample placeholders)
ENDEE_HOST = os.getenv("ENDEE_HOST", "localhost")
ENDEE_PORT = os.getenv("ENDEE_PORT", "8080")

# LLM configurations (Sample placeholders)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
