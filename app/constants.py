import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data files
DISASTER_CSV_PATH = os.path.join(BASE_DIR, "data", "disaster_declarations.csv")

# Saved index paths
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "saved_index", "disaster_faiss.index")
FAISS_METADATA_PATH = os.path.join(BASE_DIR, "saved_index", "disaster_metadata.pkl")

# OpenAI
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-3.5-turbo"
EVAL_MODEL = "gpt-4"

# Search & Retrieval
TOP_K = 5  # Number of similar results to retrieve

# Guardrails & Validation
MIN_ANSWER_LENGTH = 10  # Minimum length to be considered valid
