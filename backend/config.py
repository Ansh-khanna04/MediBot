import os
from dotenv import load_dotenv
from pathlib import Path
# Load environment variables from .env file
load_dotenv()


EURI_API_KEY = os.getenv('EURI_API_KEY')
EURI_BASE_URL = os.getenv('EURI_BASE_URL')
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'sentence-transformers/all-mpnet-base-v2')
CHUNK_SIZE = 25
CHUNK_OVERLAP = 5
VECTOR_STORE_PATH = "vector_store/"
BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_STORE_PATH = BASE_DIR / "vector_store"
CHAT_MODEL_NAME = os.getenv('CHAT_MODEL_NAME','gpt-4.1-nano')
