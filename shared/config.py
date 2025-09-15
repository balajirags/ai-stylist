from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY")
    COLLECTION: str = os.getenv("COLLECTION", "product-catalog")

    SPARSE_EMBEDDING_MODEL: str = os.getenv("SPARSE_EMBEDDING_MODEL", "Qdrant/bm25")
    DENSE_EMBEDDING_MODEL: str = os.getenv("DENSE_EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-small-en")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "384"))


    ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-3-5-sonnet-20240620")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))

settings = Settings()
