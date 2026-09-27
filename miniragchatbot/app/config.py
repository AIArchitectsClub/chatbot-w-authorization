"""Application settings, loaded from environment variables / `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Auth
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    auth_timeout_seconds: float = 10.0

    # Trust the OS certificate store (for networks with TLS-intercepting proxies).
    use_system_trust_store: bool = False

    # LLM (provider is chosen here; see app/llm/factory.py)
    llm_provider: Literal["ollama", "groq", "openai"] = "ollama"
    llm_temperature: float = 0.0
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384

    # Vector DB: a server URL wins; otherwise Qdrant runs embedded on local disk.
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_path: Path = Path("./qdrant_data")
    qdrant_collection: str = "policies"

    # Ingestion / retrieval
    docs_dir: Path = Path("./docs")
    chunk_max_chars: int = 700
    chunk_overlap_chars: int = 100
    retrieval_top_k: int = 4
    # Cosine-similarity floor for bge-small: on this corpus, on-topic questions
    # score ~0.74-0.82, personal/off-topic ones ~0.4-0.62. Chunks below the floor
    # are not shown to the model, so it cannot be misled by irrelevant context.
    retrieval_min_score: float = 0.65

    # Tool
    tool_timeout_seconds: float = 5.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
