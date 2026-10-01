"""Runtime configuration, loaded from environment variables or a .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    # LLM / embeddings. Leave OPENAI_API_KEY unset to run fully offline.
    openai_api_key: str | None = None
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Vector store
    chroma_path: str = ".chroma"
    collection: str = "knowledge_base"

    # Chunking
    chunk_size: int = 400
    chunk_overlap: int = 60

    # Retrieval
    top_k: int = 4
    candidate_k: int = 12  # candidates pulled from each retriever before fusion

    # Reliability
    min_grounding_score: float = 0.35
    max_retrieval_attempts: int = 2
    max_tool_retries: int = 2


@lru_cache
def get_settings() -> Settings:
    import os

    s = Settings()
    if s.openai_api_key is None and os.getenv("OPENAI_API_KEY"):
        s.openai_api_key = os.environ["OPENAI_API_KEY"]
    return s
