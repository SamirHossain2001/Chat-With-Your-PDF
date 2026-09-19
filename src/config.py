"""Central configuration — single source of truth for models and retrieval knobs.

Values can be overridden via environment variables or a .env file, e.g. LLM_MODEL=...
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str | None = None

    # Models (all CPU-friendly; the LLM runs on Groq)
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "qwen/qwen3.8-27b"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval: fetch a wide candidate set by vector, then rerank down to top_k
    retrieve_candidates: int = 20
    top_k: int = 5

    # Abstain if the best reranked passage scores below this (cross-encoder logit).
    # Measured: in-document questions score ~+6, off-topic ones ~-10.
    min_rerank_score: float = -4.0

    # Per-session question cap (protects the owner's shared Groq quota)
    max_questions_per_session: int = 30


settings = Settings()
