from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="RAG Server")
    environment: str = Field(default="dev")
    log_level: str = Field(default="INFO")

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    qdrant_url: str | None = Field(default=None)
    qdrant_path: str | None = Field(default=None)
    qdrant_host: str = Field(default="qdrant")
    qdrant_port: int = Field(default=6333)
    qdrant_collection: str = Field(default="documents")
    qdrant_api_key: str | None = Field(default=None)

    embedding_provider: Literal["openai"] = Field(default="openai")
    embedding_model_name: str = Field(default="text-embedding-3-small")
    embedding_dimension: int = Field(default=1536)
    openai_api_key: str | None = Field(default=None)

    chunk_size: int = Field(default=800)
    chunk_overlap: int = Field(default=120)
    chunking_strategy: Literal["semantic", "sentence"] = Field(default="semantic")
    semantic_buffer_size: int = Field(default=1)
    semantic_breakpoint_percentile: int = Field(default=90)

    default_top_k: int = Field(default=5)
    max_list_points_scan: int = Field(default=10000)
    rerank_enabled: bool = Field(default=True)
    rerank_candidate_min: int = Field(default=24)
    rerank_candidate_max: int = Field(default=120)
    rerank_candidate_multiplier: int = Field(default=6)
    rerank_semantic_weight: float = Field(default=0.65)
    rerank_lexical_weight: float = Field(default=0.3)
    rerank_sequence_weight: float = Field(default=0.05)
    canondock_api_base_url: str = Field(default="http://127.0.0.1:8020")
    canondock_service_token: str | None = Field(default=None)
    canondock_timeout_seconds: float = Field(default=15.0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
