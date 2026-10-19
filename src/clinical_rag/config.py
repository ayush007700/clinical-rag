"""ClinicalRAG configuration via environment / .env."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    app_name: str = "clinical-rag"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "*"
    log_level: str = "INFO"

    jwt_secret: str = "change-me-clinical-rag-dev-secret"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 480
    demo_api_token: str = "demo-token"

    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_raw_topic: str = "clinical.raw.events"
    kafka_fhir_topic: str = "clinical.fhir.resources"
    kafka_group_id: str = "clinical-rag-processor"

    database_url: str = "sqlite:///./data/fhir_store.db"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "ClinicalRAG#dev1"

    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    llm_provider: str = "openai"

    # Hybrid retrieval: bm25 | dense | hybrid (BM25 + dense RRF)
    retrieval_mode: str = "hybrid"
    retrieval_top_k: int = 6
    retrieval_candidate_pool: int = 20
    rrf_k: int = 60
    hybrid_bm25_weight: float = 1.0
    hybrid_dense_weight: float = 1.0
    # Dense vector size stored in pgvector (OpenAI text-embedding-3-* supports dimensions=)
    embedding_dim: int = 384
    local_embedding_dim: int = 384
    use_pgvector: bool = True

    # Grounding / cite-or-refuse
    grounding_require_retrieval: bool = True
    grounding_min_chunks: int = 1
    grounding_require_citations: bool = True

    iodineiq_model_dir: str = "data/models"
    risk_model_path: str = "data/models/patient_risk_model.joblib"
    summary_max_tokens: int = 2048

    @property
    def use_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


@lru_cache
def get_settings() -> Settings:
    return Settings()
