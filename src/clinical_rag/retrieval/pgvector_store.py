"""Postgres + pgvector store for patient-scoped dense chunk embeddings."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from sqlalchemy import Column, DateTime, String, Text, create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from clinical_rag.config import get_settings
from clinical_rag.logging_setup import get_logger

logger = get_logger(__name__)


class VectorBase(DeclarativeBase):
    pass


_engine = None
_SessionLocal = None
_enabled: bool | None = None
_dim: int | None = None
_ChunkEmbedding = None  # dynamically built for VECTOR(dim)


def _build_chunk_model(dim: int):
    from pgvector.sqlalchemy import Vector

    class ChunkEmbedding(VectorBase):
        __tablename__ = "chunk_embeddings"
        __table_args__ = {"extend_existing": True}

        chunk_id = Column(String(256), primary_key=True)
        patient_id = Column(String(128), index=True, nullable=False)
        source = Column(String(128), nullable=True)
        text = Column(Text, nullable=False)
        embedding = Column(Vector(dim), nullable=False)
        provider = Column(String(128), nullable=True)
        updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    return ChunkEmbedding


def pgvector_available() -> bool:
    """True when DATABASE_URL is Postgres and the vector extension works."""
    global _enabled
    if _enabled is not None:
        return _enabled
    settings = get_settings()
    if settings.use_sqlite:
        _enabled = False
        return False
    try:
        ensure_pgvector_schema()
        _enabled = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("pgvector_unavailable", error=str(exc))
        _enabled = False
    return _enabled


def ensure_pgvector_schema() -> None:
    """CREATE EXTENSION vector + chunk_embeddings table + optional HNSW index."""
    global _engine, _SessionLocal, _ChunkEmbedding, _dim

    settings = get_settings()
    if settings.use_sqlite:
        raise RuntimeError("pgvector requires Postgres DATABASE_URL")

    dim = int(settings.embedding_dim)
    if _engine is not None and _dim == dim and _ChunkEmbedding is not None:
        return

    _engine = create_engine(settings.database_url, future=True)
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    _dim = dim
    _ChunkEmbedding = _build_chunk_model(dim)

    with _engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    VectorBase.metadata.create_all(_engine, tables=[_ChunkEmbedding.__table__])

    # Best-effort ANN index (safe to skip if unsupported / too few rows)
    with _engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS chunk_embeddings_patient_id_idx
                ON chunk_embeddings (patient_id)
                """
            )
        )
        try:
            conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS chunk_embeddings_hnsw_cosine_idx
                    ON chunk_embeddings
                    USING hnsw (embedding vector_cosine_ops)
                    """
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("pgvector_hnsw_index_skip", error=str(exc))

    logger.info("pgvector_schema_ready", dim=dim)


def _session() -> Session:
    ensure_pgvector_schema()
    assert _SessionLocal is not None
    return _SessionLocal()


def reset_pgvector_state() -> None:
    """Clear cached engine flags (tests / settings reload)."""
    global _engine, _SessionLocal, _enabled, _dim, _ChunkEmbedding
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
    _enabled = None
    _dim = None
    _ChunkEmbedding = None


class PgVectorStore:
    """Persist and ANN-search L2-normalized chunk embeddings, patient-scoped."""

    def upsert_chunk(
        self,
        *,
        chunk_id: str,
        patient_id: str,
        text: str,
        embedding: np.ndarray,
        source: str | None = None,
        provider: str | None = None,
    ) -> None:
        if not pgvector_available():
            return
        assert _ChunkEmbedding is not None
        vec = embedding.astype(np.float32).ravel().tolist()
        with _session() as db:
            row = db.get(_ChunkEmbedding, chunk_id)
            if row is None:
                row = _ChunkEmbedding(
                    chunk_id=chunk_id,
                    patient_id=patient_id,
                    source=source,
                    text=text,
                    embedding=vec,
                    provider=provider,
                    updated_at=datetime.utcnow(),
                )
                db.add(row)
            else:
                row.patient_id = patient_id
                row.source = source
                row.text = text
                row.embedding = vec
                row.provider = provider
                row.updated_at = datetime.utcnow()
            db.commit()

    def search(
        self,
        *,
        patient_id: str,
        query_embedding: np.ndarray,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """Return chunks ordered by cosine distance (smaller distance = closer)."""
        if not pgvector_available():
            return []
        assert _ChunkEmbedding is not None
        q = query_embedding.astype(np.float32).ravel().tolist()
        with _session() as db:
            distance = _ChunkEmbedding.embedding.cosine_distance(q)
            stmt = (
                select(
                    _ChunkEmbedding.chunk_id,
                    _ChunkEmbedding.text,
                    _ChunkEmbedding.source,
                    _ChunkEmbedding.patient_id,
                    distance.label("distance"),
                )
                .where(_ChunkEmbedding.patient_id == patient_id)
                .order_by(distance)
                .limit(top_k)
            )
            rows = db.execute(stmt).all()
            out: list[dict[str, Any]] = []
            for r in rows:
                dist = float(r.distance)
                out.append(
                    {
                        "chunk_id": r.chunk_id,
                        "text": r.text,
                        "source": r.source,
                        "patient_id": r.patient_id,
                        "distance": dist,
                        # cosine distance on normalized vectors ≈ 1 - cosine_sim
                        "dense_cosine": round(1.0 - dist, 4),
                    }
                )
            return out

    def fetch_patient_chunks(self, patient_id: str, limit: int = 200) -> list[dict[str, Any]]:
        if not pgvector_available():
            return []
        assert _ChunkEmbedding is not None
        with _session() as db:
            stmt = (
                select(_ChunkEmbedding)
                .where(_ChunkEmbedding.patient_id == patient_id)
                .limit(limit)
            )
            rows = db.execute(stmt).scalars().all()
            return [
                {
                    "chunk_id": r.chunk_id,
                    "text": r.text,
                    "source": r.source,
                    "patient_id": r.patient_id,
                }
                for r in rows
            ]
