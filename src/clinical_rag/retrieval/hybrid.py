"""Patient-scoped hybrid retrieval: BM25 + pgvector dense, fused with RRF."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from clinical_rag.config import get_settings
from clinical_rag.graph.neo4j_store import ClinicalGraphStore
from clinical_rag.logging_setup import get_logger
from clinical_rag.models import RetrievedChunk
from clinical_rag.retrieval.embeddings import (
    EmbeddingClient,
    cosine_similarity_matrix,
    get_embedding_client,
)
from clinical_rag.retrieval.pgvector_store import PgVectorStore, pgvector_available

logger = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[int, float]]:
    """RRF over lists of document indices (best rank first)."""
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights length must match ranked_lists")
    scores: dict[int, float] = {}
    for weight, ranking in zip(weights, ranked_lists, strict=True):
        for rank, doc_idx in enumerate(ranking, start=1):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + float(weight) / (k + rank)
    return sorted(scores.items(), key=lambda x: (x[1], -x[0]), reverse=True)


def reciprocal_rank_fusion_ids(
    ranked_lists: list[list[str]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """RRF over chunk_id rankings."""
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights length must match ranked_lists")
    scores: dict[str, float] = {}
    for weight, ranking in zip(weights, ranked_lists, strict=True):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + float(weight) / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """BM25 (sparse) + dense (pgvector ANN or in-memory), fused via RRF.

    Always patient-scoped via ``patient_id`` filter before ranking.
    """

    def __init__(
        self,
        store: ClinicalGraphStore | None = None,
        embedder: EmbeddingClient | None = None,
        vector_store: PgVectorStore | None = None,
    ) -> None:
        self.store = store or ClinicalGraphStore()
        self.settings = get_settings()
        self.embedder = embedder or get_embedding_client()
        self.vector_store = vector_store or PgVectorStore()

    def _bm25_ranking(self, query: str, texts: list[str], pool: int) -> list[int]:
        corpus = [_tokenize(t) for t in texts]
        if not any(corpus):
            return list(range(min(pool, len(texts))))
        bm25 = BM25Okapi(corpus)
        scores = np.asarray(bm25.get_scores(_tokenize(query)), dtype=np.float64)
        # Stable descending sort so keyword hits stay ahead of zero-score docs
        order = np.argsort(-scores, kind="mergesort")
        positive = [int(i) for i in order if scores[i] > 0]
        if positive:
            return positive[:pool]
        return [int(i) for i in order[:pool]]

    def _dense_ranking_memory(
        self, query: str, texts: list[str], pool: int
    ) -> tuple[list[int], np.ndarray]:
        doc_matrix = self.embedder.embed_documents(texts)
        query_vec = self.embedder.embed_query(query)
        sims = cosine_similarity_matrix(query_vec, doc_matrix)
        order = np.argsort(sims)[::-1]
        ranking = [int(i) for i in order[:pool]]
        return ranking, sims

    def _use_pgvector(self) -> bool:
        return bool(self.settings.use_pgvector) and pgvector_available()

    def retrieve(self, query: str, patient_id: str, top_k: int | None = None) -> list[RetrievedChunk]:
        settings = self.settings
        top_k = top_k if top_k is not None else settings.retrieval_top_k
        pool = max(settings.retrieval_candidate_pool, top_k)
        mode = (settings.retrieval_mode or "hybrid").lower()

        # Prefer Neo4j chunks for BM25/graph continuity; fall back to pgvector texts
        rows = self.store.fetch_patient_chunks(patient_id, limit=200)
        if not rows and self._use_pgvector():
            rows = self.vector_store.fetch_patient_chunks(patient_id, limit=200)
        if not rows:
            return []

        by_id = {str(r.get("chunk_id")): r for r in rows}
        texts = [str(r.get("text") or "") for r in rows]
        ids = [str(r.get("chunk_id")) for r in rows]

        bm25_idx = self._bm25_ranking(query, texts, pool)
        bm25_ids = [ids[i] for i in bm25_idx]

        dense_ids: list[str] = []
        dense_cosine_by_id: dict[str, float] = {}
        provider = self.embedder.provider_name
        used_pgvector = False

        if mode != "bm25":
            if self._use_pgvector():
                qvec = self.embedder.embed_query(query)
                try:
                    hits = self.vector_store.search(
                        patient_id=patient_id, query_embedding=qvec, top_k=pool
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("pgvector_search_fallback_memory", error=str(exc))
                    hits = []
                if hits:
                    used_pgvector = True
                    dense_ids = [h["chunk_id"] for h in hits]
                    dense_cosine_by_id = {h["chunk_id"]: float(h["dense_cosine"]) for h in hits}
                    # Merge any pgvector-only rows into lookup for citation text
                    for h in hits:
                        by_id.setdefault(
                            h["chunk_id"],
                            {
                                "chunk_id": h["chunk_id"],
                                "text": h["text"],
                                "source": h.get("source"),
                                "patient_id": h.get("patient_id") or patient_id,
                            },
                        )
                    provider = f"pgvector+{self.embedder.provider_name}"

            if not dense_ids:
                dense_idx, sims = self._dense_ranking_memory(query, texts, pool)
                dense_ids = [ids[i] for i in dense_idx]
                dense_cosine_by_id = {ids[i]: float(sims[i]) for i in range(len(ids))}
                provider = f"memory+{self.embedder.provider_name}"

        if mode == "bm25":
            fused = [(cid, float(1.0 / (settings.rrf_k + rank))) for rank, cid in enumerate(bm25_ids, 1)]
            provider = "bm25-only"
        elif mode == "dense":
            fused = [(cid, float(1.0 / (settings.rrf_k + rank))) for rank, cid in enumerate(dense_ids, 1)]
        else:
            fused = reciprocal_rank_fusion_ids(
                [bm25_ids, dense_ids],
                k=settings.rrf_k,
                weights=[settings.hybrid_bm25_weight, settings.hybrid_dense_weight],
            )

        bm25_pos = {cid: rank for rank, cid in enumerate(bm25_ids, start=1)}
        dense_pos = {cid: rank for rank, cid in enumerate(dense_ids, start=1)}

        out: list[RetrievedChunk] = []
        for cid, rrf_score in fused[:top_k]:
            row = by_id.get(cid)
            if row is None:
                continue
            meta: dict[str, Any] = {
                "source": row.get("source"),
                "patient_id": row.get("patient_id") or patient_id,
                "retrieval": mode if mode in {"bm25", "dense"} else "hybrid_rrf",
                "embedding_provider": provider,
                "dense_backend": "pgvector" if used_pgvector else ("memory" if mode != "bm25" else None),
                "bm25_rank": bm25_pos.get(cid),
                "dense_rank": dense_pos.get(cid),
                "rrf_score": round(float(rrf_score), 6),
            }
            if cid in dense_cosine_by_id:
                meta["dense_cosine"] = round(dense_cosine_by_id[cid], 4)
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=str(row.get("text") or ""),
                    metadata=meta,
                    score=float(rrf_score),
                )
            )

        logger.info(
            "hybrid_retrieve",
            patient_id=patient_id,
            mode=mode if mode in {"bm25", "dense"} else "hybrid_rrf",
            provider=provider,
            dense_backend="pgvector" if used_pgvector else "memory",
            candidates=len(rows),
            returned=len(out),
        )
        return out

    def graph_context(self, patient_id: str) -> dict[str, Any]:
        return self.store.neighborhood(patient_id)
