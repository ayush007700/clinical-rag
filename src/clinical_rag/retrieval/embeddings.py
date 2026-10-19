"""Dense embedding providers for hybrid retrieval.

OpenAI when ``OPENAI_API_KEY`` is set; otherwise a local HashingVectorizer
embedding so BM25+dense hybrid works offline for demos and tests.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from clinical_rag.config import get_settings
from clinical_rag.logging_setup import get_logger

logger = get_logger(__name__)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    return normalize(matrix, norm="l2", axis=1)


class LocalHashEmbedder:
    """Deterministic local dense vectors (no network). Good offline fallback."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self._vectorizer = HashingVectorizer(
            n_features=dim,
            alternate_sign=False,
            norm=None,
            lowercase=True,
            ngram_range=(1, 2),
        )

    @property
    def provider_name(self) -> str:
        return f"local-hashing-{self.dim}"

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        matrix = self._vectorizer.transform(list(texts)).astype(np.float32).toarray()
        return _l2_normalize(matrix).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str, dimensions: int | None = None) -> None:
        from openai import OpenAI

        self.model = model
        self.dimensions = dimensions
        self._client = OpenAI(api_key=api_key)

    @property
    def provider_name(self) -> str:
        dim = f":{self.dimensions}d" if self.dimensions else ""
        return f"openai:{self.model}{dim}"

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            dim = self.dimensions or 1536
            return np.zeros((0, dim), dtype=np.float32)
        kwargs: dict = {"model": self.model, "input": list(texts)}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        response = self._client.embeddings.create(**kwargs)
        vectors = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        matrix = np.asarray(vectors, dtype=np.float32)
        return _l2_normalize(matrix).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class EmbeddingClient:
    """Facade: OpenAI if configured, else local hashing dense vectors."""

    def __init__(self, prefer_openai: bool | None = None) -> None:
        settings = get_settings()
        dim = int(settings.embedding_dim or settings.local_embedding_dim)
        use_openai = prefer_openai if prefer_openai is not None else bool(settings.openai_api_key)
        if use_openai and settings.openai_api_key:
            try:
                self._impl: LocalHashEmbedder | OpenAIEmbedder = OpenAIEmbedder(
                    settings.openai_api_key,
                    settings.openai_embedding_model,
                    dimensions=dim,
                )
                self.dim = dim
                logger.info("embeddings_provider", provider=self._impl.provider_name)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("openai_embedder_failed_fallback_local", error=str(exc))
        self._impl = LocalHashEmbedder(dim=dim)
        self.dim = dim
        logger.info("embeddings_provider", provider=self._impl.provider_name)

    @property
    def provider_name(self) -> str:
        return self._impl.provider_name

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._impl.embed_documents(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._impl.embed_query(text)


def cosine_similarity_matrix(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """Return cosine similarities assuming L2-normalized rows."""
    if doc_matrix.size == 0:
        return np.asarray([], dtype=np.float32)
    q = query_vec.reshape(1, -1)
    return (doc_matrix @ q.T).ravel().astype(np.float32)


@lru_cache
def get_embedding_client() -> EmbeddingClient:
    return EmbeddingClient()
