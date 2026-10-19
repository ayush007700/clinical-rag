"""Prometheus metrics helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUESTS = Counter("clinical_rag_requests_total", "API requests", ["endpoint", "status"])
LATENCY = Histogram("clinical_rag_request_latency_seconds", "Latency", ["endpoint"])


@contextmanager
def track_request(endpoint: str) -> Iterator[dict[str, Any]]:
    meta: dict[str, Any] = {}
    with LATENCY.labels(endpoint=endpoint).time():
        try:
            yield meta
            REQUESTS.labels(endpoint=endpoint, status="ok").inc()
        except Exception:
            REQUESTS.labels(endpoint=endpoint, status="error").inc()
            raise


def metrics_endpoint() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
