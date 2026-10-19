"""Kafka helpers for high-volume raw clinical event ingestion."""

from __future__ import annotations

import json
from typing import Any, Callable, Iterator

from clinical_rag.config import get_settings
from clinical_rag.logging_setup import get_logger

logger = get_logger(__name__)


def _producer():
    from confluent_kafka import Producer

    settings = get_settings()
    return Producer(
        {
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "linger.ms": 20,
            "compression.type": "lz4",
            "acks": "all",
        }
    )


def publish_raw_event(event: dict[str, Any], topic: str | None = None) -> None:
    settings = get_settings()
    topic = topic or settings.kafka_raw_topic
    producer = _producer()
    key = str(event.get("patient_mrn") or event.get("patient_id") or event.get("event_id"))
    producer.produce(topic, key=key.encode("utf-8"), value=json.dumps(event, default=str).encode())
    producer.flush(5)
    logger.info("kafka_publish", topic=topic, key=key)


def consume_raw_events(
    handler: Callable[[dict[str, Any]], None],
    stop_after: int | None = None,
) -> None:
    """Blocking consumer loop used by the FHIR processing worker."""
    from confluent_kafka import Consumer, KafkaError

    settings = get_settings()
    consumer = Consumer(
        {
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": settings.kafka_group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
        }
    )
    consumer.subscribe([settings.kafka_raw_topic])
    logger.info("kafka_consume_start", topic=settings.kafka_raw_topic)
    processed = 0
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    logger.error("kafka_error", error=str(msg.error()))
                continue
            payload = json.loads(msg.value().decode("utf-8"))
            handler(payload)
            processed += 1
            if stop_after is not None and processed >= stop_after:
                break
    finally:
        consumer.close()


class InMemoryBus:
    """Local fallback when Kafka is unavailable (unit tests / offline demos)."""

    def __init__(self) -> None:
        self._topics: dict[str, list[dict[str, Any]]] = {}

    def publish(self, topic: str, event: dict[str, Any]) -> None:
        self._topics.setdefault(topic, []).append(event)

    def drain(self, topic: str) -> Iterator[dict[str, Any]]:
        events = self._topics.get(topic, [])
        self._topics[topic] = []
        yield from events


_memory_bus = InMemoryBus()


def get_memory_bus() -> InMemoryBus:
    return _memory_bus
