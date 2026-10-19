"""Kafka worker: consume raw clinical events and run FHIR pipeline."""

from __future__ import annotations

from clinical_rag.kafka import consume_raw_events, get_memory_bus
from clinical_rag.logging_setup import configure_logging, get_logger
from clinical_rag.pipeline import FhirProcessingPipeline
from clinical_rag.config import get_settings

configure_logging()
logger = get_logger(__name__)


def main() -> None:
    settings = get_settings()
    pipe = FhirProcessingPipeline()

    def handle(event: dict) -> None:
        pipe.process_event(event)

    logger.info("worker_start", mode="kafka_or_memory")
    try:
        consume_raw_events(handle)
    except Exception as exc:  # noqa: BLE001
        logger.warning("kafka_unavailable_draining_memory", error=str(exc))
        bus = get_memory_bus()
        for event in bus.drain(settings.kafka_raw_topic):
            handle(event)


if __name__ == "__main__":
    main()
