"""Ingest raw Kafka/memory events → clean FHIR → store → index GraphRAG chunks."""

from __future__ import annotations

from typing import Any

from clinical_rag.fhir.converter import ensure_patient_resource, to_fhir_resources
from clinical_rag.fhir.store import upsert_resource
from clinical_rag.graph.neo4j_store import ClinicalGraphStore
from clinical_rag.logging_setup import get_logger
from clinical_rag.models import FhirResourceRecord
from clinical_rag.retrieval.embeddings import get_embedding_client
from clinical_rag.retrieval.pgvector_store import PgVectorStore, pgvector_available

logger = get_logger(__name__)


class FhirProcessingPipeline:
    def __init__(
        self,
        graph: ClinicalGraphStore | None = None,
        vector_store: PgVectorStore | None = None,
    ) -> None:
        self.graph = graph or ClinicalGraphStore()
        self.vector_store = vector_store or PgVectorStore()
        self.embedder = get_embedding_client()
        try:
            self.graph.ensure_schema()
        except Exception as exc:  # noqa: BLE001
            logger.warning("graph_schema_skip", error=str(exc))
        if pgvector_available():
            logger.info("pgvector_ingest_enabled")

    def process_event(self, raw_event: dict[str, Any]) -> list[FhirResourceRecord]:
        mrn = str(raw_event.get("patient_mrn") or "unknown")
        patient = ensure_patient_resource(mrn, name=raw_event.get("patient_name") or f"Patient {mrn}")
        upsert_resource(
            FhirResourceRecord(
                resource_type="Patient",
                resource_id=mrn,
                patient_id=f"Patient/{mrn}",
                encounter_id=None,
                resource=patient,
            )
        )
        self.graph.upsert_patient(f"Patient/{mrn}", {"mrn": mrn})

        records = to_fhir_resources(raw_event)
        for rec in records:
            upsert_resource(rec)
            self._index_for_rag(rec)
        logger.info("fhir_processed", patient=mrn, resources=len(records))
        return records

    def _index_for_rag(self, rec: FhirResourceRecord) -> None:
        if rec.resource_type != "DocumentReference":
            return
        text = ""
        for c in rec.resource.get("content") or []:
            text = (c.get("attachment") or {}).get("data_text") or text
        if not text:
            return
        chunk_id = f"{rec.resource_type}/{rec.resource_id}"
        patient_id = rec.patient_id or "Patient/unknown"
        self.graph.upsert_chunk(
            chunk_id=chunk_id,
            text=text,
            patient_id=patient_id,
            source="DocumentReference",
            metadata={"resource_id": rec.resource_id},
        )
        # Persist dense vector in Postgres/pgvector for ANN retrieve
        if pgvector_available():
            try:
                emb = self.embedder.embed_documents([text])[0]
                self.vector_store.upsert_chunk(
                    chunk_id=chunk_id,
                    patient_id=patient_id,
                    text=text,
                    embedding=emb,
                    source="DocumentReference",
                    provider=self.embedder.provider_name,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("pgvector_index_failed", chunk_id=chunk_id, error=str(exc))
