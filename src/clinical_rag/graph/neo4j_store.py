"""Neo4j clinical knowledge graph store for GraphRAG."""

from __future__ import annotations

from typing import Any

from neo4j import GraphDatabase

from clinical_rag.config import get_settings
from clinical_rag.logging_setup import get_logger

logger = get_logger(__name__)


class ClinicalGraphStore:
    def __init__(self) -> None:
        settings = get_settings()
        self._driver = None
        self._uri = settings.neo4j_uri
        self._user = settings.neo4j_user
        self._password = settings.neo4j_password
        self._memory: dict[str, list[dict[str, Any]]] = {"chunks": [], "edges": []}

    def _get_driver(self):
        if self._driver is None:
            try:
                self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
                self._driver.verify_connectivity()
            except Exception as exc:  # noqa: BLE001
                logger.warning("neo4j_unavailable_fallback_memory", error=str(exc))
                self._driver = False  # type: ignore[assignment]
        return self._driver

    def close(self) -> None:
        if self._driver and self._driver is not False:
            self._driver.close()

    def ensure_schema(self) -> None:
        driver = self._get_driver()
        if not driver:
            return
        statements = [
            "CREATE CONSTRAINT patient_id IF NOT EXISTS FOR (p:Patient) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT code_id IF NOT EXISTS FOR (m:MedicalCode) REQUIRE m.code IS UNIQUE",
        ]
        with driver.session() as session:
            for stmt in statements:
                try:
                    session.run(stmt)
                except Exception:  # noqa: BLE001
                    pass

    def upsert_patient(self, patient_id: str, props: dict[str, Any] | None = None) -> None:
        props = props or {}
        driver = self._get_driver()
        if not driver:
            return
        with driver.session() as session:
            session.run(
                """
                MERGE (p:Patient {id: $id})
                SET p += $props
                """,
                id=patient_id,
                props=props,
            )

    def upsert_chunk(
        self,
        chunk_id: str,
        text: str,
        patient_id: str,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        metadata = metadata or {}
        driver = self._get_driver()
        if not driver:
            self._memory["chunks"].append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "patient_id": patient_id,
                    "source": source,
                    **metadata,
                }
            )
            return
        with driver.session() as session:
            session.run(
                """
                MERGE (c:Chunk {id: $id})
                SET c.text = $text, c.source = $source, c.patient_id = $patient_id, c += $metadata
                WITH c
                MERGE (p:Patient {id: $patient_id})
                MERGE (p)-[:HAS_NOTE]->(c)
                """,
                id=chunk_id,
                text=text,
                source=source,
                patient_id=patient_id,
                metadata=metadata,
            )

    def link_code(self, patient_id: str, code: str, display: str, confidence: float) -> None:
        driver = self._get_driver()
        if not driver:
            self._memory["edges"].append(
                {"patient_id": patient_id, "code": code, "display": display, "confidence": confidence}
            )
            return
        with driver.session() as session:
            session.run(
                """
                MERGE (m:MedicalCode {code: $code})
                SET m.display = $display
                WITH m
                MERGE (p:Patient {id: $patient_id})
                MERGE (p)-[r:HAS_CODE]->(m)
                SET r.confidence = $confidence
                """,
                code=code,
                display=display,
                patient_id=patient_id,
                confidence=confidence,
            )

    def fetch_patient_chunks(self, patient_id: str, limit: int = 50) -> list[dict[str, Any]]:
        driver = self._get_driver()
        if not driver:
            return [c for c in self._memory["chunks"] if c.get("patient_id") == patient_id][:limit]
        with driver.session() as session:
            result = session.run(
                """
                MATCH (p:Patient {id: $patient_id})-[:HAS_NOTE]->(c:Chunk)
                RETURN c.id AS chunk_id, c.text AS text, c.source AS source, c.patient_id AS patient_id
                LIMIT $limit
                """,
                patient_id=patient_id,
                limit=limit,
            )
            return [dict(r) for r in result]

    def neighborhood(self, patient_id: str) -> dict[str, Any]:
        driver = self._get_driver()
        if not driver:
            return {
                "codes": [e for e in self._memory["edges"] if e.get("patient_id") == patient_id],
                "chunks": self.fetch_patient_chunks(patient_id),
            }
        with driver.session() as session:
            codes = session.run(
                """
                MATCH (p:Patient {id: $patient_id})-[r:HAS_CODE]->(m:MedicalCode)
                RETURN m.code AS code, m.display AS display, r.confidence AS confidence
                """,
                patient_id=patient_id,
            )
            return {
                "codes": [dict(r) for r in codes],
                "chunks": self.fetch_patient_chunks(patient_id),
            }
