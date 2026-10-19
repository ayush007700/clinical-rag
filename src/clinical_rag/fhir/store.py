"""SQLAlchemy-backed FHIR resource store (Postgres or SQLite)."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Iterator

from sqlalchemy import Column, DateTime, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from clinical_rag.config import get_settings
from clinical_rag.models import FhirResourceRecord


class Base(DeclarativeBase):
    pass


class FhirResourceRow(Base):
    __tablename__ = "fhir_resources"

    resource_type = Column(String(64), primary_key=True)
    resource_id = Column(String(128), primary_key=True)
    patient_id = Column(String(128), index=True, nullable=True)
    encounter_id = Column(String(128), index=True, nullable=True)
    resource_json = Column(Text, nullable=False)
    ingested_at = Column(DateTime, default=datetime.utcnow, nullable=False)


_engine = None
_SessionLocal = None


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        settings = get_settings()
        connect_args = {"check_same_thread": False} if settings.use_sqlite else {}
        _engine = create_engine(settings.database_url, future=True, connect_args=connect_args)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
        Base.metadata.create_all(_engine)
    return _engine


def session() -> Session:
    get_engine()
    assert _SessionLocal is not None
    return _SessionLocal()


def upsert_resource(record: FhirResourceRecord) -> None:
    with session() as db:
        row = db.get(FhirResourceRow, (record.resource_type, record.resource_id))
        payload = json.dumps(record.resource)
        if row is None:
            row = FhirResourceRow(
                resource_type=record.resource_type,
                resource_id=record.resource_id,
                patient_id=record.patient_id,
                encounter_id=record.encounter_id,
                resource_json=payload,
                ingested_at=record.ingested_at,
            )
            db.add(row)
        else:
            row.patient_id = record.patient_id
            row.encounter_id = record.encounter_id
            row.resource_json = payload
            row.ingested_at = record.ingested_at
        db.commit()


def get_resource(resource_type: str, resource_id: str) -> dict[str, Any] | None:
    with session() as db:
        row = db.get(FhirResourceRow, (resource_type, resource_id))
        if row is None:
            return None
        return json.loads(row.resource_json)


def search_resources(
    resource_type: str,
    patient_id: str | None = None,
    encounter_id: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    with session() as db:
        stmt = select(FhirResourceRow).where(FhirResourceRow.resource_type == resource_type)
        if patient_id:
            stmt = stmt.where(FhirResourceRow.patient_id == patient_id)
        if encounter_id:
            stmt = stmt.where(FhirResourceRow.encounter_id == encounter_id)
        stmt = stmt.limit(limit)
        rows = db.execute(stmt).scalars().all()
        return [json.loads(r.resource_json) for r in rows]


def iter_patient_resources(patient_id: str) -> Iterator[dict[str, Any]]:
    with session() as db:
        stmt = select(FhirResourceRow).where(FhirResourceRow.patient_id == patient_id)
        for row in db.execute(stmt).scalars():
            yield json.loads(row.resource_json)


def patient_bundle(patient_id: str) -> dict[str, Any]:
    entries = [{"resource": r} for r in iter_patient_resources(patient_id)]
    # also include Patient/{id} if stored as Patient without prefix
    bare = patient_id.replace("Patient/", "")
    patient = get_resource("Patient", bare)
    if patient:
        entries.insert(0, {"resource": patient})
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "total": len(entries),
        "entry": entries,
    }
