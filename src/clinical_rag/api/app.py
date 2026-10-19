"""FastAPI entrypoint: secure FHIR API + IodineIQ + GraphRAG summaries."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from clinical_rag.auth import AuthUser, create_access_token
from clinical_rag.config import get_settings
from clinical_rag.fhir.store import get_resource, patient_bundle, search_resources
from clinical_rag.iodineiq import IodineIQEngine
from clinical_rag.kafka import get_memory_bus, publish_raw_event
from clinical_rag.logging_setup import configure_logging, get_logger
from clinical_rag.ml import train_and_persist
from clinical_rag.monitoring import metrics_endpoint, track_request
from clinical_rag.pipeline import FhirProcessingPipeline
from clinical_rag.summaries import SummaryService

configure_logging()
logger = get_logger(__name__)
settings = get_settings()

app = FastAPI(
    title="ClinicalRAG CDS",
    description=(
        "Kafka→FHIR clinical ingestion, IodineIQ Cognitive ML, "
        "Neo4j GraphRAG patient & insurance summaries"
    ),
    version="1.0.0",
)

origins = [o.strip() for o in settings.cors_origins.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: FhirProcessingPipeline | None = None
_iodine: IodineIQEngine | None = None
_summaries: SummaryService | None = None


def pipeline() -> FhirProcessingPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = FhirProcessingPipeline()
    return _pipeline


def iodine() -> IodineIQEngine:
    global _iodine
    if _iodine is None:
        _iodine = IodineIQEngine()
    return _iodine


def summaries() -> SummaryService:
    global _summaries
    if _summaries is None:
        _summaries = SummaryService(iodineiq=iodine())
    return _summaries


class TokenRequest(BaseModel):
    username: str = "clinician"
    roles: list[str] = Field(default_factory=lambda: ["clinician", "coder"])


class IngestRequest(BaseModel):
    event: dict[str, Any]
    use_kafka: bool = False


class InteractRequest(BaseModel):
    patient_id: str
    question: str


class SummaryRequest(BaseModel):
    patient_id: str
    encounter_id: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name, "env": settings.app_env}


@app.get("/metrics")
def metrics() -> Response:
    payload, content_type = metrics_endpoint()
    return Response(content=payload, media_type=content_type)


@app.post("/v1/auth/token")
def issue_token(body: TokenRequest) -> dict[str, str]:
    token = create_access_token(body.username, roles=body.roles)
    return {"access_token": token, "token_type": "bearer"}


@app.post("/v1/ingest")
def ingest(body: IngestRequest, user: AuthUser) -> dict[str, Any]:
    with track_request("ingest") as meta:
        if body.use_kafka:
            try:
                publish_raw_event(body.event)
                meta["path"] = "kafka"
                return {"status": "queued", "topic": settings.kafka_raw_topic}
            except Exception as exc:  # noqa: BLE001
                logger.warning("kafka_publish_failed_fallback_memory", error=str(exc))
                get_memory_bus().publish(settings.kafka_raw_topic, body.event)
        records = pipeline().process_event(body.event)
        meta["resources"] = len(records)
        return {
            "status": "processed",
            "resources": [
                {"resourceType": r.resource_type, "id": r.resource_id} for r in records
            ],
        }


# --- Secure FHIR-ish API ---


@app.get("/fhir/metadata")
def capability_statement(user: AuthUser) -> dict[str, Any]:
    return {
        "resourceType": "CapabilityStatement",
        "status": "active",
        "fhirVersion": "4.0.1",
        "format": ["json"],
        "rest": [
            {
                "mode": "server",
                "resource": [
                    {"type": "Patient", "interaction": [{"code": "read"}, {"code": "search-type"}]},
                    {"type": "Observation", "interaction": [{"code": "read"}, {"code": "search-type"}]},
                    {
                        "type": "DocumentReference",
                        "interaction": [{"code": "read"}, {"code": "search-type"}],
                    },
                    {"type": "Condition", "interaction": [{"code": "read"}, {"code": "search-type"}]},
                ],
            }
        ],
    }


@app.get("/fhir/{resource_type}/{resource_id}")
def fhir_read(resource_type: str, resource_id: str, user: AuthUser) -> dict[str, Any]:
    resource = get_resource(resource_type, resource_id)
    if resource is None:
        raise HTTPException(status_code=404, detail="Resource not found")
    return resource


@app.get("/fhir/{resource_type}")
def fhir_search(
    resource_type: str,
    user: AuthUser,
    patient: str | None = Query(default=None),
    encounter: str | None = Query(default=None),
    _count: int = Query(default=50, alias="_count"),
) -> dict[str, Any]:
    patient_id = patient
    if patient_id and not patient_id.startswith("Patient/"):
        patient_id = f"Patient/{patient_id}"
    encounter_id = encounter
    if encounter_id and not encounter_id.startswith("Encounter/"):
        encounter_id = f"Encounter/{encounter_id}"
    resources = search_resources(
        resource_type, patient_id=patient_id, encounter_id=encounter_id, limit=_count
    )
    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": len(resources),
        "entry": [{"resource": r} for r in resources],
    }


@app.get("/fhir/Patient/{patient_id}/$everything")
def fhir_everything(patient_id: str, user: AuthUser) -> dict[str, Any]:
    pid = patient_id if patient_id.startswith("Patient/") else f"Patient/{patient_id}"
    return patient_bundle(pid)


# --- IodineIQ ---


@app.post("/v1/iodineiq/analyze")
def iodineiq_analyze(body: SummaryRequest, user: AuthUser) -> dict[str, Any]:
    with track_request("iodineiq") as meta:
        result = iodine().analyze(body.patient_id, encounter_id=body.encounter_id)
        meta["priority"] = result.priority_score
        return result.model_dump()


@app.post("/v1/iodineiq/interact")
def iodineiq_interact(body: InteractRequest, user: AuthUser) -> dict[str, str]:
    answer = iodine().interact_query(body.question, body.patient_id)
    return {"answer": answer}


@app.post("/v1/iodineiq/train-risk-model")
def train_risk(user: AuthUser) -> dict[str, Any]:
    return train_and_persist()


# --- Summaries (GraphRAG) ---


@app.post("/v1/summaries/patient")
def patient_summary(body: SummaryRequest, user: AuthUser) -> dict[str, Any]:
    with track_request("patient_summary"):
        return summaries().patient_summary(body.patient_id).model_dump()


@app.post("/v1/summaries/insurance")
def insurance_summary(body: SummaryRequest, user: AuthUser) -> dict[str, Any]:
    with track_request("insurance_summary"):
        return summaries().insurance_summary(body.patient_id, body.encounter_id).model_dump()
