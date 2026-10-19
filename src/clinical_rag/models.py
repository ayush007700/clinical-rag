"""Domain models for clinical CDS / GraphRAG."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RiskTier(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class RawDeviceEvent(BaseModel):
    event_id: str
    device_id: str
    patient_mrn: str
    timestamp: datetime
    metric: str
    value: float
    unit: str | None = None
    ward: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class RawEmrChunk(BaseModel):
    event_id: str
    patient_mrn: str
    encounter_id: str | None = None
    source_system: str = "EMR"
    document_type: str
    recorded_at: datetime
    clinician: str | None = None
    text: str
    codes: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class FhirResourceRecord(BaseModel):
    resource_type: str
    resource_id: str
    patient_id: str | None = None
    encounter_id: str | None = None
    resource: dict[str, Any]
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


class CdiFlag(BaseModel):
    code: str
    severity: str
    message: str
    suggested_query: str | None = None


class DrgPrediction(BaseModel):
    drg_code: str
    description: str
    probability: float
    relative_weight: float | None = None


class MedicalCodeMatch(BaseModel):
    system: str
    code: str
    display: str
    confidence: float
    source: str = "iodineiq"


class RiskPrediction(BaseModel):
    patient_id: str
    risk_score: float
    risk_tier: RiskTier
    features: dict[str, float] = Field(default_factory=dict)
    intervention_hints: list[str] = Field(default_factory=list)


class IodineIQResult(BaseModel):
    patient_id: str
    encounter_id: str | None = None
    risk: RiskPrediction
    drg: list[DrgPrediction] = Field(default_factory=list)
    codes: list[MedicalCodeMatch] = Field(default_factory=list)
    cdi_flags: list[CdiFlag] = Field(default_factory=list)
    priority_score: float = 0.0
    cognitive_rationale: str = ""


class Citation(BaseModel):
    chunk_id: str
    source: str
    excerpt: str
    score: float = 0.0


class PatientSummary(BaseModel):
    patient_id: str
    summary: str
    risk: RiskPrediction | None = None
    codes: list[MedicalCodeMatch] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class InsuranceSummary(BaseModel):
    patient_id: str
    encounter_id: str | None = None
    claim_narrative: str
    principal_diagnosis: MedicalCodeMatch | None = None
    secondary_diagnoses: list[MedicalCodeMatch] = Field(default_factory=list)
    procedures: list[MedicalCodeMatch] = Field(default_factory=list)
    predicted_drg: DrgPrediction | None = None
    cdi_flags: list[CdiFlag] = Field(default_factory=list)
    supporting_evidence: list[Citation] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0
