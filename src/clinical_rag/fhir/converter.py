"""Clean raw EMR/device payloads and map them to FHIR R4 resources."""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Any

from clinical_rag.models import FhirResourceRecord


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _patient_id(mrn: str) -> str:
    return f"Patient/{mrn}"


def device_event_to_fhir(event: dict[str, Any]) -> list[dict[str, Any]]:
    mrn = str(event["patient_mrn"])
    obs_id = f"obs-{event.get('event_id', uuid.uuid4().hex[:12])}"
    ts = event.get("timestamp") or datetime.utcnow().isoformat()
    return [
        {
            "resourceType": "Observation",
            "id": obs_id,
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "vital-signs",
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": _loinc_for_metric(str(event.get("metric", "unknown"))),
                        "display": str(event.get("metric")),
                    }
                ],
                "text": str(event.get("metric")),
            },
            "subject": {"reference": _patient_id(mrn)},
            "effectiveDateTime": ts,
            "valueQuantity": {
                "value": float(event["value"]),
                "unit": event.get("unit") or "1",
            },
            "device": {"display": str(event.get("device_id"))},
            "meta": {"source": "kafka-device"},
        }
    ]


def emr_chunk_to_fhir(event: dict[str, Any]) -> list[dict[str, Any]]:
    mrn = str(event["patient_mrn"])
    note_id = f"note-{event.get('event_id', uuid.uuid4().hex[:12])}"
    text = clean_text(str(event.get("text", "")))
    resources: list[dict[str, Any]] = [
        {
            "resourceType": "DocumentReference",
            "id": note_id,
            "status": "current",
            "type": {"text": str(event.get("document_type", "clinical-note"))},
            "subject": {"reference": _patient_id(mrn)},
            "date": event.get("recorded_at") or datetime.utcnow().isoformat(),
            "author": [{"display": event.get("clinician") or "unknown"}],
            "content": [
                {
                    "attachment": {
                        "contentType": "text/plain",
                        "title": str(event.get("document_type", "note")),
                        "data_text": text,
                    }
                }
            ],
            "context": {
                "encounter": (
                    [{"reference": f"Encounter/{event['encounter_id']}"}]
                    if event.get("encounter_id")
                    else []
                )
            },
            "meta": {"source": event.get("source_system", "EMR")},
        }
    ]
    for code in event.get("codes") or []:
        cond_id = f"cond-{uuid.uuid4().hex[:10]}"
        resources.append(
            {
                "resourceType": "Condition",
                "id": cond_id,
                "clinicalStatus": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                            "code": "active",
                        }
                    ]
                },
                "code": {
                    "coding": [
                        {
                            "system": "http://hl7.org/fhir/sid/icd-10-cm",
                            "code": str(code),
                            "display": str(code),
                        }
                    ],
                    "text": str(code),
                },
                "subject": {"reference": _patient_id(mrn)},
                "encounter": (
                    {"reference": f"Encounter/{event['encounter_id']}"}
                    if event.get("encounter_id")
                    else None
                ),
            }
        )
    return resources


def to_fhir_resources(raw_event: dict[str, Any]) -> list[FhirResourceRecord]:
    kind = raw_event.get("event_kind") or ("device" if "metric" in raw_event else "emr")
    if kind == "device" or "metric" in raw_event:
        fhir_list = device_event_to_fhir(raw_event)
    else:
        fhir_list = emr_chunk_to_fhir(raw_event)

    patient_id = _patient_id(str(raw_event.get("patient_mrn", "unknown")))
    encounter_id = (
        f"Encounter/{raw_event['encounter_id']}" if raw_event.get("encounter_id") else None
    )
    out: list[FhirResourceRecord] = []
    for res in fhir_list:
        out.append(
            FhirResourceRecord(
                resource_type=res["resourceType"],
                resource_id=str(res["id"]),
                patient_id=patient_id,
                encounter_id=encounter_id,
                resource=res,
            )
        )
    return out


def ensure_patient_resource(mrn: str, name: str = "Demo Patient") -> dict[str, Any]:
    return {
        "resourceType": "Patient",
        "id": mrn,
        "identifier": [{"system": "urn:hospital:mrn", "value": mrn}],
        "name": [{"text": name}],
        "active": True,
    }


def _loinc_for_metric(metric: str) -> str:
    mapping = {
        "heart_rate": "8867-4",
        "spo2": "2708-6",
        "blood_pressure_systolic": "8480-6",
        "blood_pressure_diastolic": "8462-4",
        "temperature": "8310-5",
        "respiratory_rate": "9279-1",
    }
    return mapping.get(metric.lower(), "85353-1")
