"""Seed demo patient chart + vitals through the FHIR pipeline and train risk model."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinical_rag.ml import train_and_persist
from clinical_rag.pipeline import FhirProcessingPipeline


def main() -> None:
    train_and_persist()
    pipe = FhirProcessingPipeline()
    mrn = "demo-001"
    enc = "enc-001"
    base = datetime.utcnow() - timedelta(hours=6)

    events = [
        {
            "event_kind": "emr",
            "event_id": "note-1",
            "patient_mrn": mrn,
            "patient_name": "Alex Demo",
            "encounter_id": enc,
            "document_type": "H&P",
            "recorded_at": base.isoformat(),
            "clinician": "Dr. Patel",
            "text": (
                "68M with COPD admitted for dyspnea and hypoxia. "
                "Chest x-ray with right lower lobe infiltrate concerning for pneumonia. "
                "History of type 2 diabetes and CKD stage 3. "
                "Started on oxygen and empiric antibiotics. "
                "Risk of clinical deterioration; monitor for sepsis."
            ),
            "codes": ["J44.1", "E11.9"],
        },
        {
            "event_kind": "emr",
            "event_id": "note-2",
            "patient_mrn": mrn,
            "encounter_id": enc,
            "document_type": "progress-note",
            "recorded_at": (base + timedelta(hours=3)).isoformat(),
            "clinician": "Dr. Patel",
            "text": (
                "Persistent hypoxia SpO2 88% on room air improving to 93% on 2L NC. "
                "Leukocytosis. Heart failure markers pending. "
                "Consider CDI clarification for pneumonia specificity."
            ),
            "codes": [],
        },
    ]

    metrics = [
        ("heart_rate", 112, "beats/min"),
        ("spo2", 88, "%"),
        ("respiratory_rate", 26, "/min"),
        ("temperature", 38.6, "C"),
        ("blood_pressure_systolic", 98, "mmHg"),
    ]
    for i, (metric, value, unit) in enumerate(metrics):
        events.append(
            {
                "event_kind": "device",
                "event_id": f"vit-{i}",
                "device_id": "monitor-icu-7",
                "patient_mrn": mrn,
                "timestamp": (base + timedelta(minutes=10 * i)).isoformat(),
                "metric": metric,
                "value": value,
                "unit": unit,
                "ward": "ICU",
            }
        )

    for ev in events:
        pipe.process_event(ev)

    print(f"Seeded Patient/{mrn} with {len(events)} events.")
    print("Try: POST /v1/summaries/patient with {\"patient_id\":\"Patient/demo-001\"}")


if __name__ == "__main__":
    main()
