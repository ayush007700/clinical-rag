"""Publish synthetic high-volume device events to Kafka (or in-process memory)."""

from __future__ import annotations

import argparse
import random
import sys
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinical_rag.config import get_settings
from clinical_rag.kafka import get_memory_bus, publish_raw_event
from clinical_rag.pipeline import FhirProcessingPipeline


METRICS = [
    ("heart_rate", 60, 140, "beats/min"),
    ("spo2", 85, 100, "%"),
    ("respiratory_rate", 12, 32, "/min"),
    ("temperature", 36.0, 39.5, "C"),
    ("blood_pressure_systolic", 90, 170, "mmHg"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--patient", default="demo-001")
    parser.add_argument("--kafka", action="store_true")
    parser.add_argument("--process-local", action="store_true", help="Also run FHIR pipeline locally")
    args = parser.parse_args()

    settings = get_settings()
    pipe = FhirProcessingPipeline() if args.process_local else None
    rng = random.Random(7)

    for i in range(args.n):
        metric, lo, hi, unit = rng.choice(METRICS)
        event = {
            "event_kind": "device",
            "event_id": uuid.uuid4().hex,
            "device_id": f"monitor-{rng.randint(1, 20)}",
            "patient_mrn": args.patient,
            "timestamp": datetime.utcnow().isoformat(),
            "metric": metric,
            "value": round(rng.uniform(lo, hi), 1),
            "unit": unit,
            "ward": rng.choice(["ICU", "Stepdown", "MedSurg"]),
        }
        if args.kafka:
            try:
                publish_raw_event(event)
            except Exception:
                get_memory_bus().publish(settings.kafka_raw_topic, event)
        elif pipe:
            pipe.process_event(event)
        else:
            get_memory_bus().publish(settings.kafka_raw_topic, event)

    print(f"Produced {args.n} device events for patient {args.patient}")


if __name__ == "__main__":
    main()
