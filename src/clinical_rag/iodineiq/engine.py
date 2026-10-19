"""IodineIQ Cognitive ML/AI engine — CDI, Forecast, Concurrent, Interact/Retrospect."""

from __future__ import annotations

import re
from typing import Any

from clinical_rag.fhir.store import iter_patient_resources
from clinical_rag.logging_setup import get_logger
from clinical_rag.ml.risk_model import PatientRiskModel
from clinical_rag.models import (
    CdiFlag,
    DrgPrediction,
    IodineIQResult,
    MedicalCodeMatch,
)

logger = get_logger(__name__)

# Lightweight clinical lexicon for demo coding / CDI (extend with enterprise terminologies)
ICD10_LEXICON: list[tuple[str, str, list[str]]] = [
    ("J18.9", "Pneumonia, unspecified organism", ["pneumonia", "infiltrate", "consolidation"]),
    ("I50.9", "Heart failure, unspecified", ["heart failure", "chf", "ef ", "edema"]),
    ("E11.9", "Type 2 diabetes mellitus without complications", ["diabetes", "dm2", "type 2"]),
    ("N18.3", "Chronic kidney disease, stage 3", ["ckd", "chronic kidney", "stage 3"]),
    ("J44.1", "COPD with acute exacerbation", ["copd", "exacerbation", "wheeze"]),
    ("A41.9", "Sepsis, unspecified organism", ["sepsis", "septic", "bacteremia"]),
    ("I21.9", "Acute myocardial infarction", ["mi ", "stemi", "nstemi", "myocardial infarction"]),
    ("J96.01", "Acute respiratory failure with hypoxia", ["respiratory failure", "hypoxia", "intubat"]),
]

DRG_MAP: list[tuple[str, str, float, list[str]]] = [
    ("193", "Simple pneumonia & pleurisy w MCC", 1.45, ["J18.9", "J96.01"]),
    ("291", "Heart failure & shock w MCC", 1.52, ["I50.9"]),
    ("871", "Septicemia or severe sepsis w MV >96h", 5.8, ["A41.9"]),
    ("280", "Acute myocardial infarction w MCC", 1.9, ["I21.9"]),
    ("190", "COPD w MCC", 1.2, ["J44.1"]),
]


class IodineIQEngine:
    """Cognitive ML engine modeled after Iodine Software Concurrent/Forecast/CDI modules."""

    def __init__(self, risk_model: PatientRiskModel | None = None) -> None:
        self.risk_model = risk_model or PatientRiskModel()

    def ingest_patient_corpus(self, patient_id: str) -> list[dict[str, Any]]:
        return list(iter_patient_resources(patient_id))

    def _clinical_text(self, resources: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for r in resources:
            if r.get("resourceType") == "DocumentReference":
                for c in r.get("content") or []:
                    att = c.get("attachment") or {}
                    if att.get("data_text"):
                        parts.append(str(att["data_text"]))
            if r.get("resourceType") == "Condition":
                parts.append(str((r.get("code") or {}).get("text") or ""))
        return "\n".join(parts).lower()

    def cognitive_emulation(self, text: str) -> list[MedicalCodeMatch]:
        """Mimic physician diagnostic perspective → candidate final medical codes."""
        matches: list[MedicalCodeMatch] = []
        for code, display, cues in ICD10_LEXICON:
            hits = [c for c in cues if c in text]
            if not hits:
                continue
            conf = min(0.55 + 0.12 * len(hits), 0.97)
            matches.append(
                MedicalCodeMatch(
                    system="ICD-10-CM",
                    code=code,
                    display=display,
                    confidence=round(conf, 3),
                    source="iodineiq-cognitive",
                )
            )
        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches[:8]

    def cdi_flags(self, text: str, codes: list[MedicalCodeMatch]) -> list[CdiFlag]:
        flags: list[CdiFlag] = []
        if "pneumonia" in text and not any(c.code.startswith("J18") for c in codes):
            flags.append(
                CdiFlag(
                    code="CDI-PNA-SPEC",
                    severity="high",
                    message="Pneumonia documented without specificity / organism",
                    suggested_query="Please clarify pneumonia type and causal organism if known.",
                )
            )
        if re.search(r"\bsepsis\b", text) and "organ dysfunction" not in text:
            flags.append(
                CdiFlag(
                    code="CDI-SEPSIS-OD",
                    severity="high",
                    message="Sepsis mentioned without explicit organ dysfunction linkage",
                    suggested_query="Document organ dysfunction related to sepsis if clinically appropriate.",
                )
            )
        if "heart failure" in text and "systolic" not in text and "diastolic" not in text:
            flags.append(
                CdiFlag(
                    code="CDI-HF-TYPE",
                    severity="medium",
                    message="Heart failure type (HFrEF/HFpEF) not specified",
                    suggested_query="Please specify HF type and acuity (acute/chronic).",
                )
            )
        if "hypoxia" in text and not any(c.code.startswith("J96") for c in codes):
            flags.append(
                CdiFlag(
                    code="CDI-ARF",
                    severity="medium",
                    message="Hypoxia noted; acute respiratory failure may be under-coded",
                    suggested_query="Clarify if acute respiratory failure with hypoxia is present.",
                )
            )
        if not flags and codes:
            flags.append(
                CdiFlag(
                    code="CDI-OK",
                    severity="low",
                    message="No major documentation gaps detected on heuristic scan",
                    suggested_query=None,
                )
            )
        return flags

    def forecast_drg(self, codes: list[MedicalCodeMatch]) -> list[DrgPrediction]:
        code_set = {c.code for c in codes}
        preds: list[DrgPrediction] = []
        for drg, desc, weight, need in DRG_MAP:
            overlap = len(code_set.intersection(need))
            if overlap == 0:
                continue
            prob = min(0.4 + 0.25 * overlap, 0.95)
            preds.append(
                DrgPrediction(
                    drg_code=drg,
                    description=desc,
                    probability=round(prob, 3),
                    relative_weight=weight,
                )
            )
        preds.sort(key=lambda d: d.probability, reverse=True)
        if not preds:
            preds.append(
                DrgPrediction(
                    drg_code="999",
                    description="Ungroupable / insufficient documentation",
                    probability=0.5,
                    relative_weight=None,
                )
            )
        return preds[:3]

    def concurrent_priority(self, risk_score: float, cdi_flags: list[CdiFlag]) -> float:
        severity_w = {"low": 0.05, "medium": 0.15, "high": 0.3}
        bump = sum(severity_w.get(f.severity, 0.1) for f in cdi_flags if f.code != "CDI-OK")
        return round(min(risk_score + bump, 1.0), 4)

    def interact_query(self, question: str, patient_id: str) -> str:
        """Physician query / post-bill (Retrospect) lightweight answer from chart + codes."""
        result = self.analyze(patient_id)
        q = question.lower()
        if "drg" in q or "group" in q:
            top = result.drg[0] if result.drg else None
            return (
                f"Forecast DRG {top.drg_code} ({top.description}) @ {top.probability:.0%}"
                if top
                else "Insufficient data for DRG forecast."
            )
        if "risk" in q or "deterior" in q:
            return (
                f"Risk score {result.risk.risk_score:.2f} ({result.risk.risk_tier.value}). "
                + "; ".join(result.risk.intervention_hints)
            )
        if "cdi" in q or "documentation" in q:
            return "; ".join(f.message for f in result.cdi_flags) or "No CDI flags."
        return result.cognitive_rationale or "No additional cognitive insight available."

    def analyze(self, patient_id: str, encounter_id: str | None = None, age: float = 65.0) -> IodineIQResult:
        resources = self.ingest_patient_corpus(patient_id)
        text = self._clinical_text(resources)
        codes = self.cognitive_emulation(text)
        # merge explicit Condition codes from FHIR
        for r in resources:
            if r.get("resourceType") != "Condition":
                continue
            coding = ((r.get("code") or {}).get("coding") or [{}])[0]
            code = coding.get("code")
            if not code:
                continue
            if any(c.code == code for c in codes):
                continue
            codes.append(
                MedicalCodeMatch(
                    system=coding.get("system") or "ICD-10-CM",
                    code=str(code),
                    display=str(coding.get("display") or code),
                    confidence=0.9,
                    source="fhir-condition",
                )
            )
        features = self.risk_model.extract_features(resources, age=age)
        risk = self.risk_model.predict(patient_id, features)
        flags = self.cdi_flags(text, codes)
        drg = self.forecast_drg(codes)
        priority = self.concurrent_priority(risk.risk_score, flags)
        rationale = (
            f"Cognitive review of {len(resources)} FHIR resources yields "
            f"{len(codes)} candidate codes; top DRG={(drg[0].drg_code if drg else 'n/a')}; "
            f"risk_tier={risk.risk_tier.value}."
        )
        logger.info(
            "iodineiq_analyze",
            patient_id=patient_id,
            codes=len(codes),
            priority=priority,
            risk=risk.risk_score,
        )
        return IodineIQResult(
            patient_id=patient_id,
            encounter_id=encounter_id,
            risk=risk,
            drg=drg,
            codes=codes,
            cdi_flags=flags,
            priority_score=priority,
            cognitive_rationale=rationale,
        )
