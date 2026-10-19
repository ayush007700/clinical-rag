"""Patient & insurance summary orchestration."""

from __future__ import annotations

from clinical_rag.agent.graph import ClinicalGraphRAGAgent
from clinical_rag.iodineiq.engine import IodineIQEngine
from clinical_rag.models import (
    Citation,
    InsuranceSummary,
    MedicalCodeMatch,
    PatientSummary,
)


class SummaryService:
    def __init__(
        self,
        agent: ClinicalGraphRAGAgent | None = None,
        iodineiq: IodineIQEngine | None = None,
    ) -> None:
        self.iodineiq = iodineiq or IodineIQEngine()
        self.agent = agent or ClinicalGraphRAGAgent(iodineiq=self.iodineiq)

    def patient_summary(self, patient_id: str) -> PatientSummary:
        state = self.agent.run(patient_id=patient_id, mode="patient")
        iq = self.iodineiq.analyze(patient_id)
        return PatientSummary(
            patient_id=patient_id,
            summary=str(state.get("answer") or ""),
            risk=iq.risk,
            codes=iq.codes,
            citations=list(state.get("citations") or []),
        )

    def insurance_summary(
        self, patient_id: str, encounter_id: str | None = None
    ) -> InsuranceSummary:
        state = self.agent.run(
            patient_id=patient_id,
            mode="insurance",
            encounter_id=encounter_id,
        )
        iq = self.iodineiq.analyze(patient_id, encounter_id=encounter_id)
        principal = iq.codes[0] if iq.codes else None
        secondary = iq.codes[1:5] if len(iq.codes) > 1 else []
        procedures: list[MedicalCodeMatch] = [
            c for c in iq.codes if c.system.upper().startswith("CPT")
        ]
        return InsuranceSummary(
            patient_id=patient_id,
            encounter_id=encounter_id,
            claim_narrative=str(state.get("answer") or ""),
            principal_diagnosis=principal,
            secondary_diagnoses=secondary,
            procedures=procedures,
            predicted_drg=iq.drg[0] if iq.drg else None,
            cdi_flags=iq.cdi_flags,
            supporting_evidence=list(state.get("citations") or []),
        )
