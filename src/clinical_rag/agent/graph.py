"""LangGraph multi-agent clinical GraphRAG for medical summarization."""

from __future__ import annotations

import uuid
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from clinical_rag.config import get_settings
from clinical_rag.guardrails.pipeline import GuardrailPipeline
from clinical_rag.iodineiq.engine import IodineIQEngine
from clinical_rag.logging_setup import get_logger
from clinical_rag.models import Citation, RetrievedChunk
from clinical_rag.retrieval.hybrid import HybridRetriever

logger = get_logger(__name__)

REFUSAL_NO_EVIDENCE = (
    "Insufficient patient-scoped chart evidence to generate a grounded summary. "
    "Ingest clinical notes for this patient, then retry. (cite-or-refuse)"
)


class GraphState(TypedDict, total=False):
    patient_id: str
    encounter_id: str | None
    mode: str  # patient | insurance
    query: str
    allowed: bool
    guardrail_reasons: list[str]
    retrieved: list[RetrievedChunk]
    graph_facts: dict[str, Any]
    iodineiq: dict[str, Any]
    answer: str
    citations: list[Citation]
    refused: bool
    grounding_ok: bool
    trace_id: str


PATIENT_SYSTEM = """You are ClinicalRAG, a hospital clinical decision-support summarizer.
Write a concise clinician-facing patient summary using ONLY the provided evidence.
Include: presentation, key findings, active problems/codes, risk highlights, next actions.
Do not invent labs, vitals, or diagnoses. If evidence is thin, say what is missing.
Every factual clinical statement must be supportable by a cited chunk id like [DocumentReference/...].
This is decision support — not a substitute for clinical judgment.
"""

INSURANCE_SYSTEM = """You are ClinicalRAG preparing an insurance submission narrative for the patient/payer.
Produce a clear claim narrative: principal diagnosis, secondary diagnoses, procedures if any,
clinical course supporting medical necessity, and coding rationale.
Use only provided evidence and CDS coding/DRG hints. Avoid speculative statements.
Cite supporting chunk ids like [DocumentReference/...] for clinical course claims.
"""


class ClinicalGraphRAGAgent:
    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        iodineiq: IodineIQEngine | None = None,
    ) -> None:
        self.settings = get_settings()
        self.retriever = retriever or HybridRetriever()
        self.iodineiq = iodineiq or IodineIQEngine()
        self.guardrails = GuardrailPipeline()
        self.graph = self._build_graph()

    def _build_graph(self):
        g = StateGraph(GraphState)
        g.add_node("guard_input", self.guard_input)
        g.add_node("enrich_iodineiq", self.enrich_iodineiq)
        g.add_node("retrieve", self.retrieve)
        g.add_node("check_evidence", self.check_evidence)
        g.add_node("generate", self.generate)
        g.add_node("guard_output", self.guard_output)

        g.set_entry_point("guard_input")
        g.add_conditional_edges(
            "guard_input",
            lambda s: "enrich_iodineiq" if s.get("allowed") else "end",
            {"enrich_iodineiq": "enrich_iodineiq", "end": END},
        )
        g.add_edge("enrich_iodineiq", "retrieve")
        g.add_edge("retrieve", "check_evidence")
        g.add_conditional_edges(
            "check_evidence",
            lambda s: "generate" if not s.get("refused") else "end",
            {"generate": "generate", "end": END},
        )
        g.add_edge("generate", "guard_output")
        g.add_edge("guard_output", END)
        return g.compile()

    def guard_input(self, state: GraphState) -> GraphState:
        ok, reasons = self.guardrails.check_input(state.get("query", ""))
        state["allowed"] = ok
        state["guardrail_reasons"] = reasons
        state["refused"] = not ok
        if not ok:
            state["answer"] = "Request blocked by clinical safety guardrails."
            state["citations"] = []
        return state

    def enrich_iodineiq(self, state: GraphState) -> GraphState:
        result = self.iodineiq.analyze(
            state["patient_id"],
            encounter_id=state.get("encounter_id"),
        )
        state["iodineiq"] = result.model_dump()
        for code in result.codes:
            self.retriever.store.link_code(
                state["patient_id"], code.code, code.display, code.confidence
            )
        return state

    def retrieve(self, state: GraphState) -> GraphState:
        state["retrieved"] = self.retriever.retrieve(
            state.get("query") or "clinical summary",
            state["patient_id"],
            top_k=8,
        )
        state["graph_facts"] = self.retriever.graph_context(state["patient_id"])
        return state

    def check_evidence(self, state: GraphState) -> GraphState:
        ok, reasons = self.guardrails.check_retrieval_evidence(
            state.get("retrieved"),
            state.get("iodineiq"),
        )
        if not ok:
            state["refused"] = True
            state["grounding_ok"] = False
            state["answer"] = REFUSAL_NO_EVIDENCE
            state["citations"] = []
            state["guardrail_reasons"] = (state.get("guardrail_reasons") or []) + reasons
            logger.info(
                "grounding_refuse_empty_retrieval",
                patient_id=state.get("patient_id"),
                reasons=reasons,
                trace_id=state.get("trace_id"),
            )
        else:
            state["grounding_ok"] = True
        return state

    def _format_context(self, state: GraphState) -> str:
        blocks = []
        for item in state.get("retrieved") or []:
            blocks.append(f"[{item.chunk_id}] ({item.metadata.get('source')})\n{item.text}")
        iq = state.get("iodineiq") or {}
        blocks.append("CDS cognitive codes:\n" + str(iq.get("codes")))
        blocks.append("CDS DRG forecast:\n" + str(iq.get("drg")))
        blocks.append("CDS risk:\n" + str(iq.get("risk")))
        blocks.append("Graph neighborhood:\n" + str(state.get("graph_facts")))
        return "\n\n---\n\n".join(blocks)

    def generate(self, state: GraphState) -> GraphState:
        if state.get("refused"):
            return state
        mode = state.get("mode") or "patient"
        system = INSURANCE_SYSTEM if mode == "insurance" else PATIENT_SYSTEM
        context = self._format_context(state)
        answer = self._llm_or_template(system, state.get("query", ""), context, state)
        citations = [
            Citation(
                chunk_id=c.chunk_id,
                source=str(c.metadata.get("source") or "chart"),
                excerpt=c.text[:280],
                score=c.score,
            )
            for c in (state.get("retrieved") or [])[:5]
        ]
        state["answer"] = answer
        state["citations"] = citations
        return state

    def _llm_or_template(
        self, system: str, query: str, context: str, state: GraphState
    ) -> str:
        retrieved = state.get("retrieved") or []
        if not retrieved and self.settings.grounding_require_retrieval:
            return REFUSAL_NO_EVIDENCE

        if self.settings.openai_api_key:
            try:
                from langchain_openai import ChatOpenAI
                from langchain_core.messages import HumanMessage, SystemMessage

                llm = ChatOpenAI(
                    model=self.settings.openai_chat_model,
                    temperature=0.1,
                    api_key=self.settings.openai_api_key,
                    max_tokens=self.settings.summary_max_tokens,
                )
                msg = llm.invoke(
                    [
                        SystemMessage(content=system),
                        HumanMessage(
                            content=(
                                f"Query: {query}\n\nEvidence:\n{context}\n\n"
                                "Write the summary. Cite chunk ids in brackets."
                            )
                        ),
                    ]
                )
                return str(msg.content)
            except Exception as exc:  # noqa: BLE001
                logger.warning("llm_fallback_template", error=str(exc))

        iq = state.get("iodineiq") or {}
        risk = iq.get("risk") or {}
        codes = iq.get("codes") or []
        drg = (iq.get("drg") or [{}])[0]
        mode = state.get("mode") or "patient"
        cite_ids = ", ".join(f"[{c.chunk_id}]" for c in retrieved[:3])
        snippets = " ".join((c.text[:120] for c in retrieved[:3]))
        code_line = ", ".join(f"{c.get('code')} ({c.get('display')})" for c in codes[:5]) or "n/a"
        if mode == "insurance":
            return (
                f"Insurance clinical narrative for {state['patient_id']}. "
                f"Principal/candidate diagnoses: {code_line}. "
                f"Predicted DRG {drg.get('drg_code')} — {drg.get('description')} "
                f"(p={drg.get('probability')}). "
                f"Medical necessity is supported by chart evidence {cite_ids}: {snippets}. "
                f"CDI items should be resolved prior to final bill submission."
            )
        return (
            f"Patient summary for {state['patient_id']}. "
            f"Risk {risk.get('risk_score')} ({risk.get('risk_tier')}). "
            f"Active coding candidates: {code_line}. "
            f"Interventions: {'; '.join((risk.get('intervention_hints') or [])[:3])}. "
            f"Chart highlights {cite_ids}: {snippets}."
        )

    def guard_output(self, state: GraphState) -> GraphState:
        if state.get("refused"):
            return state
        ok_out, reasons_out = self.guardrails.check_output(state.get("answer", ""))
        ok_g, reasons_g = self.guardrails.check_grounding(
            state.get("answer", ""),
            state.get("retrieved"),
            state.get("citations"),
            refused=bool(state.get("refused")),
        )
        reasons = reasons_out + reasons_g
        state["grounding_ok"] = ok_g and ok_out
        if not ok_out or not ok_g:
            state["refused"] = True
            joined = "; ".join(reasons)
            state["answer"] = f"Summary suppressed by grounding/safety guardrails ({joined})."
            state["citations"] = []
            state["guardrail_reasons"] = (state.get("guardrail_reasons") or []) + reasons
            logger.info(
                "grounding_refuse_output",
                reasons=reasons,
                trace_id=state.get("trace_id"),
            )
        return state

    def run(
        self,
        patient_id: str,
        mode: str = "patient",
        encounter_id: str | None = None,
        query: str | None = None,
    ) -> GraphState:
        default_q = (
            "Generate insurance claim clinical summary"
            if mode == "insurance"
            else "Generate clinician patient summary"
        )
        init: GraphState = {
            "patient_id": patient_id,
            "encounter_id": encounter_id,
            "mode": mode,
            "query": query or default_q,
            "trace_id": uuid.uuid4().hex,
        }
        return self.graph.invoke(init)
