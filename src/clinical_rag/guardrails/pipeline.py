"""Clinical safety / privacy / grounding guardrails."""

from __future__ import annotations

import re
from typing import Any, Sequence

from clinical_rag.config import get_settings
from clinical_rag.models import Citation, RetrievedChunk

BLOCKED_INPUT = [
    r"ignore previous instructions",
    r"exfiltrate",
    r"dump all patients",
]

BLOCKED_OUTPUT = [
    r"\bssn\b",
    r"social security",
]

SPECULATIVE = [
    r"\bpossibly invent",
    r"\bas an ai\b",
    r"\bi made up\b",
]


class GuardrailPipeline:
    def check_input(self, text: str) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        lowered = (text or "").lower()
        if len(lowered.strip()) < 3:
            reasons.append("query_too_short")
        for pat in BLOCKED_INPUT:
            if re.search(pat, lowered):
                reasons.append(f"blocked_input:{pat}")
        return (len(reasons) == 0, reasons)

    def check_output(self, text: str) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        lowered = (text or "").lower()
        for pat in BLOCKED_OUTPUT:
            if re.search(pat, lowered):
                reasons.append(f"blocked_output:{pat}")
        for pat in SPECULATIVE:
            if re.search(pat, lowered):
                reasons.append(f"blocked_speculative:{pat}")
        return (len(reasons) == 0, reasons)

    def check_retrieval_evidence(
        self,
        retrieved: Sequence[RetrievedChunk] | None,
        iodineiq: dict[str, Any] | None = None,
    ) -> tuple[bool, list[str]]:
        """Cite-or-refuse gate: require enough patient-scoped chart evidence."""
        settings = get_settings()
        reasons: list[str] = []
        chunks = list(retrieved or [])
        if not settings.grounding_require_retrieval:
            return True, reasons

        min_chunks = max(int(settings.grounding_min_chunks), 0)
        if len(chunks) < min_chunks:
            reasons.append(f"insufficient_retrieval:{len(chunks)}<{min_chunks}")

        # Soft signal: CDS codes alone are not substitute for note evidence when required
        if min_chunks > 0 and not chunks:
            codes = (iodineiq or {}).get("codes") or []
            if codes:
                reasons.append("codes_without_note_evidence")

        return (len(reasons) == 0, reasons)

    def check_grounding(
        self,
        answer: str,
        retrieved: Sequence[RetrievedChunk] | None,
        citations: Sequence[Citation] | None,
        *,
        refused: bool = False,
    ) -> tuple[bool, list[str]]:
        """Post-generation: citations required; refuse hallucinated empty narratives."""
        settings = get_settings()
        reasons: list[str] = []
        if refused:
            return True, reasons

        chunks = list(retrieved or [])
        cites = list(citations or [])

        if settings.grounding_require_retrieval and not chunks:
            reasons.append("answer_without_retrieval")

        if settings.grounding_require_citations and chunks and not cites:
            reasons.append("missing_citations")

        if chunks and cites:
            allowed = {c.chunk_id for c in chunks}
            unknown = [c.chunk_id for c in cites if c.chunk_id not in allowed]
            if unknown:
                reasons.append(f"citation_not_in_retrieval:{unknown[:3]}")

        text = (answer or "").strip()
        if not text or len(text) < 20:
            reasons.append("answer_too_short")

        # Lightweight lexical grounding: at least one content token from evidence in answer
        if chunks and text and settings.grounding_require_retrieval:
            evidence = " ".join(c.text for c in chunks).lower()
            answer_l = text.lower()
            tokens = {t for t in re.findall(r"[a-z]{4,}", evidence) if t not in _STOP}
            overlap = sum(1 for t in list(tokens)[:80] if t in answer_l)
            if tokens and overlap == 0:
                reasons.append("no_lexical_overlap_with_evidence")

        return (len(reasons) == 0, reasons)


_STOP = {
    "that",
    "with",
    "from",
    "this",
    "have",
    "been",
    "were",
    "their",
    "about",
    "would",
    "there",
    "which",
    "patient",
    "clinical",
    "summary",
    "documentation",
}
