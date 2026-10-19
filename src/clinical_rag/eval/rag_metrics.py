"""Tiny offline RAG evaluation metrics (retrieval + citations + refuse)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)


def _norm(text: str) -> str:
    return (text or "").lower()


def term_recall(must_terms: Sequence[str], corpus: str) -> float:
    if not must_terms:
        return 1.0
    blob = _norm(corpus)
    hits = sum(1 for t in must_terms if t.lower() in blob)
    return hits / len(must_terms)


def evaluate_case(
    case: dict[str, Any],
    *,
    answer: str,
    retrieved_texts: Sequence[str],
    citation_ids: Sequence[str],
    refused: bool,
) -> CaseResult:
    case_id = str(case.get("id") or "unknown")
    failures: list[str] = []
    metrics: dict[str, Any] = {"refused": refused}

    expect_refuse = bool(case.get("expect_refuse", False))
    if expect_refuse:
        metrics["refuse_ok"] = refused
        if not refused:
            failures.append("expected_refuse")
        return CaseResult(case_id=case_id, passed=len(failures) == 0, metrics=metrics, failures=failures)

    if refused:
        failures.append("unexpected_refuse")
        metrics["refuse_ok"] = False
        return CaseResult(case_id=case_id, passed=False, metrics=metrics, failures=failures)

    retrieved_blob = "\n".join(retrieved_texts)
    ret_terms = list(case.get("must_retrieve_terms") or [])
    ans_terms = list(case.get("must_answer_terms") or [])
    metrics["retrieval_term_recall"] = term_recall(ret_terms, retrieved_blob)
    metrics["answer_term_coverage"] = term_recall(ans_terms, answer)
    metrics["citation_count"] = len(citation_ids)
    metrics["has_citations"] = len(citation_ids) > 0

    min_ret = float(case.get("min_retrieval_term_recall", 0.5))
    min_ans = float(case.get("min_answer_term_coverage", 0.5))
    if metrics["retrieval_term_recall"] < min_ret:
        failures.append(
            f"retrieval_term_recall {metrics['retrieval_term_recall']:.2f} < {min_ret}"
        )
    if metrics["answer_term_coverage"] < min_ans:
        failures.append(
            f"answer_term_coverage {metrics['answer_term_coverage']:.2f} < {min_ans}"
        )
    if case.get("require_citations", True) and not citation_ids:
        failures.append("missing_citations")

    return CaseResult(
        case_id=case_id,
        passed=len(failures) == 0,
        metrics=metrics,
        failures=failures,
    )
