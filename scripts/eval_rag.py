"""Offline RAG eval harness — retrieval recall, citations, cite-or-refuse.

Usage (from repo root, venv active, after seed):

    python scripts/seed_demo.py
    python scripts/eval_rag.py
    python scripts/eval_rag.py --golden data/eval/rag_golden.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinical_rag.agent.graph import ClinicalGraphRAGAgent  # noqa: E402
from clinical_rag.eval.rag_metrics import evaluate_case  # noqa: E402
from clinical_rag.logging_setup import configure_logging, get_logger  # noqa: E402

configure_logging()
logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="ClinicalRAG tiny GraphRAG eval")
    parser.add_argument(
        "--golden",
        type=Path,
        default=ROOT / "data" / "eval" / "rag_golden.json",
        help="Path to golden JSON",
    )
    args = parser.parse_args()
    payload = json.loads(args.golden.read_text(encoding="utf-8"))
    cases = payload.get("cases") or []
    agent = ClinicalGraphRAGAgent()

    results = []
    for case in cases:
        state = agent.run(
            patient_id=str(case["patient_id"]),
            mode=str(case.get("mode") or "patient"),
            encounter_id=case.get("encounter_id"),
            query=case.get("query"),
        )
        retrieved = state.get("retrieved") or []
        citations = state.get("citations") or []
        result = evaluate_case(
            case,
            answer=str(state.get("answer") or ""),
            retrieved_texts=[c.text for c in retrieved],
            citation_ids=[c.chunk_id for c in citations],
            refused=bool(state.get("refused")),
        )
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.case_id} metrics={result.metrics} failures={result.failures}")

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\nSummary: {passed}/{total} passed")
    logger.info("eval_rag_done", passed=passed, total=total)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
