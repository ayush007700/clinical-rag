"""Core unit tests — FHIR conversion, IodineIQ, summaries (no external deps required)."""

from __future__ import annotations

from datetime import datetime

from clinical_rag.fhir.converter import to_fhir_resources
from clinical_rag.iodineiq import IodineIQEngine
from clinical_rag.ml.risk_model import PatientRiskModel, train_and_persist
from clinical_rag.pipeline import FhirProcessingPipeline
from clinical_rag.retrieval.embeddings import LocalHashEmbedder, cosine_similarity_matrix
from clinical_rag.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from clinical_rag.summaries import SummaryService


def test_device_to_fhir_observation():
    resources = to_fhir_resources(
        {
            "event_kind": "device",
            "event_id": "e1",
            "device_id": "dev-1",
            "patient_mrn": "p1",
            "timestamp": datetime.utcnow().isoformat(),
            "metric": "heart_rate",
            "value": 110,
            "unit": "beats/min",
        }
    )
    assert resources[0].resource_type == "Observation"
    assert resources[0].resource["valueQuantity"]["value"] == 110


def test_emr_to_document_and_condition():
    resources = to_fhir_resources(
        {
            "event_kind": "emr",
            "event_id": "n1",
            "patient_mrn": "p1",
            "encounter_id": "enc1",
            "document_type": "progress",
            "recorded_at": datetime.utcnow().isoformat(),
            "text": "Patient with pneumonia and hypoxia.",
            "codes": ["J18.9"],
        }
    )
    types = {r.resource_type for r in resources}
    assert "DocumentReference" in types
    assert "Condition" in types


def test_risk_model_train_and_predict(tmp_path):
    path = tmp_path / "risk.joblib"
    stats = train_and_persist(str(path), n=200)
    assert stats["accuracy"] >= 0.5
    model = PatientRiskModel(str(path))
    pred = model.predict(
        "Patient/p1",
        {
            "age": 70,
            "heart_rate_mean": 120,
            "spo2_min": 88,
            "sys_bp_mean": 95,
            "temp_max": 38.5,
            "resp_rate_mean": 28,
            "num_conditions": 4,
            "los_days": 3,
            "prior_admissions": 2,
            "lab_abnormal_count": 5,
        },
    )
    assert 0.0 <= pred.risk_score <= 1.0


def test_end_to_end_pipeline_and_summaries(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'fhir.db'}")
    monkeypatch.setenv("RISK_MODEL_PATH", str(tmp_path / "risk.joblib"))
    from clinical_rag.config import get_settings

    get_settings.cache_clear()

    pipe = FhirProcessingPipeline()
    pipe.process_event(
        {
            "event_kind": "emr",
            "event_id": "seed",
            "patient_mrn": "demo-001",
            "encounter_id": "enc-001",
            "document_type": "H&P",
            "recorded_at": datetime.utcnow().isoformat(),
            "text": "COPD exacerbation with pneumonia and hypoxia. Type 2 diabetes.",
            "codes": ["J44.1"],
        }
    )
    pipe.process_event(
        {
            "event_kind": "device",
            "event_id": "v1",
            "device_id": "m1",
            "patient_mrn": "demo-001",
            "timestamp": datetime.utcnow().isoformat(),
            "metric": "spo2",
            "value": 89,
            "unit": "%",
        }
    )

    iq = IodineIQEngine()
    result = iq.analyze("Patient/demo-001")
    assert result.codes
    assert result.risk.risk_score >= 0

    svc = SummaryService(iodineiq=iq)
    # Share in-memory graph from ingest when Neo4j is unavailable in tests
    svc.agent.retriever.store = pipe.graph
    patient = svc.patient_summary("Patient/demo-001")
    assert "Patient/demo-001" in patient.summary or patient.summary
    assert patient.citations or "Insufficient" in patient.summary
    insurance = svc.insurance_summary("Patient/demo-001", "Encounter/enc-001")
    assert insurance.claim_narrative
    assert insurance.predicted_drg is not None

    get_settings.cache_clear()


def test_grounding_refuses_without_retrieval(monkeypatch):
    monkeypatch.setenv("GROUNDING_REQUIRE_RETRIEVAL", "true")
    monkeypatch.setenv("GROUNDING_MIN_CHUNKS", "1")
    monkeypatch.setenv("USE_PGVECTOR", "false")
    from clinical_rag.agent.graph import REFUSAL_NO_EVIDENCE, ClinicalGraphRAGAgent
    from clinical_rag.config import get_settings
    from clinical_rag.retrieval.pgvector_store import reset_pgvector_state

    get_settings.cache_clear()
    reset_pgvector_state()

    class _EmptyStore:
        def fetch_patient_chunks(self, patient_id: str, limit: int = 50):
            return []

        def neighborhood(self, patient_id: str):
            return {"codes": [], "chunks": []}

        def link_code(self, *args, **kwargs):
            return None

    agent = ClinicalGraphRAGAgent()
    agent.retriever.store = _EmptyStore()
    state = agent.run(patient_id="Patient/empty", mode="patient")
    assert state.get("refused") is True
    assert "cite-or-refuse" in (state.get("answer") or "").lower() or REFUSAL_NO_EVIDENCE[:20] in (
        state.get("answer") or ""
    )
    assert not state.get("citations")
    get_settings.cache_clear()
    reset_pgvector_state()


def test_eval_metrics_refuse_and_terms():
    from clinical_rag.eval.rag_metrics import evaluate_case

    refuse = evaluate_case(
        {"id": "r1", "expect_refuse": True},
        answer="",
        retrieved_texts=[],
        citation_ids=[],
        refused=True,
    )
    assert refuse.passed

    ok = evaluate_case(
        {
            "id": "c1",
            "must_retrieve_terms": ["pneumonia"],
            "must_answer_terms": ["pneumonia"],
            "require_citations": True,
        },
        answer="Patient has pneumonia [DocumentReference/note-1].",
        retrieved_texts=["Admitted with pneumonia and hypoxia"],
        citation_ids=["DocumentReference/note-1"],
        refused=False,
    )
    assert ok.passed


def test_reciprocal_rank_fusion_prefers_consensus():
    # doc 1 ranks high in both lists → should beat doc 0 (high in one only)
    fused = reciprocal_rank_fusion([[0, 1, 2], [1, 2, 0]], k=60)
    assert fused[0][0] == 1


def test_local_dense_embeddings_cosine():
    embedder = LocalHashEmbedder(dim=128)
    docs = embedder.embed_documents(
        [
            "pneumonia with hypoxia and infiltrate",
            "routine orthopedic follow up appointment",
        ]
    )
    query = embedder.embed_query("pneumonia hypoxia respiratory")
    sims = cosine_similarity_matrix(query, docs)
    assert sims[0] > sims[1]


def test_hybrid_bm25_dense_patient_scoped(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'fhir.db'}")
    monkeypatch.setenv("RETRIEVAL_MODE", "hybrid")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("USE_PGVECTOR", "false")
    from clinical_rag.config import get_settings
    from clinical_rag.retrieval.embeddings import get_embedding_client
    from clinical_rag.retrieval.pgvector_store import reset_pgvector_state

    get_settings.cache_clear()
    get_embedding_client.cache_clear()
    reset_pgvector_state()

    store_chunks = [
        {
            "chunk_id": "c1",
            "text": "Acute pneumonia with hypoxia requiring oxygen",
            "patient_id": "Patient/demo-001",
            "source": "H&P",
        },
        {
            "chunk_id": "c2",
            "text": "Patient enjoys gardening on weekends",
            "patient_id": "Patient/demo-001",
            "source": "social",
        },
        {
            "chunk_id": "c-other",
            "text": "Acute pneumonia with hypoxia requiring oxygen",
            "patient_id": "Patient/other",
            "source": "H&P",
        },
    ]

    class _MemStore:
        def fetch_patient_chunks(self, patient_id: str, limit: int = 50):
            return [c for c in store_chunks if c["patient_id"] == patient_id][:limit]

        def neighborhood(self, patient_id: str):
            return {"codes": [], "chunks": self.fetch_patient_chunks(patient_id)}

    retriever = HybridRetriever(store=_MemStore(), embedder=LocalHashEmbedder(dim=128))
    hits = retriever.retrieve("pneumonia hypoxia oxygen", "Patient/demo-001", top_k=2)
    assert hits
    assert all(h.metadata.get("patient_id") == "Patient/demo-001" for h in hits)
    assert hits[0].metadata.get("retrieval") == "hybrid_rrf"
    assert "c-other" not in {h.chunk_id for h in hits}
    by_id = {h.chunk_id: h for h in hits}
    assert "c1" in by_id
    # Keyword BM25 should keep clinical note ahead of unrelated social text
    assert by_id["c1"].score >= by_id.get("c2", by_id["c1"]).score or "pneumonia" in hits[0].text.lower()

    get_settings.cache_clear()
    get_embedding_client.cache_clear()
    reset_pgvector_state()


def test_pgvector_store_roundtrip_optional():
    """Against Docker pgvector when available; otherwise no-op."""
    from clinical_rag.config import get_settings
    from clinical_rag.retrieval.embeddings import LocalHashEmbedder
    from clinical_rag.retrieval.pgvector_store import (
        PgVectorStore,
        pgvector_available,
        reset_pgvector_state,
    )

    reset_pgvector_state()
    get_settings.cache_clear()
    settings = get_settings()
    if not settings.database_url.startswith("postgresql"):
        return
    if not pgvector_available():
        return

    dim = settings.embedding_dim
    embedder = LocalHashEmbedder(dim=dim)
    store = PgVectorStore()
    text = "Pneumonia with hypoxia and oxygen therapy"
    emb = embedder.embed_documents([text])[0]
    store.upsert_chunk(
        chunk_id="DocumentReference/test-pgv",
        patient_id="Patient/pgv-test",
        text=text,
        embedding=emb,
        source="unit-test",
        provider=embedder.provider_name,
    )
    hits = store.search(
        patient_id="Patient/pgv-test",
        query_embedding=embedder.embed_query("pneumonia hypoxia"),
        top_k=3,
    )
    assert hits
    assert hits[0]["chunk_id"] == "DocumentReference/test-pgv"
    reset_pgvector_state()
    get_settings.cache_clear()
