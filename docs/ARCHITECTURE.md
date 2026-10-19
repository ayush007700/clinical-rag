# Architecture — ClinicalRAG CDS

## Goals (resume-aligned)

1. **Predictive CDS** — patient risk scoring and intervention hints (Scikit-learn).
2. **CDI / coding accuracy** — classical ML signals plus deterministic rule engines for documentation gaps and DRG/coding candidates.
3. **GraphRAG on EHR** — Neo4j knowledge graph + LangGraph for grounded, citeable provider and payer summarization.
4. **Data isolation** — all retrieval and generation scoped by `patient_id` (no cross-patient corpus search).
5. **Operational capacity** — concurrent `priority_score` ranks charts for review (risk + CDI severity).
6. **Scale ingest** — Kafka → FHIR R4 → Postgres + graph indexing.

## Data flow

1. **Kafka** accepts raw device metrics and EMR note payloads on `clinical.raw.events`.
2. **Worker** (`clinical_rag.workers.kafka_worker`) cleans payloads and maps to FHIR R4.
3. Resources persist in **Postgres** (`fhir_resources`); note text is chunked into **Neo4j** under the patient node.
4. **Secure FHIR API** exposes read/search/`$everything` behind JWT / demo bearer auth.
5. **CDS engine** loads the patient FHIR corpus → cognitive coding rules, CDI flags, DRG forecast, sklearn risk, concurrent priority.
6. **LangGraph GraphRAG** retrieves **only that patient’s** chunks via **hybrid BM25 + pgvector dense (RRF)** (in-memory dense fallback if pgvector off), plus graph neighborhood + CDS outputs, then generates summaries (LLM if configured, evidence-tied templates otherwise).

## ML + rules vs GraphRAG (boundary)

```text
                    ┌─────────────────────────────┐
   FHIR corpus ────►│  CDS: risk ML + CDI/DRG rules│──► scores, flags, codes
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
   Neo4j chunks ──►│  GraphRAG: retrieve + generate│──► citeable narratives
                    │  (patient_id filter required) │
                    └─────────────────────────────┘
```

- **Do not** use the LLM to invent risk scores or ICD/DRG codes.
- **Do not** retrieve outside the patient scope.
- **Do** pass CDS structured outputs into the generator as hints, with chart citations as evidence.

## LangGraph stages

`guard_input → enrich_cds → retrieve → check_evidence (cite-or-refuse) → generate → guard_output`

## Package layout

```text
src/clinical_rag/
  api/           FastAPI surface
  auth/          JWT + demo token
  kafka/         produce / consume / memory bus
  fhir/          converter + SQL store
  iodineiq/      CDS engine (risk + CDI + DRG + priority)
  ml/            Pandas / sklearn risk model
  graph/         Neo4j clinical knowledge graph
  retrieval/     Patient-scoped hybrid BM25 + pgvector dense (RRF)
  agent/         LangGraph GraphRAG workflow
  summaries/     Patient + insurance orchestration
  pipeline/      End-to-end ingest path (indexes Neo4j + pgvector)
  workers/       Kafka consumer process
```

## Failure modes & resilience

- If Kafka is down, `/v1/ingest` processes synchronously (or memory bus for workers).
- If Neo4j is down, graph store falls back to in-memory chunks (process-local).
- If LLM key is absent, summaries use evidence-grounded templates so demos still run.

## Security notes for production

- Replace demo JWT secret; use Cognito / Auth0 / Okta.
- Enforce TLS; encrypt Postgres (KMS) and Neo4j at rest.
- PHI: no logs of free-text notes; tokenize MRNs; audit every FHIR read.
- Network: private subnets for data plane, ALB only on API.
- Retrieval: mandatory `patient_id` predicate before any ranking/ANN.

## Interview prep

See [interview.md](../interview.md) for architecture Q&A, system design, and hard questions.
See [PRODUCTION_CHECKLIST.md](PRODUCTION_CHECKLIST.md) for hospital-readiness gaps (AWS deferred).
