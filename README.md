# ClinicalRAG

AI/ML **clinical decision support (CDS)** platform that integrates:

- **Predictive risk models** (Python, Pandas, Scikit-learn) for patient risk scoring and intervention hints
- **Deterministic rule engines** for CDI documentation workflows and DRG/coding candidates
- **Patient-scoped GraphRAG** (Neo4j + LangGraph) for grounded, citeable summaries for care providers and payers
- **Kafka → FHIR R4 → Postgres** ingestion with a secure FHIR API
- AWS-ready Docker + Terraform sketch + CI/CD

Aligned to the Emids CDS architecture narrative: classical ML + rules for scores/flags; GraphRAG for EHR narrative with **data isolation** by `patient_id`.

## Architecture

```text
Medical devices / EMR
        │
        ▼
   Kafka (raw events)
        │
        ▼
 FHIR Processing Engine ──► PostgreSQL (FHIR resources)
        │                         │
        ▼                         ▼
  CDS ML + CDI/DRG rules    Secure FHIR API (JWT)
        │                         │
        └──────────► Neo4j clinical knowledge graph
                         │
                         ▼
              LangGraph GraphRAG (patient-scoped)
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
   Provider summary          Payer / insurance summary
                    (+ citations, no cross-patient retrieval)
```

## Design principles (interview anchors)

| Concern | Approach |
|---------|----------|
| Risk / interventions | Sklearn GBM on vitals + comorbidity features |
| CDI / DRG / coding | Lexicon + deterministic rules (swappable for enterprise engines) |
| Summarization | GraphRAG: patient-scoped **BM25 + pgvector dense RRF** + graph neighborhood |
| Isolation | `patient_id` filter on every retrieve; separate provider vs payer prompts |
| Resilience | Template fallback if LLM unavailable; sync ingest if Kafka down |

## Quick start (local, minimal — no Kafka required)

```bash
cd clinical-rag
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
copy .env.example .env   # or: cp .env.example .env

python scripts/seed_demo.py

```

Open http://localhost:8000/docs

Use Authorization header: `Bearer demo-token`

```bash
curl -X POST http://localhost:8000/v1/summaries/patient ^
  -H "Authorization: Bearer demo-token" ^
  -H "Content-Type: application/json" ^
  -d "{\"patient_id\":\"Patient/demo-001\"}"

curl -X POST http://localhost:8000/v1/summaries/insurance ^
  -H "Authorization: Bearer demo-token" ^
  -H "Content-Type: application/json" ^
  -d "{\"patient_id\":\"Patient/demo-001\",\"encounter_id\":\"Encounter/enc-001\"}"
```

## Full stack (Docker Compose)

```bash
copy .env.example .env
docker compose up --build -d
docker compose exec api python scripts/seed_demo.py
```

| Component | URL |
|-----------|-----|
| API / Swagger | http://localhost:8000/docs |
| Kafka UI | http://localhost:8080 |
| Neo4j Browser | http://localhost:7474 |
| Prometheus | http://localhost:9090 |

## CDS modules

| Module | Endpoint / API | Purpose |
|--------|----------------|---------|
| Ingestion | `/v1/ingest`, Kafka worker | Continuous EMR/device read + clean |
| Analyze | `/v1/iodineiq/analyze` | Risk, ICD candidates, CDI flags, DRG, priority |
| CDI | included in analyze | Documentation gap flags / queries |
| Concurrent priority | `priority_score` | Live chart prioritization for capacity |
| DRG forecast | `drg` predictions | Coding-oriented DRG candidates |
| Interact | `/v1/iodineiq/interact` | Clinician Q&A over CDS outputs |
| Provider summary | `/v1/summaries/patient` | GraphRAG clinician narrative + citations |
| Payer summary | `/v1/summaries/insurance` | GraphRAG insurance narrative + citations |

Predictive risk models: `clinical_rag.ml.risk_model` (`GradientBoostingClassifier`).  
GraphRAG agent: `clinical_rag.agent.graph` (LangGraph).  
Interview prep: **[interview.md](interview.md)**.  
Production readiness: **[docs/PRODUCTION_CHECKLIST.md](docs/PRODUCTION_CHECKLIST.md)**.  
RAG eval: `python scripts/eval_rag.py` (after `seed_demo.py`).

## Secure FHIR API

- `GET /fhir/metadata`
- `GET /fhir/{type}/{id}`
- `GET /fhir/{type}?patient=...`
- `GET /fhir/Patient/{id}/$everything`

All routes require `Authorization: Bearer <jwt|demo-token>`.

## AWS deployment

See [docs/AWS_DEPLOYMENT.md](docs/AWS_DEPLOYMENT.md) and [infra/terraform](infra/terraform).

## Disclaimer

Demo software for architecture / portfolio use. Not a medical device. Do not use with real PHI without BAA, encryption, audit logging, and clinical validation.
